"""
pointnet_seg.py — PointNet segmentation (no external deps)

Estructura esperada:
  <dataset_dir>/train/*.ply
  <dataset_dir>/test/*.ply

Features:
  - XYZ o XYZ+Intensity (flag --use_intensity)
  - stride cap antes de voxel para controlar memoria
  - voxel downsample en train/val (en test se permite sin voxel)
  - normalización robusta (mediana/IQR)
  - sampling ponderado por clase (para combatir desbalance)
  - augmentations opcionales (rotZ/scale/jitter) para train

Uso (smoke test):
  PYTHONUNBUFFERED=1 python -u pointnet_seg.py --dataset_dir data/raw/WHU-Railway3D/plateau_railway \
    --mode train --train_limit 10 --epochs 5 --use_intensity --voxel 0.15 --max_points_per_tile 1000000

Infer:
  PYTHONUNBUFFERED=1 python -u pointnet_seg.py --dataset_dir data/raw/WHU-Railway3D/plateau_railway \
    --mode infer --ckpt outputs/checkpoints/pointnet_plateau.pt
"""
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData


# --------------------- IO utils ---------------------
def _find_label_key(names) -> Optional[str]:
    for k in ["class", "label", "semantic", "classification", "category", "category_id", "seg_label"]:
        if k in names:
            return k
    return None


def normalize_feats(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0, keepdims=True)
    q1 = np.percentile(X, 25, axis=0, keepdims=True)
    q3 = np.percentile(X, 75, axis=0, keepdims=True)
    iqr = np.maximum(1e-6, (q3 - q1))
    return ((X - med) / iqr).astype(np.float32)


def voxel_downsample(X: np.ndarray, y: Optional[np.ndarray], voxel: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel is None or voxel <= 0:
        return X, y
    xyz = X[:, :3]
    mins = xyz.min(axis=0, keepdims=True)
    q = np.floor((xyz - mins) / voxel).astype(np.int64)
    key = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    _, idx = np.unique(key, return_index=True)
    X2 = X[idx]
    y2 = y[idx] if y is not None else None
    return X2, y2


def read_ply(
    ply_path: Path,
    max_points: Optional[int],
    use_intensity: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve:
      X_raw: (N,C) float32 [xyz o xyz+I] SIN normalizar
      y_raw: (N,) int64 (valores de clase originales)
      xyz_raw: (N,3) float32
    """
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    names = v.dtype.names

    key = _find_label_key(names)
    if key is None:
        raise SystemExit(f"No encuentro columna de etiquetas en {ply_path.name} (class/label/...)")

    n_total = len(v)
    if max_points is not None and n_total > max_points:
        stride = int(np.ceil(n_total / max_points))
        v = v[::stride]
        print(f"   · stride 1/{stride}  ({n_total:,} -> {len(v):,} puntos)")
    else:
        print(f"   · sin stride  ({n_total:,} puntos)")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    if use_intensity:
        if "intensity" in names:
            inten = np.asarray(v["intensity"], dtype=np.float32).reshape(-1, 1)
        else:
            inten = np.zeros((len(xyz), 1), dtype=np.float32)
        X = np.concatenate([xyz, inten], axis=1).astype(np.float32)
    else:
        X = xyz.astype(np.float32)

    y = np.asarray(v[key], dtype=np.int64)
    return X, y, xyz


# --------------------- Augmentations (train) ---------------------
def augment_xyz(X: np.ndarray, rot_z: bool = True, scale: bool = True, jitter: bool = True) -> np.ndarray:
    X = X.copy()
    xyz = X[:, :3]

    if rot_z:
        ang = np.random.uniform(0, 2 * np.pi)
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, -sa, 0.0],
                      [sa,  ca, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = xyz @ R.T

    if scale:
        s = np.random.uniform(0.95, 1.05)
        xyz = xyz * s

    if jitter:
        xyz = xyz + np.random.normal(0.0, 0.01, size=xyz.shape).astype(np.float32)

    X[:, :3] = xyz
    return X


# --------------------- Dataset ---------------------
class SampledPointCloudDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tiles: List[Tuple[np.ndarray, np.ndarray]],
        n_points: int,
        sample_per_epoch: int,
        class_sampling_power: float = 0.5,
        augment: bool = False
    ):
        """
        tiles: list of (X, y_idx) ya normalizados y con y reindexado 0..C-1
        class_sampling_power:
          0.0 => uniforme
          0.5 => 1/sqrt(freq)
          1.0 => 1/freq (más agresivo)
        """
        self.tiles = tiles
        self.n_points = int(n_points)
        self.sample_per_epoch = int(sample_per_epoch)
        self.augment = augment

        # construir distribución de muestreo por punto (más prob a clases raras)
        # p(point) ∝ 1/(freq[class]^power)
        all_y = np.concatenate([y for (_, y) in self.tiles], axis=0)
        ncls = int(all_y.max() + 1)
        hist = np.bincount(all_y, minlength=ncls).astype(np.float64)
        inv = 1.0 / np.maximum(1.0, hist) ** class_sampling_power

        self.tile_probs = []
        for (X, y) in self.tiles:
            w = inv[y].astype(np.float64)
            w = w / np.maximum(1e-12, w.sum())
            self.tile_probs.append(w)

    def __len__(self):
        return max(1, self.sample_per_epoch // self.n_points)

    def __getitem__(self, idx):
        ti = np.random.randint(0, len(self.tiles))
        X, y = self.tiles[ti]
        p = self.tile_probs[ti]
        # sample con replacement para estabilidad
        sel = np.random.choice(len(X), size=self.n_points, replace=True, p=p)
        Xs = X[sel]
        ys = y[sel]
        if self.augment:
            Xs = augment_xyz(Xs)
        return torch.from_numpy(Xs), torch.from_numpy(ys)


# --------------------- Model: PointNet Segmentation (simplificado) ---------------------
class PointNetSeg(nn.Module):
    def __init__(self, in_dim: int, ncls: int):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(256 + 256, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Linear(128, ncls)
        )

    def forward(self, x):
        """
        x: (B, N, in_dim)
        """
        B, N, D = x.shape
        # MLP por punto: necesitamos BN sobre (B*N, C)
        x1 = x.reshape(B * N, D)
        f = self.mlp1(x1)                 # (B*N, 256)
        f = f.reshape(B, N, 256)          # (B, N, 256)

        g = torch.max(f, dim=1).values    # (B, 256) global max
        g_expand = g.unsqueeze(1).expand(-1, N, -1)  # (B, N, 256)

        cat = torch.cat([f, g_expand], dim=-1)       # (B, N, 512)
        cat2 = cat.reshape(B * N, 512)
        out = self.mlp2(cat2)                        # (B*N, ncls)
        return out.reshape(B, N, -1)


# --------------------- Metrics ---------------------
def confusion_matrix(pred: np.ndarray, gt: np.ndarray, ncls: int) -> np.ndarray:
    C = np.zeros((ncls, ncls), dtype=np.int64)
    for p, g in zip(pred, gt):
        if 0 <= g < ncls and 0 <= p < ncls:
            C[g, p] += 1
    return C


def iou_from_conf(C: np.ndarray) -> np.ndarray:
    ious = []
    for i in range(C.shape[0]):
        tp = C[i, i]
        fp = C[:, i].sum() - tp
        fn = C[i, :].sum() - tp
        den = tp + fp + fn
        ious.append(float(tp) / den if den > 0 else 0.0)
    return np.asarray(ious, dtype=np.float64)


# --------------------- Train / Eval helpers ---------------------
@torch.no_grad()
def eval_on_tile_sample(model, X: np.ndarray, y: np.ndarray, ncls: int, device, eval_points: int, batch_points: int):
    model.eval()
    # sample fijo para evaluación rápida
    M = min(len(X), int(eval_points))
    sel = np.random.choice(len(X), size=M, replace=False) if M < len(X) else np.arange(len(X))
    Xs = X[sel]
    ys = y[sel]

    # procesar en bloques de N puntos (PointNet necesita nube, no puntos sueltos)
    N = int(batch_points)
    C = np.zeros((ncls, ncls), dtype=np.int64)

    for s in range(0, len(Xs), N):
        chunk = Xs[s:s + N]
        gtc = ys[s:s + N]
        if len(chunk) < N:
            # pad para completar el bloque
            pad = N - len(chunk)
            idx_pad = np.random.choice(len(chunk), size=pad, replace=True)
            chunk = np.concatenate([chunk, chunk[idx_pad]], axis=0)
            gtc = np.concatenate([gtc, gtc[idx_pad]], axis=0)

        xb = torch.from_numpy(chunk[None, ...]).to(device)  # (1, N, D)
        logits = model(xb)[0]  # (N, ncls)
        pred = logits.argmax(-1).cpu().numpy().astype(np.int64)

        C += confusion_matrix(pred, gtc.astype(np.int64), ncls)

    ious = iou_from_conf(C)
    oa = float(np.trace(C) / max(1, C.sum()))
    miou = float(np.mean(ious))
    present = (C.sum(axis=1) > 0)
    miou_present = float(np.mean(ious[present])) if present.any() else miou
    return oa, miou, miou_present, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/pointnet_plateau.pt")

    ap.add_argument("--train_limit", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--sample_per_epoch", type=int, default=800_000)

    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--max_points_per_tile", type=int, default=1_000_000)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use_intensity", action="store_true")
    ap.add_argument("--augment", action="store_true")

    ap.add_argument("--eval_points", type=int, default=200_000)
    ap.add_argument("--eval_block_points", type=int, default=8192)
    ap.add_argument("--class_sampling_power", type=float, default=0.5, help="0=uniforme, 0.5=1/sqrt(freq), 1=1/freq")

    args = ap.parse_args()
    ds_dir = Path(args.dataset_dir).expanduser().resolve()
    train_dir = ds_dir / "train"
    test_dir = ds_dir / "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    if args.mode == "train":
        files = sorted(train_dir.glob("*.ply"))
        if args.train_limit is not None:
            files = files[:args.train_limit]
        if len(files) < 2:
            raise SystemExit("Necesito al menos 2 tiles para separar train/val (usa --train_limit >=2)")

        # Cargamos tiles (X_raw,y_raw) -> cap/voxel/normalize, pero y aún sin reindex
        tiles_raw = []
        label_set = set()
        print(f"[load] train_limit={len(files)} tiles")
        for p in files:
            print(f" - Leyendo {p.name}")
            X, y_raw, _ = read_ply(p, max_points=args.max_points_per_tile, use_intensity=args.use_intensity)
            X, y_raw = voxel_downsample(X, y_raw, args.voxel)
            print(f"   · voxel {args.voxel:.2f} -> {len(X):,} puntos")
            X = normalize_feats(X)
            tiles_raw.append((X, y_raw))
            label_set.update(np.unique(y_raw).tolist())

        label_values = np.array(sorted(list(label_set)), dtype=np.int64)
        ncls = int(len(label_values))
        in_dim = int(4 if args.use_intensity else 3)

        # reindex y a 0..ncls-1
        tiles = []
        for (X, y_raw) in tiles_raw:
            y_idx = np.searchsorted(label_values, y_raw).astype(np.int64)
            tiles.append((X, y_idx))

        # split: último tile para val
        train_tiles = tiles[:-1]
        val_tile = tiles[-1]

        model = PointNetSeg(in_dim=in_dim, ncls=ncls).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # class weights (suaves) para CE: 1/sqrt(freq)
        all_y = np.concatenate([y for (_, y) in train_tiles], axis=0)
        hist = np.bincount(all_y, minlength=ncls).astype(np.float32)
        w = 1.0 / np.sqrt(hist + 1.0)   # suave
        w = w / w.mean()
        w_t = torch.tensor(w, dtype=torch.float32, device=device)

        ds = SampledPointCloudDataset(
            train_tiles,
            n_points=args.n_points,
            sample_per_epoch=args.sample_per_epoch,
            class_sampling_power=args.class_sampling_power,
            augment=args.augment
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        print(f"[train] ncls={ncls} labels={label_values.tolist()} in_dim={in_dim}")
        for ep in range(args.epochs):
            model.train()
            losses = []
            for Xb, yb in dl:
                Xb = Xb.to(device)  # (B,N,D)
                yb = yb.to(device)  # (B,N)
                logits = model(Xb)  # (B,N,C)
                loss = F.cross_entropy(logits.reshape(-1, ncls), yb.reshape(-1), weight=w_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            # eval rápido en val tile
            Xv, yv = val_tile
            oa, miou, miou_p, _ = eval_on_tile_sample(
                model, Xv, yv, ncls=ncls, device=device,
                eval_points=args.eval_points, batch_points=args.eval_block_points
            )
            print(f"[ep {ep+1:03d}] loss={np.mean(losses):.4f} val_OA={oa:.3f} val_mIoU={miou:.3f} val_mIoU_present={miou_p:.3f}")

        ckpt_path = Path(args.ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "num_classes": ncls,
            "in_dim": in_dim,
            "label_values": label_values.tolist(),
            "voxel": args.voxel,
            "max_points_per_tile": args.max_points_per_tile,
        }, ckpt_path)
        print(f"[OK] Guardado checkpoint en {ckpt_path}")

    else:
        # Inferencia por chunks (aprox) en test tiles
        ckpt = torch.load(args.ckpt, map_location="cpu")
        ncls = int(ckpt["num_classes"])
        in_dim = int(ckpt.get("in_dim", 3))
        label_values = np.asarray(ckpt.get("label_values", list(range(ncls))), dtype=np.int64)

        model = PointNetSeg(in_dim=in_dim, ncls=ncls).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        out_dir = Path("outputs/predictions/plateau_pointnet")
        out_dir.mkdir(parents=True, exist_ok=True)

        test_files = sorted(test_dir.glob("*.ply"))
        print(f"[infer] {len(test_files)} tiles")
        for p in test_files:
            print(f" - Leyendo {p.name}")
            X, y_raw, _ = read_ply(p, max_points=None, use_intensity=(in_dim == 4))
            # en test NO aplico voxel para mantener puntos (pero ojo: es muy grande)
            Xn = normalize_feats(X)

            N = 8192
            preds = np.zeros((len(Xn),), dtype=np.int64)

            for s in range(0, len(Xn), N):
                chunk = Xn[s:s+N]
                if len(chunk) < N:
                    pad = N - len(chunk)
                    idx_pad = np.random.choice(len(chunk), size=pad, replace=True)
                    chunk2 = np.concatenate([chunk, chunk[idx_pad]], axis=0)
                    xb = torch.from_numpy(chunk2[None, ...]).to(device)
                    with torch.no_grad():
                        pc = model(xb)[0].argmax(-1).cpu().numpy().astype(np.int64)
                    preds[s:s+len(chunk)] = pc[:len(chunk)]
                else:
                    xb = torch.from_numpy(chunk[None, ...]).to(device)
                    with torch.no_grad():
                        pc = model(xb)[0].argmax(-1).cpu().numpy().astype(np.int64)
                    preds[s:s+N] = pc

            # guardar en labels originales (si quieres)
            pred_labels = label_values[preds]
            np.save(out_dir / (p.stem + ".npy"), pred_labels.astype(np.int64))
            print(f"   [OK] {out_dir/(p.stem + '.npy')}  ({len(pred_labels):,} pred)")

        print("[DONE] Predicciones guardadas para análisis.")

if __name__ == "__main__":
    main()

