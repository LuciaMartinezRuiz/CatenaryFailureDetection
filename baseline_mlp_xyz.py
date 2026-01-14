"""
baseline_mlp_xyz.py — FAST (Plateau-friendly) — robusto

Estructura esperada:
  <dataset_dir>/train/*.ply
  <dataset_dir>/test/*.ply
  <dataset_dir>/lists/train.txt (opcional)
  <dataset_dir>/lists/test.txt  (opcional)

Cambios clave:
- SIN val (no relee el tile dos veces)
- Remapeo de labels a 0..K-1 (evita num_classes gigante)
- Sampler configurable: random (rápido) o balanced (más lento)

Uso (smoke test recomendado):
  PYTHONUNBUFFERED=1 python -u baseline_mlp_xyz.py --dataset_dir data/raw/WHU-Railway3D/plateau_railway \
    --mode train --train_limit 1 --epochs 1 --batch_points 32768 \
    --sample_per_epoch 100000 --voxel 0.30 --max_points_per_tile 1000000 --sampler random

Infer:
  PYTHONUNBUFFERED=1 python -u baseline_mlp_xyz.py --dataset_dir data/raw/WHU-Railway3D/plateau_railway --mode infer
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from plyfile import PlyData

# --------------------- utils ---------------------
def _find_label_key(names) -> Optional[str]:
    candidates = [
        "label","class","semantic","classification",
        "category","category_id","seg_label",
        "scalar_Label","scalar_label","Scalar_Label"
    ]
    for k in candidates:
        if k in names:
            return k
    return None

def read_ply_X_y(ply_path: Path, max_points: Optional[int], use_intensity: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data

    n_total = len(v)
    if (max_points is not None) and (n_total > max_points):
        stride = int(np.ceil(n_total / max_points))
        v = v[::stride]
        print(f"   · stride 1/{stride}  ({n_total:,} -> {len(v):,} puntos)")
    else:
        print(f"   · sin stride  ({n_total:,} puntos)")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    if use_intensity:
        inten = np.asarray(v["intensity"], dtype=np.float32).reshape(-1, 1) if "intensity" in v.dtype.names else np.zeros((len(xyz), 1), dtype=np.float32)
        X = np.concatenate([xyz, inten], axis=1).astype(np.float32)
    else:
        X = xyz

    key = _find_label_key(v.dtype.names)
    y = np.asarray(v[key], dtype=np.int64) if key is not None else None
    return X, y

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
    key = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
    _, idx = np.unique(key, return_index=True)
    X2 = X[idx]
    y2 = y[idx] if y is not None else None
    return X2, y2

def load_list_or_glob(dir_path: Path, list_path: Optional[Path]) -> List[Path]:
    if list_path is not None and list_path.exists():
        names = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return [dir_path / n for n in names]
    return sorted(dir_path.glob("*.ply"))

# --------------------- dataset ---------------------
class TilesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        limit_tiles: Optional[int],
        voxel: float,
        sample_per_epoch: int,
        max_points_per_tile: Optional[int],
        use_intensity: bool,
        sampler: str
    ):
        assert split in ("train", "test")
        assert sampler in ("random", "balanced")
        self.split = split
        self.sampler = sampler
        self.sample_per_epoch = sample_per_epoch

        train_dir = dataset_dir / "train"
        test_dir  = dataset_dir / "test"
        lists_dir = dataset_dir / "lists"
        if split == "train":
            tiles_dir = train_dir
            list_file = lists_dir / "train.txt"
        else:
            tiles_dir = test_dir
            list_file = lists_dir / "test.txt"

        self.paths = load_list_or_glob(tiles_dir, list_file)
        if limit_tiles is not None:
            self.paths = self.paths[:limit_tiles]

        # En test: mantener 1:1 => sin stride y sin voxel
        mp = None if split == "test" else max_points_per_tile
        vx = 0.0 if split == "test" else voxel

        print(f"[{split}] cargando {len(self.paths)} tile(s)…")
        self.per_tile: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
        for p in self.paths:
            print(f" - Leyendo {p.name}")
            X, y = read_ply_X_y(p, mp, use_intensity)
            if split != "test":
                X, y = voxel_downsample(X, y, vx)
                print(f"   · voxel {vx:.2f} -> {len(X):,} puntos")
            else:
                print(f"   · test: sin voxel/stride -> {len(X):,} puntos")
            X = normalize_feats(X)
            self.per_tile.append((X, y))

        self.label_values = None  # para mapear pred -> etiqueta original

        if split == "train":
            # remapeo robusto a 0..K-1
            all_y = np.concatenate([y for (_, y) in self.per_tile if y is not None])
            uniq = np.unique(all_y)
            self.label_values = uniq.astype(np.int64)  # etiquetas originales ordenadas
            self.num_classes = int(len(self.label_values))

            for i, (X, y) in enumerate(self.per_tile):
                # vectorizado: y_mapped en 0..K-1
                y_m = np.searchsorted(self.label_values, y)
                self.per_tile[i] = (X, y_m.astype(np.int64))

            # preparar balanced indices SOLO si se pide
            self.idx_by_class = None
            if self.sampler == "balanced":
                self.idx_by_class = [[] for _ in range(self.num_classes)]
                for ti, (X, y) in enumerate(self.per_tile):
                    order = np.argsort(y)
                    ys = y[order]
                    # boundaries por cambios de clase
                    cuts = np.flatnonzero(np.r_[True, ys[1:] != ys[:-1], True])
                    for a, b in zip(cuts[:-1], cuts[1:]):
                        c = int(ys[a])
                        idxs = order[a:b]
                        self.idx_by_class[c].extend([(ti, int(j)) for j in idxs.tolist()])
        else:
            self.num_classes = 11
            self.idx_by_class = None

    def __len__(self):
        return max(1, self.sample_per_epoch // 4096)

    def __getitem__(self, i):
        if self.split == "test":
            ti = i % len(self.per_tile)
            X, _ = self.per_tile[ti]
            return {"x": torch.from_numpy(X), "y": None, "tile_index": ti}

        # TRAIN
        if self.sampler == "random":
            ti = np.random.randint(0, len(self.per_tile))
            X, y = self.per_tile[ti]
            idx = np.random.randint(0, len(X), size=4096)
            xb = torch.from_numpy(X[idx].astype(np.float32))
            yb = torch.from_numpy(y[idx].astype(np.int64))
            return {"x": xb, "y": yb}

        # balanced
        pcs, ys = [], []
        target = 4096
        c = 0
        while len(pcs) < target and c < target * 3:
            c += 1
            sizes = np.array([len(ix) for ix in self.idx_by_class], dtype=np.float32)
            if sizes.sum() == 0:
                ti = np.random.randint(0, len(self.per_tile))
                X, y = self.per_tile[ti]
                idx = np.random.randint(0, len(X))
                pcs.append(X[idx]); ys.append(y[idx])
                continue

            probs = (sizes + 1e-6) ** -1
            probs /= probs.sum()
            cls = int(np.random.choice(len(self.idx_by_class), p=probs))
            if len(self.idx_by_class[cls]) == 0:
                continue
            ti, li = self.idx_by_class[cls][np.random.randint(0, len(self.idx_by_class[cls]))]
            X, y = self.per_tile[ti]
            pcs.append(X[li]); ys.append(y[li])

        xb = torch.from_numpy(np.asarray(pcs, dtype=np.float32))
        yb = torch.from_numpy(np.asarray(ys, dtype=np.int64))
        return {"x": xb, "y": yb}

# --------------------- model ---------------------
class MLPPerPoint(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# --------------------- metrics ---------------------
def confusion_matrix(pred, gt, n):
    m = np.zeros((n, n), dtype=np.int64)
    for p, g in zip(pred, gt):
        if g < 0 or g >= n:
            continue
        m[g, p] += 1
    return m

def compute_metrics(C):
    oa = np.trace(C) / max(1, C.sum())
    ious = []
    for i in range(C.shape[0]):
        tp = C[i, i]
        fp = C[:, i].sum() - tp
        fn = C[i, :].sum() - tp
        den = tp + fp + fn
        ious.append(float(tp) / den if den > 0 else 0.0)
    miou = float(np.mean(ious)) if ious else 0.0
    return oa, miou

# --------------------- train / infer ---------------------
def train_loop(args):
    ds_dir = Path(args.dataset_dir).expanduser().resolve()
    ds_train = TilesDataset(
        ds_dir, "train",
        limit_tiles=args.train_limit,
        voxel=args.voxel,
        sample_per_epoch=args.sample_per_epoch,
        max_points_per_tile=args.max_points_per_tile,
        use_intensity=args.use_intensity,
        sampler=args.sampler
    )

    ncls = ds_train.num_classes
    in_dim = 4 if args.use_intensity else 3

    model = MLPPerPoint(in_dim=in_dim, num_classes=ncls)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # class weights opcional
    weight = None
    if args.class_weights:
        tmp = []
        for (_, y) in ds_train.per_tile:
            if y is not None:
                tmp.append(y)
        y_all = np.concatenate(tmp) if tmp else np.array([], dtype=np.int64)
        hist = np.bincount(y_all, minlength=ncls).astype(np.float32) if y_all.size else np.ones(ncls, dtype=np.float32)
        alpha = 0.5
        w = (hist + 1e-3) ** (-alpha)
        w = np.clip(w, 0, np.median(w) * 20)  # cap para evitar colapso
        w /= w.sum()
        weight = torch.tensor(w, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for ep in range(args.epochs):
        losses = []
        C = np.zeros((ncls, ncls), dtype=np.int64)
        iters = max(1, args.sample_per_epoch // args.batch_points)

        for _ in range(iters):
            Xs, Ys = [], []
            while sum([len(x) for x in Xs]) < args.batch_points:
                b = ds_train[np.random.randint(0, len(ds_train))]
                Xs.append(b["x"]); Ys.append(b["y"])
            x = torch.cat(Xs, dim=0).to(device)
            y = torch.cat(Ys, dim=0).to(device)

            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()

            losses.append(float(loss.item()))
            with torch.no_grad():
                pred = logits.argmax(1).cpu().numpy()
                C += confusion_matrix(pred, y.cpu().numpy(), ncls)

        oa, miou = compute_metrics(C)
        print(f"[ep {ep+1:02d}] loss={np.mean(losses):.4f} OA={oa:.3f} mIoU={miou:.3f}")

    ckpt_path = Path("outputs/checkpoints/mlp_plateau.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "num_classes": ncls,
        "in_dim": in_dim,
        "label_values": ds_train.label_values  # para desmapear en infer
    }, ckpt_path)
    print(f"[OK] Guardado checkpoint en {ckpt_path}")

def infer_loop(args):
    ds_dir = Path(args.dataset_dir).expanduser().resolve()
    ckpt = torch.load(args.ckpt, map_location="cpu")

    ncls = int(ckpt.get("num_classes", 11))
    in_dim = int(ckpt.get("in_dim", 3))
    label_values = ckpt.get("label_values", None)
    if label_values is not None:
        label_values = np.asarray(label_values, dtype=np.int64)

    model = MLPPerPoint(in_dim=in_dim, num_classes=ncls)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds_test = TilesDataset(
        ds_dir, "test",
        limit_tiles=args.test_limit,
        voxel=0.0,
        sample_per_epoch=1,
        max_points_per_tile=None,
        use_intensity=(in_dim == 4),
        sampler="random"
    )

    out_dir = Path("outputs/submissions/plateau_railway")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ti, (X, _) in enumerate(ds_test.per_tile):
        B = 200_000
        preds = []
        for s in range(0, len(X), B):
            chunk = torch.from_numpy(X[s:s+B]).to(device)
            with torch.no_grad():
                p = model(chunk).argmax(1).cpu().numpy().astype(np.int64)
            preds.append(p)

        pred_idx = np.concatenate(preds, axis=0)

        if label_values is not None and label_values.size == ncls:
            pred_lbl = label_values[pred_idx]
        else:
            pred_lbl = pred_idx

        # dtype razonable
        if pred_lbl.max() <= 255:
            pred_lbl = pred_lbl.astype(np.uint8)
        else:
            pred_lbl = pred_lbl.astype(np.uint16)

        npy_name = ds_test.paths[ti].with_suffix(".npy").name
        np.save(out_dir / npy_name, pred_lbl)
        print(f"[OK] Escrito {out_dir / npy_name}  ({len(pred_lbl)} etiquetas)")

    print("[DONE] Comprime outputs/submissions/plateau_railway/*.npy en un zip si necesitas entregar.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/mlp_plateau.pt")
    ap.add_argument("--train_limit", type=int, default=2)
    ap.add_argument("--test_limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_points", type=int, default=65536)
    ap.add_argument("--sample_per_epoch", type=int, default=400_000)
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_points_per_tile", type=int, default=1_000_000)
    ap.add_argument("--use_intensity", action="store_true")
    ap.add_argument("--sampler", type=str, default="random", choices=["random", "balanced"])
    ap.add_argument("--class_weights", action="store_true", help="Pondera CE por inv-freq (puede ser más lento)")
    args = ap.parse_args()

    if args.mode == "train":
        train_loop(args)
    else:
        infer_loop(args)

if __name__ == "__main__":
    main()

