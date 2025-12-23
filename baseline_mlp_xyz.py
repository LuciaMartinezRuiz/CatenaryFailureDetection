"""
baseline_mlp_xyz.py  — FAST

Baseline rápido: MLP per-point con XYZ normalizado (solo Urban).
Añadidos:
  - Lectura con PROGRESO por tile.
  - Cap de puntos por tile ANTES de normalizar (stride), para ir rápido.
  - Downsample por voxel (solo train/val).
  - En test NO se toca el número de puntos (1 etiqueta por punto).

Uso recomendado para primer smoke test:
  python baseline_mlp_xyz.py --base data/raw/WHU-Railway3D \
    --train_limit 1 --epochs 1 --batch_points 32768 \
    --sample_per_epoch 100000 --voxel 0.30 --max_points_per_tile 1000000
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from plyfile import PlyData

# --------------------- utils de IO ---------------------
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

def read_ply_xyz_label(ply_path: Path, max_points: Optional[int]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Lee un .ply y devuelve xyz, y. Si max_points no es None, hace stride antes de normalizar."""
    ply = PlyData.read(str(ply_path))
    v = ply['vertex'].data
    # stride EARLY para no comerse memoria/tiempo con tiles gigantes
    n_total = len(v)
    if (max_points is not None) and (n_total > max_points):
        stride = int(np.ceil(n_total / max_points))
        v = v[::stride]
        print(f"   · stride 1/{stride}  ({n_total:,} -> {len(v):,} puntos)")
    else:
        print(f"   · sin stride  ({n_total:,} puntos)")

    xyz = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float32)
    key = _find_label_key(v.dtype.names)
    y = np.asarray(v[key], dtype=np.int64) if key is not None else None
    return xyz, y

def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    med = np.median(xyz, axis=0, keepdims=True)
    q1 = np.percentile(xyz, 25, axis=0, keepdims=True)
    q3 = np.percentile(xyz, 75, axis=0, keepdims=True)
    iqr = np.maximum(1e-6, (q3 - q1))
    x = (xyz - med)/iqr
    return x.astype(np.float32)

def voxel_downsample(xyz: np.ndarray, labels: Optional[np.ndarray], voxel_size: float):
    """Downsample para acelerar (solo train/val)."""
    if voxel_size is None or voxel_size <= 0:
        return xyz, labels
    mins = xyz.min(axis=0, keepdims=True)
    q = np.floor((xyz - mins) / voxel_size).astype(np.int64)
    key = q[:,0] * 73856093 ^ q[:,1] * 19349663 ^ q[:,2] * 83492791
    _, idx = np.unique(key, return_index=True)
    xyz_ds = xyz[idx]
    labels_ds = labels[idx] if labels is not None else None
    return xyz_ds, labels_ds

# --------------------- dataset ---------------------
class TilesDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir: Path, split: str, list_file: Optional[Path]=None,
                 limit_tiles: Optional[int]=None, voxel_size: float=0.1,
                 sample_per_epoch: int=200_000, max_points_per_tile: Optional[int]=1_000_000):
        """
        max_points_per_tile: cap duro con stride en LECTURA (solo train/val).
        En test NO se aplica para mantener #puntos==#etiquetas.
        """
        assert split in ("train","val","test")
        self.base_dir = base_dir
        self.split = split
        self.max_points_per_tile = None if split=="test" else max_points_per_tile

        tiles_dir = base_dir / "Urban" / "tiles" / ("train" if split!="test" else "test")
        if list_file is None:
            list_file = (base_dir / "Urban" / "lists" /
                         ("test.txt" if split=="test" else "train.txt"))
        names = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if limit_tiles is not None:
            names = names[:limit_tiles]
        self.paths = [tiles_dir / n for n in names]
        if split == "val" and len(self.paths) > 0:
            self.paths = self.paths[-1:]  # 1 tile como val
        self.voxel = voxel_size
        self.sample_per_epoch = sample_per_epoch

        print(f"[{split}] cargando {len(self.paths)} tile(s)…")
        self.per_tile = []
        for p in self.paths:
            print(f" - Leyendo {p.name}")
            xyz, y = read_ply_xyz_label(p, max_points=self.max_points_per_tile)
            if split != "test":
                xyz, y = voxel_downsample(xyz, y, self.voxel)
                print(f"   · voxel {self.voxel:.2f} -> {len(xyz):,} puntos")
            else:
                print(f"   · test: sin voxel (mantener 1:1) -> {len(xyz):,} puntos")
            xyz = normalize_xyz(xyz)
            self.per_tile.append((xyz, y))

        labels_all = np.concatenate([t[1] for t in self.per_tile if t[1] is not None]) if split!="test" else np.array([])
        self.num_classes = int(labels_all.max()+1) if labels_all.size>0 else 11

        self.idx_by_class = None
        if split != "test":
            self.idx_by_class = []
            for c in range(self.num_classes):
                idxs = []
                for ti, (xyz, y) in enumerate(self.per_tile):
                    if y is None: continue
                    found = np.where(y==c)[0]
                    if found.size:
                        idxs.extend([(ti, int(i)) for i in found.tolist()])
                self.idx_by_class.append(idxs)

    def __len__(self):
        return max(1, self.sample_per_epoch // 4096)

    def __getitem__(self, i):
        if self.split == "test":
            ti = i % len(self.per_tile)
            xyz, y = self.per_tile[ti]
            return {"xyz": torch.from_numpy(xyz), "y": None, "tile_index": ti}
        pcs, ys = [], []
        target = 4096
        c = 0
        while len(pcs) < target and c < target*3:
            c += 1
            sizes = np.array([len(ix) for ix in self.idx_by_class], dtype=np.float32)
            if sizes.sum() == 0:
                ti = np.random.randint(0, len(self.per_tile))
                xyz, y = self.per_tile[ti]
                idx = np.random.randint(0, len(xyz))
                pcs.append(xyz[idx]); ys.append(y[idx])
                continue
            probs = (sizes + 1e-6)**-1
            probs /= probs.sum()
            cls = int(np.random.choice(len(self.idx_by_class), p=probs))
            if len(self.idx_by_class[cls]) == 0:
                continue
            ti, li = self.idx_by_class[cls][np.random.randint(0, len(self.idx_by_class[cls]))]
            xyz, y = self.per_tile[ti]
            pcs.append(xyz[li]); ys.append(y[li])
        xyzb = torch.from_numpy(np.asarray(pcs, dtype=np.float32))
        yb   = torch.from_numpy(np.asarray(ys, dtype=np.int64))
        return {"xyz": xyzb, "y": yb}

# --------------------- modelo ---------------------
class MLPPerPoint(nn.Module):
    def __init__(self, in_dim=3, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(True),
            nn.Linear(128, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# --------------------- métricas ---------------------
def confusion_matrix(pred, gt, n):
    m = np.zeros((n,n), dtype=np.int64)
    for p,g in zip(pred, gt):
        if g<0 or g>=n: continue
        m[g,p]+=1
    return m

def compute_metrics(C):
    oa = np.trace(C)/max(1, C.sum())
    ious=[]
    for i in range(C.shape[0]):
        tp=C[i,i]; fp=C[:,i].sum()-tp; fn=C[i,:].sum()-tp
        den=tp+fp+fn
        ious.append(float(tp)/den if den>0 else 0.0)
    miou = float(np.mean(ious)) if ious else 0.0
    return oa, ious, miou

# --------------------- train / infer ---------------------
def train_loop(args):
    base = Path(args.base).expanduser().resolve()
    ds_train = TilesDataset(base, "train", limit_tiles=args.train_limit,
                            voxel_size=args.voxel, sample_per_epoch=args.sample_per_epoch,
                            max_points_per_tile=args.max_points_per_tile)
    ds_val   = TilesDataset(base, "val", limit_tiles=args.train_limit,
                            voxel_size=args.voxel, sample_per_epoch=max(4096, args.batch_points),
                            max_points_per_tile=args.max_points_per_tile)
    ncls = ds_train.num_classes
    model = MLPPerPoint(in_dim=3, num_classes=ncls)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # pesos por clase (inv. frecuencia)
    tmp = []
    for (xyz,y) in ds_train.per_tile:
        if y is None: continue
        tmp.append(y)
    y_all = np.concatenate(tmp)
    hist = np.bincount(y_all, minlength=ncls).astype(np.float32)
    w = (hist + 1e-3)**-1; w/=w.sum(); w = torch.tensor(w, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for ep in range(args.epochs):
        losses=[]; C = np.zeros((ncls,ncls), dtype=np.int64)
        iters = max(1, args.sample_per_epoch // args.batch_points)
        for _ in range(iters):
            Xs=[]; Ys=[]
            while sum([len(x) for x in Xs]) < args.batch_points:
                b = ds_train[np.random.randint(0, len(ds_train))]
                Xs.append(b["xyz"]); Ys.append(b["y"])
            x = torch.cat(Xs, dim=0).to(device)
            y = torch.cat(Ys, dim=0).to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
            with torch.no_grad():
                pred = logits.argmax(1).cpu().numpy()
                C += confusion_matrix(pred, y.cpu().numpy(), ncls)
        oa, ious, miou = compute_metrics(C)
        print(f"[ep {ep+1:02d}] loss={np.mean(losses):.4f} OA={oa:.3f} mIoU={miou:.3f}")

    ckpt_path = Path("outputs/checkpoints/mlp_xyz.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_classes": ncls}, ckpt_path)
    print(f"[OK] Guardado checkpoint en {ckpt_path}")

def infer_loop(args):
    base = Path(args.base).expanduser().resolve()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ncls = int(ckpt.get("num_classes", 11))
    model = MLPPerPoint(in_dim=3, num_classes=ncls)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # TEST: sin stride ni voxel para mantener 1:1
    ds_test = TilesDataset(base, "test", voxel_size=0.0, limit_tiles=args.test_limit,
                           sample_per_epoch=1, max_points_per_tile=None)
    out_dir = Path("outputs/submissions/urban")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ti, (xyz, y) in enumerate(ds_test.per_tile):
        B = 200_000
        preds = []
        for s in range(0, len(xyz), B):
            chunk = torch.from_numpy(xyz[s:s+B]).to(device)
            with torch.no_grad():
                logits = model(chunk)
                p = logits.argmax(1).cpu().numpy().astype(np.uint8)
            preds.append(p)
        pred_all = np.concatenate(preds, axis=0)
        npy_name = ds_test.paths[ti].with_suffix(".npy").name
        np.save(out_dir / npy_name, pred_all)
        print(f"[OK] Escrito {out_dir / npy_name}  ({len(pred_all)} etiquetas)")
    print("[DONE] Si quieres evaluar, comprime estos .npy en un zip.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="Ruta base WHU-Railway3D (que contiene Urban)")
    ap.add_argument("--mode", type=str, default="train", choices=["train","infer"])
    ap.add_argument("--ckpt", type=str, default="outputs/checkpoints/mlp_xyz.pt")
    ap.add_argument("--train_limit", type=int, default=2, help="limitar a N tiles de train")
    ap.add_argument("--test_limit", type=int, default=None, help="limitar a N tiles de test")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_points", type=int, default=65536)
    ap.add_argument("--sample_per_epoch", type=int, default=400_000)
    ap.add_argument("--voxel", type=float, default=0.1, help="downsample en metros aprox (solo train/val)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_points_per_tile", type=int, default=1_000_000,
                    help="cap de puntos por tile con stride (solo train/val)")
    args = ap.parse_args()
    if args.mode == "train":
        train_loop(args)
    else:
        infer_loop(args)

if __name__ == "__main__":
    main()
