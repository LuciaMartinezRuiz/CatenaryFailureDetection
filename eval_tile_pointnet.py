import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData

def _find_label_key(names):
    for k in ["class","label","semantic","classification","category","category_id","seg_label"]:
        if k in names: return k
    return None

def normalize_feats(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0, keepdims=True)
    q1 = np.percentile(X, 25, axis=0, keepdims=True)
    q3 = np.percentile(X, 75, axis=0, keepdims=True)
    iqr = np.maximum(1e-6, (q3 - q1))
    return ((X - med) / iqr).astype(np.float32)

def voxel_downsample_xyz(X, y, voxel: float):
    if voxel is None or voxel <= 0:
        return X, y
    xyz = X[:, :3]
    mins = xyz.min(axis=0, keepdims=True)
    q = np.floor((xyz - mins) / voxel).astype(np.int64)
    key = q[:,0] * 73856093 ^ q[:,1] * 19349663 ^ q[:,2] * 83492791
    _, idx = np.unique(key, return_index=True)
    return X[idx], y[idx]

def confusion(pred, gt, ncls):
    C = np.zeros((ncls,ncls), dtype=np.int64)
    for p,g in zip(pred, gt):
        if 0 <= g < ncls and 0 <= p < ncls:
            C[g,p] += 1
    return C

def iou_from_conf(C):
    ious=[]
    for i in range(C.shape[0]):
        tp=C[i,i]; fp=C[:,i].sum()-tp; fn=C[i,:].sum()-tp
        den=tp+fp+fn
        ious.append(tp/den if den>0 else 0.0)
    return np.array(ious, dtype=np.float64)

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
        # x: (B,N,D)
        B,N,D = x.shape
        x1 = x.reshape(B*N, D)
        f = self.mlp1(x1).reshape(B, N, 256)
        g = torch.max(f, dim=1).values                    # (B,256)
        g = g.unsqueeze(1).expand(-1, N, -1)             # (B,N,256)
        cat = torch.cat([f, g], dim=-1).reshape(B*N, 512)
        out = self.mlp2(cat).reshape(B, N, -1)
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--max_points", type=int, default=1_000_000)
    ap.add_argument("--block_points", type=int, default=8192, help="tamaño de bloque PointNet")
    ap.add_argument("--min_support", type=int, default=1)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ncls = int(ckpt["num_classes"])
    in_dim = int(ckpt.get("in_dim", 3))
    label_values = np.asarray(ckpt.get("label_values", list(range(ncls))), dtype=np.int64)

    model = PointNetSeg(in_dim=in_dim, ncls=ncls)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ply = PlyData.read(args.ply)
    v = ply["vertex"].data
    names = v.dtype.names
    key = _find_label_key(names)
    if key is None:
        raise SystemExit("No encuentro columna de etiqueta (class/label/...)")

    n_total = len(v)
    if n_total > args.max_points:
        stride = int(np.ceil(n_total / args.max_points))
        v = v[::stride]
        print(f"stride 1/{stride}: {n_total:,} -> {len(v):,}")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    if in_dim == 4:
        inten = np.asarray(v["intensity"], dtype=np.float32).reshape(-1,1) if "intensity" in names else np.zeros((len(xyz),1), np.float32)
        X = np.concatenate([xyz, inten], axis=1).astype(np.float32)
    else:
        X = xyz.astype(np.float32)

    y_raw = np.asarray(v[key], dtype=np.int64)
    y = np.searchsorted(label_values, y_raw).astype(np.int64)

    X, y = voxel_downsample_xyz(X, y, args.voxel)
    X = normalize_feats(X)

    N = int(args.block_points)
    preds = np.zeros((len(X),), dtype=np.int64)

    # procesar en bloques secuenciales (determinista)
    with torch.no_grad():
        for s in range(0, len(X), N):
            chunk = X[s:s+N]
            take = len(chunk)
            if take < N:
                pad = N - take
                chunk = np.concatenate([chunk, chunk[:pad]], axis=0)
            xb = torch.from_numpy(chunk[None, ...]).to(device)
            logits = model(xb)[0]  # (N,ncls)
            p = logits.argmax(-1).cpu().numpy().astype(np.int64)
            preds[s:s+take] = p[:take]

    C = confusion(preds, y, ncls)
    ious = iou_from_conf(C)
    oa = float(np.trace(C)/max(1, C.sum()))
    miou = float(np.mean(ious))

    supports = C.sum(axis=1)
    present = supports >= int(args.min_support)
    miou_present = float(np.mean(ious[present])) if present.any() else miou

    print(f"Eval points: {len(X):,} | OA={oa:.3f} | mIoU={miou:.3f} | mIoU_present(s>={args.min_support})={miou_present:.3f}")
    for i, lab in enumerate(label_values):
        print(f"  class {int(lab):>3}  IoU={ious[i]:.3f}  support={int(supports[i]):,}")

if __name__ == "__main__":
    main()

