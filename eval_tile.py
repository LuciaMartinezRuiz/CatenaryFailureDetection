import argparse
import numpy as np
import torch
from plyfile import PlyData

def _find_label_key(names):
    for k in ["class","label","semantic","classification","category","category_id","seg_label"]:
        if k in names:
            return k
    return None

def normalize_feats(X):
    med = np.median(X, axis=0, keepdims=True)
    q1 = np.percentile(X, 25, axis=0, keepdims=True)
    q3 = np.percentile(X, 75, axis=0, keepdims=True)
    iqr = np.maximum(1e-6, (q3 - q1))
    return ((X - med) / iqr).astype(np.float32)

def voxel_downsample_xyz(X, y, voxel):
    if voxel <= 0:
        return X, y
    xyz = X[:, :3]
    mins = xyz.min(axis=0, keepdims=True)
    q = np.floor((xyz - mins) / voxel).astype(np.int64)
    key = q[:,0] * 73856093 ^ q[:,1] * 19349663 ^ q[:,2] * 83492791
    _, idx = np.unique(key, return_index=True)
    return X[idx], y[idx]

def confusion(pred, gt, n):
    C = np.zeros((n, n), dtype=np.int64)
    for p, g in zip(pred, gt):
        if 0 <= g < n and 0 <= p < n:
            C[g, p] += 1
    return C

def iou_from_conf(C):
    ious = []
    for i in range(C.shape[0]):
        tp = C[i, i]
        fp = C[:, i].sum() - tp
        fn = C[i, :].sum() - tp
        den = tp + fp + fn
        ious.append(tp / den if den > 0 else 0.0)
    return np.array(ious, dtype=np.float64)

class MLP(torch.nn.Module):
    def __init__(self, in_dim, ncls):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128), torch.nn.ReLU(True),
            torch.nn.Linear(128, 128), torch.nn.ReLU(True),
            torch.nn.Linear(128, 64),  torch.nn.ReLU(True),
            torch.nn.Linear(64, ncls),
        )
    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--voxel", type=float, default=0.30, help="downsample para evaluar rápido")
    ap.add_argument("--max_points", type=int, default=1_000_000, help="cap con stride antes de voxel")
    ap.add_argument("--use_intensity", action="store_true",
                    help="Fuerza usar intensity si existe. Si el ckpt tiene in_dim=4, se activa solo.")
    ap.add_argument("--present_min_support", type=int, default=1,
                    help="umbral para mIoU_present (solo clases con support >= este valor)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ncls = int(ckpt["num_classes"])
    in_dim = int(ckpt.get("in_dim", 3))
    label_values = ckpt.get("label_values", None)
    if label_values is not None:
        label_values = np.asarray(label_values, dtype=np.int64)

    # Auto: si el modelo fue entrenado con intensity (in_dim==4), lo activamos sí o sí
    use_intensity = args.use_intensity or (in_dim == 4)

    model = MLP(in_dim, ncls)
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

    if use_intensity:
        if "intensity" in names:
            inten = np.asarray(v["intensity"], dtype=np.float32).reshape(-1, 1)
        else:
            inten = np.zeros((len(xyz), 1), dtype=np.float32)
        X = np.concatenate([xyz, inten], axis=1).astype(np.float32)
    else:
        X = xyz

    y_raw = np.asarray(v[key], dtype=np.int64)
    if label_values is not None:
        # map raw labels -> [0..ncls-1] usando label_values
        y = np.searchsorted(label_values, y_raw).astype(np.int64)
    else:
        y = y_raw

    X, y = voxel_downsample_xyz(X, y, args.voxel)
    X = normalize_feats(X)

    # infer por chunks
    B = 200_000
    preds = []
    for s in range(0, len(X), B):
        xb = torch.from_numpy(X[s:s+B]).to(device)
        with torch.no_grad():
            p = model(xb).argmax(1).cpu().numpy()
        preds.append(p)
    pred = np.concatenate(preds)

    C = confusion(pred, y, ncls)
    ious = iou_from_conf(C)
    oa = np.trace(C) / max(1, C.sum())
    miou = float(np.mean(ious))

    supports = np.array([C[i, :].sum() for i in range(ncls)], dtype=np.int64)
    present = supports >= int(args.present_min_support)
    miou_present = float(np.mean(ious[present])) if np.any(present) else 0.0

    print(f"Eval points: {len(X):,} | OA={oa:.3f} | mIoU={miou:.3f} | mIoU_present(s>={args.present_min_support})={miou_present:.3f}")
    for i, val in enumerate(label_values if label_values is not None else range(ncls)):
        print(f"  class {int(val):>3}  IoU={ious[i]:.3f}  support={int(supports[i]):,}")

if __name__ == "__main__":
    main()

