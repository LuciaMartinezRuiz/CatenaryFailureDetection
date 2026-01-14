import argparse
import warnings
import numpy as np
import torch
from plyfile import PlyData

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch\.load` with `weights_only=False`.*",
)

from pointnet2_lite_seg import PointNet2LiteSeg

def _find_label_key(names):
    for k in ["class", "label", "semantic", "classification", "category", "category_id", "seg_label"]:
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
    key = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
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

def miou_present(ious, supports):
    mask = supports >= 1
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(ious[mask]))

@torch.no_grad()
def infer_blocks(model, X, block_points=1024, seed=0, device="cpu"):
    """
    Predice cada punto 1 vez (sin solape) para estabilidad:
      - permutamos indices
      - partimos en bloques de tamaño block_points
    """
    rs = np.random.RandomState(seed)
    N = len(X)
    idx = rs.permutation(N)

    pred = np.empty((N,), dtype=np.int64)

    for s in range(0, N, block_points):
        sel = idx[s:s+block_points]
        xb = X[sel]
        xb_t = torch.from_numpy(xb).unsqueeze(0).to(device)  # (1,B,in_dim)
        logits = model(xb_t)  # (1,B,ncls)
        p = logits.argmax(-1).squeeze(0).cpu().numpy()
        pred[sel] = p

    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--max_points", type=int, default=1000000)
    ap.add_argument("--block_points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)

    # True/False/auto
    ap.add_argument("--use_global", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--use_intensity", type=str, default="auto", choices=["auto", "true", "false"])

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    ncls = int(ckpt["num_classes"])
    in_dim = int(ckpt.get("in_dim", 4))

    label_values = ckpt.get("label_values", None)
    if label_values is not None:
        label_values = np.asarray(label_values, dtype=np.int64)

    ckpt_use_global = bool(ckpt.get("use_global", True))
    ckpt_use_intensity = bool(ckpt.get("use_intensity", (in_dim == 4)))

    if args.use_global == "auto":
        use_global = ckpt_use_global
    else:
        use_global = (args.use_global == "true")

    if args.use_intensity == "auto":
        use_intensity = ckpt_use_intensity
    else:
        use_intensity = (args.use_intensity == "true")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2LiteSeg(in_dim=in_dim, ncls=ncls, use_global=use_global).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[eval] model=PointNet2LiteSeg in_dim={in_dim} ncls={ncls} use_global={use_global} use_intensity={use_intensity}")

    ply = PlyData.read(args.ply)
    v = ply["vertex"].data
    names = v.dtype.names
    key = _find_label_key(names)
    if key is None:
        raise SystemExit("No encuentro columna de etiqueta (class/label/...)")

    # stride si es enorme
    n_total = len(v)
    if n_total > args.max_points:
        stride = int(np.ceil(n_total / args.max_points))
        v = v[::stride]
        print(f"stride 1/{stride}: {n_total:,} -> {len(v):,}")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    if use_intensity and in_dim == 4:
        if "intensity" in names:
            inten = np.asarray(v["intensity"], dtype=np.float32).reshape(-1, 1)
        else:
            inten = np.zeros((len(xyz), 1), np.float32)
        X = np.concatenate([xyz, inten], axis=1).astype(np.float32)
    else:
        X = xyz.astype(np.float32)

    y_raw = np.asarray(v[key], dtype=np.int64)
    if label_values is not None:
        y = np.searchsorted(label_values, y_raw).astype(np.int64)
    else:
        y = y_raw.astype(np.int64)

    X, y = voxel_downsample_xyz(X, y, args.voxel)
    X = normalize_feats(X)

    pred = infer_blocks(model, X, block_points=args.block_points, seed=args.seed, device=device)

    C = confusion(pred, y, ncls)
    ious = iou_from_conf(C)
    oa = float(np.trace(C) / max(1, C.sum()))
    miou = float(np.mean(ious))
    supports = np.array([C[i, :].sum() for i in range(ncls)], dtype=np.int64)
    miou_p = miou_present(ious, supports)

    print(f"Eval points: {len(X):,} | OA={oa:.3f} | mIoU={miou:.3f} | mIoU_present(s>=1)={miou_p:.3f}")
    for i, val in enumerate(label_values if label_values is not None else range(ncls)):
        print(f"  class {int(val):>3}  IoU={ious[i]:.3f}  support={int(supports[i]):,}")

if __name__ == "__main__":
    main()

