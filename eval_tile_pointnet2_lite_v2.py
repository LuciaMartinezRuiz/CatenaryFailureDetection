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

# ---------- Utils ----------
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

def apply_stride_if_needed(X, y, max_points):
    n_total = len(X)
    if max_points is None or max_points <= 0 or n_total <= max_points:
        return X, y, 1, n_total, n_total
    stride = int(np.ceil(n_total / max_points))
    X2 = X[::stride]
    y2 = y[::stride]
    return X2, y2, stride, n_total, len(X2)

def voxel_downsample_xyz(X, y, voxel):
    if voxel is None or voxel <= 0:
        return X, y
    xyz = X[:, :3]
    mins = xyz.min(axis=0, keepdims=True)
    q = np.floor((xyz - mins) / voxel).astype(np.int64)
    key = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
    _, idx = np.unique(key, return_index=True)
    return X[idx], y[idx]

def read_ply_xyzI_and_labels(ply_path, use_intensity=True):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    names = v.dtype.names

    key = _find_label_key(names)
    if key is None:
        raise SystemExit(f"No encuentro columna etiqueta (class/label/...) en {ply_path}. Fields={names}")

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

    if use_intensity:
        if "intensity" in names:
            inten = np.asarray(v["intensity"], dtype=np.float32).reshape(-1, 1)
        else:
            inten = np.zeros((len(xyz), 1), np.float32)
        X = np.concatenate([xyz, inten], axis=1).astype(np.float32)
    else:
        X = xyz.astype(np.float32)

    y_raw = np.asarray(v[key], dtype=np.int64)
    return X, y_raw

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

def miou_present(ious, supports, present_min_support=1):
    mask = supports >= int(present_min_support)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(ious[mask]))

# ---------- Model (MUST match training ckpts) ----------
class PointNet2LiteSeg(torch.nn.Module):
    def __init__(self, in_dim: int, ncls: int, use_global: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.ncls = int(ncls)
        self.use_global = bool(use_global)

        self.stem = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, 64), torch.nn.ReLU(True),
            torch.nn.Linear(64, 128), torch.nn.ReLU(True),
        )

        local_in = 128 + 3 + (128 if self.use_global else 0)
        self.local_mlp = torch.nn.Sequential(
            torch.nn.Linear(local_in, 128), torch.nn.ReLU(True),
            torch.nn.Linear(128, 128), torch.nn.ReLU(True),
        )

        head_in = 128 + 128 + (128 if self.use_global else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(head_in, 128), torch.nn.ReLU(True),
            torch.nn.Linear(128, 64), torch.nn.ReLU(True),
            torch.nn.Linear(64, self.ncls),
        )

    def forward(self, x):
        xyz = x[..., :3]
        pf = self.stem(x)  # (B,N,128)

        if self.use_global:
            gf, _ = torch.max(pf, dim=1)  # (B,128)
            gf_rep = gf.unsqueeze(1).expand(-1, pf.shape[1], -1)  # (B,N,128)
            local_in = torch.cat([pf, gf_rep, xyz], dim=-1)
            local = self.local_mlp(local_in)
            head_in = torch.cat([pf, local, gf_rep], dim=-1)
        else:
            local_in = torch.cat([pf, xyz], dim=-1)
            local = self.local_mlp(local_in)
            head_in = torch.cat([pf, local], dim=-1)

        return self.head(head_in)

@torch.no_grad()
def eval_tile(model, X, y, ncls, block_points=1024, seed=0, device="cpu"):
    rs = np.random.RandomState(seed)
    N = len(X)
    idx = rs.permutation(N)
    pred = np.empty((N,), dtype=np.int64)

    for s in range(0, N, block_points):
        sel = idx[s:s+block_points]
        xb = X[sel]
        xb_t = torch.from_numpy(xb).unsqueeze(0).to(device)
        out = model(xb_t)
        p = out.argmax(-1).squeeze(0).cpu().numpy()
        pred[sel] = p

    C = confusion(pred, y, ncls)
    ious = iou_from_conf(C)
    oa = float(np.trace(C) / max(1, C.sum()))
    miou = float(np.mean(ious))
    supports = np.array([C[i, :].sum() for i in range(ncls)], dtype=np.int64)
    return oa, miou, ious, supports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--max_points", type=int, default=1000000)
    ap.add_argument("--block_points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_global", choices=["auto","true","false"], default="auto")
    ap.add_argument("--use_intensity", choices=["auto","true","false"], default="auto")
    ap.add_argument("--present_min_support", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    in_dim = int(ckpt.get("in_dim", 4))
    ncls = int(ckpt.get("num_classes", 0))
    label_values = np.asarray(ckpt.get("label_values", list(range(ncls))), dtype=np.int64)
    if ncls <= 0:
        ncls = int(len(label_values))

    ck_use_global = bool(ckpt.get("use_global", True))
    ck_use_intensity = bool(ckpt.get("use_intensity", (in_dim == 4)))

    if args.use_global == "auto":
        use_global = ck_use_global
    else:
        use_global = (args.use_global == "true")

    if args.use_intensity == "auto":
        use_intensity = ck_use_intensity
    else:
        use_intensity = (args.use_intensity == "true")

    model = PointNet2LiteSeg(in_dim=in_dim, ncls=ncls, use_global=use_global).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(f"[eval] model=PointNet2LiteSeg in_dim={in_dim} ncls={ncls} use_global={use_global} use_intensity={use_intensity}")

    X, y_raw = read_ply_xyzI_and_labels(args.ply, use_intensity=use_intensity)

    X, y_raw, stride, before, after = apply_stride_if_needed(X, y_raw, args.max_points)
    if stride > 1:
        print(f"stride 1/{stride}: {before:,} -> {after:,}")

    X, y_raw = voxel_downsample_xyz(X, y_raw, args.voxel)
    X = normalize_feats(X)

    # remap labels to 0..ncls-1 using ckpt label_values
    yi = np.searchsorted(label_values, y_raw).astype(np.int64)
    # sanity check: if some labels not in label_values -> ignore (rare)
    bad = (yi < 0) | (yi >= ncls) | (label_values[yi] != y_raw)
    if bad.any():
        # drop bad points
        keep = ~bad
        X = X[keep]
        yi = yi[keep]
        print(f"[warn] dropped {bad.sum()} points with unknown labels (not in ckpt label_values)")

    oa, miou, ious, supports = eval_tile(
        model, X, yi, ncls=ncls, block_points=args.block_points, seed=args.seed, device=device
    )
    miou_p = miou_present(ious, supports, present_min_support=args.present_min_support)

    print(f"Eval points: {len(X):,} | OA={oa:.3f} | mIoU={miou:.3f} | mIoU_present(s>={args.present_min_support})={miou_p:.3f}")
    for i in range(ncls):
        print(f"  class {label_values[i]:>3d}  IoU={ious[i]:.3f}  support={supports[i]:,}")

if __name__ == "__main__":
    main()

