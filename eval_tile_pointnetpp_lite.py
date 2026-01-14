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

# -------- shared utils --------
def _find_label_key(names):
    for k in ["class", "label", "semantic", "classification", "category", "category_id", "seg_label", "sem_class"]:
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

def apply_stride_if_needed(X, y, max_points):
    n_total = len(X)
    if max_points is None or max_points <= 0 or n_total <= max_points:
        return X, y, 1, n_total, n_total
    stride = int(np.ceil(n_total / max_points))
    X2 = X[::stride]
    y2 = y[::stride]
    return X2, y2, stride, n_total, len(X2)

def read_ply_xyzI_and_labels(ply_path, use_intensity=True):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    names = v.dtype.names

    key = _find_label_key(names)
    if key is None:
        raise SystemExit(f"No label field found. Fields={names}")

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

# -------- model (must match train) --------
def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def knn_indices(query_xyz, xyz, k):
    d = torch.cdist(query_xyz, xyz)
    k = min(int(k), xyz.shape[1])
    _, idx = torch.topk(d, k=k, dim=-1, largest=False, sorted=False)
    return idx

def sample_indices(N, npoint, seed=None, device="cpu"):
    npoint = min(int(npoint), int(N))
    if npoint <= 0:
        npoint = 1
    if seed is None:
        perm = torch.randperm(N, device=device)[:npoint]
    else:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        perm = torch.randperm(N, generator=g, device=device)[:npoint]
    return perm

class SharedMLP(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(in_ch, out_ch), torch.nn.ReLU(True))
    def forward(self, x):
        return self.net(x)

class PointNetSetAbstractionLite(torch.nn.Module):
    def __init__(self, npoint, k, in_ch, out_ch):
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)
        self.mlp1 = SharedMLP(in_ch, out_ch)
        self.mlp2 = SharedMLP(out_ch, out_ch)

    def forward(self, xyz, feat, seed=None):
        B, N, _ = xyz.shape
        S = min(self.npoint, N)

        idx_s_list = []
        for b in range(B):
            s = None if seed is None else (int(seed) + b * 17)
            idx_s_list.append(sample_indices(N, S, seed=s, device=xyz.device))
        idx_s = torch.stack(idx_s_list, dim=0)

        new_xyz = index_points(xyz, idx_s)
        idx = knn_indices(new_xyz, xyz, self.k)

        group_xyz = index_points(xyz, idx)
        rel_xyz = group_xyz - new_xyz.unsqueeze(2)

        if feat is None:
            group_in = rel_xyz
        else:
            group_feat = index_points(feat, idx)
            group_in = torch.cat([rel_xyz, group_feat], dim=-1)

        x = self.mlp1(group_in)
        x = self.mlp2(x)
        new_feat = torch.max(x, dim=2)[0]
        return new_xyz, new_feat

class PointNetFeaturePropagationLite(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp1 = SharedMLP(in_ch, out_ch)
        self.mlp2 = SharedMLP(out_ch, out_ch)

    def forward(self, xyz1, xyz2, feat1, feat2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interp = feat2.repeat(1, N, 1)
        else:
            d = torch.cdist(xyz1, xyz2)
            k = min(3, S)
            dist, idx = torch.topk(d, k=k, dim=-1, largest=False, sorted=False)
            dist = torch.clamp(dist, min=1e-10)
            w = 1.0 / dist
            w = w / torch.sum(w, dim=-1, keepdim=True)

            f2 = index_points(feat2, idx)
            interp = torch.sum(f2 * w.unsqueeze(-1), dim=2)

        if feat1 is None:
            x = interp
        else:
            x = torch.cat([feat1, interp], dim=-1)

        x = self.mlp1(x)
        x = self.mlp2(x)
        return x

class PointNetPPLiteSeg(torch.nn.Module):
    def __init__(self, in_dim, ncls, use_global=True,
                 sa1_npoint=256, sa1_k=32,
                 sa2_npoint=64,  sa2_k=32,
                 dropout=0.2):
        super().__init__()
        self.in_dim = int(in_dim)
        self.ncls = int(ncls)
        self.use_global = bool(use_global)

        self.embed = torch.nn.Sequential(torch.nn.Linear(self.in_dim, 64), torch.nn.ReLU(True))
        self.sa1 = PointNetSetAbstractionLite(sa1_npoint, sa1_k, in_ch=3 + 64, out_ch=128)
        self.sa2 = PointNetSetAbstractionLite(sa2_npoint, sa2_k, in_ch=3 + 128, out_ch=256)
        self.fp2 = PointNetFeaturePropagationLite(in_ch=256 + 128, out_ch=128)
        self.fp1 = PointNetFeaturePropagationLite(in_ch=128 + 64, out_ch=128)

        head_in = 128 + (128 if self.use_global else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(head_in, 128), torch.nn.ReLU(True),
            torch.nn.Dropout(p=float(dropout)),
            torch.nn.Linear(128, 64), torch.nn.ReLU(True),
            torch.nn.Linear(64, self.ncls),
        )

    def forward(self, x, seed=None):
        xyz = x[..., :3]
        feat0 = self.embed(x)

        xyz1, feat1 = self.sa1(xyz, feat0, seed=seed)
        xyz2, feat2 = self.sa2(xyz1, feat1, seed=None if seed is None else (int(seed) + 1000))

        feat1_up = self.fp2(xyz1, xyz2, feat1, feat2)
        feat0_up = self.fp1(xyz, xyz1, feat0, feat1_up)

        if self.use_global:
            gf, _ = torch.max(feat0_up, dim=1)
            gf_rep = gf.unsqueeze(1).expand(-1, feat0_up.shape[1], -1)
            head_in = torch.cat([feat0_up, gf_rep], dim=-1)
        else:
            head_in = feat0_up

        return self.head(head_in)

# -------- eval main --------
@torch.no_grad()
def eval_tile(model, X, y, ncls, block_points=1024, seed=0, present_min_support=1, device="cpu"):
    rs = np.random.RandomState(seed)
    N = len(X)
    idx = rs.permutation(N)
    pred = np.empty((N,), dtype=np.int64)

    model.eval()
    for s in range(0, N, block_points):
        sel = idx[s:s+block_points]
        xb = X[sel]
        xb_t = torch.from_numpy(xb).unsqueeze(0).to(device)
        out = model(xb_t, seed=seed + s)
        p = out.argmax(-1).squeeze(0).cpu().numpy()
        pred[sel] = p

    C = confusion(pred, y, ncls)
    ious = iou_from_conf(C)
    oa = float(np.trace(C) / max(1, C.sum()))
    miou = float(np.mean(ious))
    supports = np.array([C[i, :].sum() for i in range(ncls)], dtype=np.int64)
    miou_p = miou_present(ious, supports, present_min_support=present_min_support)
    return oa, miou, miou_p, ious, supports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--max_points", type=int, default=1000000)
    ap.add_argument("--block_points", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_global", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--use_intensity", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--present_min_support", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    in_dim = int(ckpt.get("in_dim", 4))
    ncls = int(ckpt.get("num_classes"))
    label_values = np.asarray(ckpt.get("label_values", list(range(ncls))), dtype=np.int64)

    ck_use_global = bool(ckpt.get("use_global", True))
    ck_use_intensity = bool(ckpt.get("use_intensity", True))

    if args.use_global == "auto":
        use_global = ck_use_global
    else:
        use_global = (args.use_global == "true")

    if args.use_intensity == "auto":
        use_intensity = ck_use_intensity
    else:
        use_intensity = (args.use_intensity == "true")

    # SA params
    sa1_npoint = int(ckpt.get("sa1_npoint", 256))
    sa1_k = int(ckpt.get("sa1_k", 32))
    sa2_npoint = int(ckpt.get("sa2_npoint", 64))
    sa2_k = int(ckpt.get("sa2_k", 32))
    dropout = float(ckpt.get("dropout", 0.2))

    model = PointNetPPLiteSeg(
        in_dim=in_dim,
        ncls=ncls,
        use_global=use_global,
        sa1_npoint=sa1_npoint,
        sa1_k=sa1_k,
        sa2_npoint=sa2_npoint,
        sa2_k=sa2_k,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    print(f"[eval] model=PointNetPPLiteSeg in_dim={in_dim} ncls={ncls} use_global={use_global} use_intensity={use_intensity}")

    X, y_raw = read_ply_xyzI_and_labels(args.ply, use_intensity=use_intensity)
    X, y_raw, stride, before, after = apply_stride_if_needed(X, y_raw, args.max_points)
    if stride > 1:
        print(f"stride 1/{stride}: {before:,} -> {after:,}")

    X, y_raw = voxel_downsample_xyz(X, y_raw, args.voxel)

    # map labels to indices
    mapping = {int(v): i for i, v in enumerate(label_values.tolist())}
    y = np.array([mapping.get(int(v), -1) for v in y_raw], dtype=np.int64)
    valid = (y >= 0)
    Xn = normalize_feats(X)

    oa, miou, miou_p, ious, supports = eval_tile(
        model,
        Xn[valid],
        y[valid],
        ncls=ncls,
        block_points=args.block_points,
        seed=args.seed,
        present_min_support=args.present_min_support,
        device=device,
    )

    print(f"Eval points: {int(valid.sum()):,} | OA={oa:.3f} | mIoU={miou:.3f} | mIoU_present(s>={args.present_min_support})={miou_p:.3f}")
    for i, lv in enumerate(label_values.tolist()):
        print(f"  class {lv:>3d}  IoU={ious[i]:.3f}  support={supports[i]:,}")

if __name__ == "__main__":
    main()

