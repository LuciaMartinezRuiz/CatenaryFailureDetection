import argparse
from pathlib import Path
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

# Silencia warnings molestos de torch.load
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch\.load` with `weights_only=False`.*",
)

# ----------------------------
# Utils dataset / métricas
# ----------------------------
def _find_label_key(names):
    for k in ["class", "label", "semantic", "classification", "category", "category_id", "seg_label", "sem_class"]:
        if k in names:
            return k
    return None

def normalize_feats(X):
    """Robust scaling por mediana + IQR."""
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

def load_tile(ply_path, use_intensity, voxel, max_points_per_tile):
    X, y_raw = read_ply_xyzI_and_labels(ply_path, use_intensity=use_intensity)

    X, y_raw, stride, before, after = apply_stride_if_needed(X, y_raw, max_points_per_tile)
    if stride > 1:
        print(f"   · stride 1/{stride}  ({before:,} -> {after:,} puntos)")

    X, y_raw = voxel_downsample_xyz(X, y_raw, voxel)
    print(f"   · voxel {voxel:g} -> {len(X):,} puntos")
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

# ----------------------------
# Sparse supervision helper
# ----------------------------
def make_sparse_labels(y, keep_ratio, n_labeled, ignore_index, rs: np.random.RandomState):
    """
    y: (B, N) int64, labels ya remapeadas [0..ncls-1]
    - Si n_labeled>0: mantiene exactamente n_labeled por bloque (si puede)
    - Si no: usa keep_ratio (proporción)
    """
    y = y.copy()
    B, N = y.shape
    total_kept = 0

    for b in range(B):
        if n_labeled is not None and n_labeled > 0:
            k = min(int(n_labeled), N)
        else:
            k = int(max(1, round(float(keep_ratio) * N))) if keep_ratio > 0 else N
            k = min(k, N)

        idx = rs.choice(N, size=k, replace=False) if k < N else np.arange(N)
        mask = np.zeros((N,), dtype=np.bool_)
        mask[idx] = True
        y[b, ~mask] = ignore_index
        total_kept += int(mask.sum())

    return y, total_kept

# ----------------------------
# PointNet++-Lite blocks
# ----------------------------
def index_points(points, idx):
    """
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    return: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=device, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def knn_indices(query_xyz, xyz, k):
    """
    query_xyz: (B, S, 3)
    xyz: (B, N, 3)
    return idx: (B, S, k)
    """
    d = torch.cdist(query_xyz, xyz)  # (B,S,N)
    k = min(int(k), xyz.shape[1])
    _, idx = torch.topk(d, k=k, dim=-1, largest=False, sorted=False)
    return idx

def sample_indices(N, npoint, seed=None, device="cpu"):
    """
    Deterministic random sampling (seeded) o aleatorio si seed=None.
    Devuelve idx shape (npoint,)
    """
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
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_ch, out_ch),
            torch.nn.ReLU(True),
        )
    def forward(self, x):
        return self.net(x)

class PointNetSetAbstractionLite(torch.nn.Module):
    """
    SA layer: sample S points + kNN group + mini PointNet + maxpool over neighbors.
    """
    def __init__(self, npoint, k, in_ch, out_ch):
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)
        self.mlp1 = SharedMLP(in_ch, out_ch)
        self.mlp2 = SharedMLP(out_ch, out_ch)

    def forward(self, xyz, feat, seed=None):
        """
        xyz:  (B, N, 3)
        feat: (B, N, C)  (puede ser None -> usamos solo xyz rel)
        return:
          new_xyz: (B, S, 3)
          new_feat:(B, S, out_ch)
        """
        B, N, _ = xyz.shape
        S = min(self.npoint, N)

        idx_s_list = []
        for b in range(B):
            s = None if seed is None else (int(seed) + b * 17)
            idx_s_list.append(sample_indices(N, S, seed=s, device=xyz.device))
        idx_s = torch.stack(idx_s_list, dim=0)  # (B,S)

        new_xyz = index_points(xyz, idx_s)  # (B,S,3)
        idx = knn_indices(new_xyz, xyz, self.k)  # (B,S,k)

        group_xyz = index_points(xyz, idx)  # (B,S,k,3)
        rel_xyz = group_xyz - new_xyz.unsqueeze(2)  # (B,S,k,3)

        if feat is None:
            group_in = rel_xyz
        else:
            group_feat = index_points(feat, idx)  # (B,S,k,C)
            group_in = torch.cat([rel_xyz, group_feat], dim=-1)  # (B,S,k,3+C)

        x = self.mlp1(group_in)
        x = self.mlp2(x)
        new_feat = torch.max(x, dim=2)[0]  # (B,S,out_ch)
        return new_xyz, new_feat

class PointNetFeaturePropagationLite(torch.nn.Module):
    """
    FP: interp features from xyz2->xyz1 using 3NN, concat skip feat, MLP.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp1 = SharedMLP(in_ch, out_ch)
        self.mlp2 = SharedMLP(out_ch, out_ch)

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        xyz1: (B, N, 3) target (dense)
        xyz2: (B, S, 3) source (coarse)
        feat1:(B, N, C1) skip (puede ser None)
        feat2:(B, S, C2) source features
        return: (B, N, out_ch)
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interp = feat2.repeat(1, N, 1)
        else:
            d = torch.cdist(xyz1, xyz2)  # (B,N,S)
            k = min(3, S)
            dist, idx = torch.topk(d, k=k, dim=-1, largest=False, sorted=False)  # (B,N,k)
            dist = torch.clamp(dist, min=1e-10)
            w = 1.0 / dist
            w = w / torch.sum(w, dim=-1, keepdim=True)

            f2 = index_points(feat2, idx)  # (B,N,k,C2)
            interp = torch.sum(f2 * w.unsqueeze(-1), dim=2)  # (B,N,C2)

        if feat1 is None:
            x = interp
        else:
            x = torch.cat([feat1, interp], dim=-1)

        x = self.mlp1(x)
        x = self.mlp2(x)
        return x

# ----------------------------
# Modelo PointNet++-Lite Seg
# ----------------------------
class PointNetPPLiteSeg(torch.nn.Module):
    def __init__(self, in_dim, ncls, use_global=True,
                 sa1_npoint=256, sa1_k=32,
                 sa2_npoint=64,  sa2_k=32,
                 dropout=0.2):
        super().__init__()
        self.in_dim = int(in_dim)
        self.ncls = int(ncls)
        self.use_global = bool(use_global)

        # Initial per-point embedding (NOT neighborhood yet)
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(self.in_dim, 64),
            torch.nn.ReLU(True),
        )

        # SA layers (neighborhood-based)
        # SA1 input: rel_xyz(3) + feat(64) => 67
        self.sa1 = PointNetSetAbstractionLite(sa1_npoint, sa1_k, in_ch=3 + 64, out_ch=128)
        # SA2 input: rel_xyz(3) + feat(128) => 131
        self.sa2 = PointNetSetAbstractionLite(sa2_npoint, sa2_k, in_ch=3 + 128, out_ch=256)

        # FP layers
        # FP2: interpolate 256 -> to SA1 points, concat skip(128) => 384
        self.fp2 = PointNetFeaturePropagationLite(in_ch=256 + 128, out_ch=128)
        # FP1: interpolate 128 -> to original points, concat skip(64) => 192
        self.fp1 = PointNetFeaturePropagationLite(in_ch=128 + 64, out_ch=128)

        head_in = 128 + (128 if self.use_global else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(head_in, 128),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=float(dropout)),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, self.ncls),
        )

    def forward(self, x, seed=None):
        """
        x: (B, N, in_dim) con xyz en las 3 primeras dims
        seed: para hacer el sampling determinista (eval)
        """
        xyz = x[..., :3]
        feat0 = self.embed(x)  # (B,N,64)

        xyz1, feat1 = self.sa1(xyz, feat0, seed=seed)        # (B,S1,3), (B,S1,128)
        xyz2, feat2 = self.sa2(xyz1, feat1, seed=None if seed is None else (int(seed) + 1000))  # (B,S2,3), (B,S2,256)

        feat1_up = self.fp2(xyz1, xyz2, feat1, feat2)        # (B,S1,128)
        feat0_up = self.fp1(xyz,  xyz1, feat0, feat1_up)     # (B,N,128)

        if self.use_global:
            gf, _ = torch.max(feat0_up, dim=1)               # (B,128)
            gf_rep = gf.unsqueeze(1).expand(-1, feat0_up.shape[1], -1)
            head_in = torch.cat([feat0_up, gf_rep], dim=-1)  # (B,N,256)
        else:
            head_in = feat0_up

        logits = self.head(head_in)  # (B,N,ncls)
        return logits

# ----------------------------
# Losses
# ----------------------------
def focal_loss(logits, targets, gamma=2.0, weight=None, ignore_index=-1):
    """
    logits: (B,N,C)
    targets:(B,N) int64, puede incluir ignore_index
    """
    B, N, C = logits.shape
    logits2 = logits.reshape(-1, C)
    t2 = targets.reshape(-1)

    mask = (t2 != int(ignore_index))
    if mask.sum() == 0:
        return logits2.sum() * 0.0

    logits_m = logits2[mask]
    t_m = t2[mask]

    ce = F.cross_entropy(logits_m, t_m, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** float(gamma)) * ce
    return loss.mean()

# ----------------------------
# Validación estable (bloques fijos)
# ----------------------------
@torch.no_grad()
def eval_on_arrays(model, X, y, ncls, block_points=1024, seed=0, device="cpu", present_min_support=1):
    rs = np.random.RandomState(seed)
    N = len(X)
    idx = rs.permutation(N)
    pred = np.empty((N,), dtype=np.int64)

    model.eval()
    for s in range(0, N, block_points):
        sel = idx[s:s+block_points]
        xb = X[sel]
        xb_t = torch.from_numpy(xb).unsqueeze(0).to(device)
        out = model(xb_t, seed=seed + s)  # deterministic sampling per block
        p = out.argmax(-1).squeeze(0).cpu().numpy()
        pred[sel] = p

    C = confusion(pred, y, ncls)
    ious = iou_from_conf(C)
    oa = float(np.trace(C) / max(1, C.sum()))
    miou = float(np.mean(ious))
    supports = np.array([C[i, :].sum() for i in range(ncls)], dtype=np.int64)
    miou_p = miou_present(ious, supports, present_min_support=present_min_support)
    return oa, miou, miou_p

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", required=True, type=str)
    ap.add_argument("--train_limit", type=int, default=10, help="Número total de tiles a considerar (el último se usa como val).")
    ap.add_argument("--epochs", type=int, default=20)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_points", type=int, default=1024)
    ap.add_argument("--sample_per_epoch", type=int, default=400000)

    ap.add_argument("--voxel", type=float, default=0.15)
    ap.add_argument("--max_points_per_tile", type=int, default=1000000)

    ap.add_argument("--use_intensity", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--use_global", action="store_true")

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val_block_points", type=int, default=1024)
    ap.add_argument("--val_max_points", type=int, default=200000, help="Cap puntos val (0=sin cap).")
    ap.add_argument("--present_min_support", type=int, default=1)

    # PointNet++-lite params
    ap.add_argument("--sa1_npoint", type=int, default=256)
    ap.add_argument("--sa1_k", type=int, default=32)
    ap.add_argument("--sa2_npoint", type=int, default=64)
    ap.add_argument("--sa2_k", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)

    # Imbalance / losses
    ap.add_argument("--class_weights", action="store_true", help="1/sqrt(freq) con cap opcional.")
    ap.add_argument("--class_weight_cap", type=float, default=5.0, help="Cap multiplicativo para pesos de clase.")
    ap.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    # Sparse supervision
    ap.add_argument("--label_keep_ratio", type=float, default=0.0, help="Proporción de labels por bloque (0=full).")
    ap.add_argument("--sparse_n_labeled", type=int, default=0, help="Número fijo de labels por bloque (0=off).")
    ap.add_argument("--ignore_index", type=int, default=-1, help="Valor para unlabeled en sparse supervision.")

    ap.add_argument("--out_ckpt", type=str, default="outputs/checkpoints/pnpp_lite_plateau.pt")

    args = ap.parse_args()

    # Seeds (numpy + torch)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    train_dir = Path(args.dataset_dir) / "train"
    if not train_dir.exists():
        raise SystemExit(f"No existe: {train_dir}")

    tiles = sorted(train_dir.glob("*.ply"))
    if len(tiles) == 0:
        raise SystemExit(f"No hay .ply en {train_dir}")

    if args.train_limit > 0:
        tiles = tiles[:args.train_limit]

    if len(tiles) < 2:
        raise SystemExit("Necesito al menos 2 tiles (train+val).")

    train_tiles = tiles[:-1]
    val_tiles = tiles[-1:]

    print(f"[train] loading {len(train_tiles)} tile(s)…")
    X_train_raw, y_train_raw = [], []
    for p in train_tiles:
        print(f" - Leyendo {p.name}")
        Xp, yp = load_tile(p, args.use_intensity, args.voxel, args.max_points_per_tile)
        X_train_raw.append(Xp)
        y_train_raw.append(yp)

    # labels solo desde train
    all_train_labels = np.concatenate(y_train_raw, axis=0)
    label_values = np.unique(all_train_labels).astype(np.int64)
    label_values = np.sort(label_values)
    ncls = int(len(label_values))
    print(f"[train] ncls={ncls} labels={label_values.tolist()}")

    # remapeo train -> idx [0..ncls-1] + normalización
    X_train, y_train = [], []
    for Xp, yp in zip(X_train_raw, y_train_raw):
        yi = np.searchsorted(label_values, yp).astype(np.int64)
        X_train.append(normalize_feats(Xp))
        y_train.append(yi)

    # val
    print(f"[val] loading {len(val_tiles)} tile(s)…")
    X_val, y_val = [], []
    for p in val_tiles:
        print(f" - Leyendo {p.name}")
        Xp, yp = load_tile(p, args.use_intensity, args.voxel, args.max_points_per_tile)
        yi = np.searchsorted(label_values, yp).astype(np.int64)
        X_val.append(normalize_feats(Xp))
        y_val.append(yi)

    # cap val
    if args.val_max_points and args.val_max_points > 0:
        Xv, yv = X_val[0], y_val[0]
        if len(Xv) > args.val_max_points:
            stride = int(np.ceil(len(Xv) / args.val_max_points))
            X_val[0] = Xv[::stride]
            y_val[0] = yv[::stride]
            print(f"[val] stride 1/{stride}: {len(Xv):,} -> {len(X_val[0]):,}")

    in_dim = 4 if args.use_intensity else 3

    model = PointNetPPLiteSeg(
        in_dim=in_dim,
        ncls=ncls,
        use_global=args.use_global,
        sa1_npoint=args.sa1_npoint,
        sa1_k=args.sa1_k,
        sa2_npoint=args.sa2_npoint,
        sa2_k=args.sa2_k,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # class weights
    ce_weight = None
    if args.class_weights:
        counts = np.zeros((ncls,), dtype=np.int64)
        for yt in y_train:
            counts += np.bincount(yt, minlength=ncls)
        w = 1.0 / np.sqrt(counts.astype(np.float64) + 1e-6)
        w = w / max(1e-12, w.mean())
        w = np.clip(w, 0.0, float(args.class_weight_cap))
        ce_weight = torch.from_numpy(w.astype(np.float32)).to(device)

    steps_per_epoch = int(np.ceil(args.sample_per_epoch / (args.batch_size * args.n_points)))
    rs = np.random.RandomState(args.seed)

    # Sparse config info
    sparse_on = (args.label_keep_ratio > 0.0) or (args.sparse_n_labeled and args.sparse_n_labeled > 0)
    if sparse_on:
        print(f"[sparse] label_keep_ratio={args.label_keep_ratio} sparse_n_labeled={args.sparse_n_labeled} ignore_index={args.ignore_index}")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        labeled_counts = []

        for _ in range(steps_per_epoch):
            xb_list, yb_list = [], []

            for _b in range(args.batch_size):
                tid = rs.randint(0, len(X_train))
                Xt = X_train[tid]
                yt = y_train[tid]

                if len(Xt) >= args.n_points:
                    idx = rs.choice(len(Xt), size=args.n_points, replace=False)
                else:
                    idx = rs.choice(len(Xt), size=args.n_points, replace=True)

                xb = Xt[idx].copy()
                yb = yt[idx].copy()

                # augment xyz
                if args.augment:
                    ang = rs.uniform(0, 2 * np.pi)
                    c, s = np.cos(ang), np.sin(ang)
                    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                    xb[:, :3] = xb[:, :3] @ R.T
                    xb[:, :3] += rs.normal(0.0, 0.01, size=xb[:, :3].shape).astype(np.float32)

                xb_list.append(xb)
                yb_list.append(yb)

            xb_np = np.stack(xb_list, axis=0)  # (B,N,in_dim)
            yb_np = np.stack(yb_list, axis=0)  # (B,N)

            kept = args.batch_size * args.n_points
            if sparse_on:
                yb_np, kept = make_sparse_labels(
                    yb_np,
                    keep_ratio=args.label_keep_ratio,
                    n_labeled=args.sparse_n_labeled,
                    ignore_index=args.ignore_index,
                    rs=rs,
                )
            labeled_counts.append(float(kept))

            xb_t = torch.from_numpy(xb_np).to(device)
            yb_t = torch.from_numpy(yb_np).to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb_t)  # (B,N,ncls)

            if args.loss == "ce":
                loss = F.cross_entropy(
                    logits.reshape(-1, ncls),
                    yb_t.reshape(-1),
                    weight=ce_weight,
                    ignore_index=int(args.ignore_index),
                )
            else:
                loss = focal_loss(
                    logits, yb_t,
                    gamma=args.focal_gamma,
                    weight=ce_weight,
                    ignore_index=int(args.ignore_index),
                )

            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        # Val estable
        oa, miou, miou_p = eval_on_arrays(
            model,
            X_val[0],
            y_val[0],
            ncls=ncls,
            block_points=args.val_block_points,
            seed=args.seed + ep,
            device=device,
            present_min_support=args.present_min_support,
        )

        if sparse_on:
            print(f"[ep {ep:03d}] loss={np.mean(losses):.4f} mean_labeled/step={np.mean(labeled_counts):.1f} val_OA={oa:.3f} val_mIoU={miou:.3f} val_mIoU_present={miou_p:.3f}")
        else:
            print(f"[ep {ep:03d}] loss={np.mean(losses):.4f} val_OA={oa:.3f} val_mIoU={miou:.3f} val_mIoU_present={miou_p:.3f}")

    # Save ckpt
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_class": "PointNetPPLiteSeg",
        "model": model.state_dict(),
        "in_dim": int(in_dim),
        "num_classes": int(ncls),
        "label_values": label_values.astype(np.int64).tolist(),
        "use_global": bool(args.use_global),
        "use_intensity": bool(args.use_intensity),
        "sa1_npoint": int(args.sa1_npoint),
        "sa1_k": int(args.sa1_k),
        "sa2_npoint": int(args.sa2_npoint),
        "sa2_k": int(args.sa2_k),
        "dropout": float(args.dropout),
    }
    torch.save(ckpt, str(out_path))
    print(f"[OK] saved checkpoint to {out_path}")

if __name__ == "__main__":
    main()

