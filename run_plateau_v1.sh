#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
DATA="data/raw/WHU-Railway3D/plateau_railway"
VOX="0.15"
MAXP="1000000"
SEED="42"

# Tiles que quieres reportar sí o sí (los que ya has usado en el texto)
REPORT_TILES=(Tibet-5 Tibet-10 Tibet-17 Tibet-18)

# Checkpoints "finales" (ajusta si quieres otro)
CKPT_MLP="outputs/checkpoints/mlp_plateau_xyzI_vox015_wsqrt.pt"
CKPT_PN="outputs/checkpoints/pointnet_plateau_xyzI_vox015_uniform.pt"
CKPT_PN2_CLEAN="outputs/checkpoints/pointnet2lite_plateau_xyzI_vox015_global_clean.pt"
CKPT_PN2_W="outputs/checkpoints/pointnet2lite_plateau_xyzI_vox015_global_weights_only.pt"

OUT="outputs/plateau_v1"
mkdir -p "$OUT"/{checkpoints,splits,eval,summary,logs}

echo "[1/5] Copying checkpoints to $OUT/checkpoints ..."
cp -f "$CKPT_MLP" "$OUT/checkpoints/" || true
cp -f "$CKPT_PN" "$OUT/checkpoints/" || true
cp -f "$CKPT_PN2_CLEAN" "$OUT/checkpoints/" || true
cp -f "$CKPT_PN2_W" "$OUT/checkpoints/" || true

echo "[2/5] Creating fixed splits (train/val from train/, test from test/) ..."
python - <<PY
import os, glob, random

seed = int("$SEED")
random.seed(seed)

train_dir = os.path.join("$DATA", "train")
test_dir  = os.path.join("$DATA", "test")

train_files = sorted(glob.glob(os.path.join(train_dir, "*.ply")))
test_files  = sorted(glob.glob(os.path.join(test_dir, "*.ply")))

report = set(["Tibet-5","Tibet-10","Tibet-17","Tibet-18"])

# pool para elegir val: train excluyendo report tiles
pool = [os.path.basename(p) for p in train_files if os.path.basename(p).replace(".ply","") not in report]
pool_sorted = sorted(pool)

random.shuffle(pool_sorted)
val_n = 4  # siguiendo estilo 12/4/4
val = sorted(pool_sorted[:val_n])

# train final: todo train menos val (y report tiles se quedan en train también, pero no en val)
train = sorted([os.path.basename(p) for p in train_files if os.path.basename(p) not in set(val)])

test = sorted([os.path.basename(p) for p in test_files])

os.makedirs("$OUT/splits", exist_ok=True)
for name, lst in [("train",train),("val",val),("test",test)]:
    with open(os.path.join("$OUT/splits", f"{name}.txt"), "w") as f:
        for x in lst:
            f.write(x + "\n")

print("Seed:", seed)
print("Train tiles:", len(train))
print("Val tiles:", len(val), val)
print("Test tiles:", len(test), test)
print("Report tiles:", sorted([x + ".ply" for x in report]))
PY

echo "[3/5] Evaluating models on REPORT + VAL + TEST tiles ..."

readarray -t VAL_TILES < "$OUT/splits/val.txt"
readarray -t TEST_TILES < "$OUT/splits/test.txt"

# Helper: run eval and tee output
run_eval () {
  local model="$1"
  local split="$2"
  local tile="$3"
  local cmd="$4"

  mkdir -p "$OUT/eval/$model/$split"
  echo "---- [$model][$split] $tile ----"
  eval "$cmd" | tee "$OUT/eval/$model/$split/${tile%.ply}.txt"
}

# --- Define evaluation sets ---
REPORT_PLY=()
for t in "${REPORT_TILES[@]}"; do REPORT_PLY+=("${t}.ply"); done

# --- MLP eval (eval_tile.py) ---
MLP_CK="$OUT/checkpoints/$(basename "$CKPT_MLP")"
for split in report val test; do
  tiles=()
  if [[ "$split" == "report" ]]; then tiles=("${REPORT_PLY[@]}"); fi
  if [[ "$split" == "val" ]]; then tiles=("${VAL_TILES[@]}"); fi
  if [[ "$split" == "test" ]]; then tiles=("${TEST_TILES[@]}"); fi

  for tile in "${tiles[@]}"; do
    if [[ "$split" == "test" ]]; then PLY="$DATA/test/$tile"; else PLY="$DATA/train/$tile"; fi
    run_eval "mlp" "$split" "$tile" \
      "PYTHONUNBUFFERED=1 python -u eval_tile.py --ply $PLY --ckpt $MLP_CK --voxel $VOX --max_points $MAXP --use_intensity --present_min_support 1"
  done
done

# --- PointNet eval (eval_tile_pointnet.py) ---
PN_CK="$OUT/checkpoints/$(basename "$CKPT_PN")"
for split in report val test; do
  tiles=()
  if [[ "$split" == "report" ]]; then tiles=("${REPORT_PLY[@]}"); fi
  if [[ "$split" == "val" ]]; then tiles=("${VAL_TILES[@]}"); fi
  if [[ "$split" == "test" ]]; then tiles=("${TEST_TILES[@]}"); fi

  for tile in "${tiles[@]}"; do
    if [[ "$split" == "test" ]]; then PLY="$DATA/test/$tile"; else PLY="$DATA/train/$tile"; fi
    run_eval "pointnet" "$split" "$tile" \
      "PYTHONUNBUFFERED=1 python -u eval_tile_pointnet.py --ply $PLY --ckpt $PN_CK --voxel $VOX --max_points $MAXP --block_points 4096 --min_support 1"
  done
done

# --- PointNet2Lite eval (eval_tile_pointnet2_lite.py) ---
PN2C_CK="$OUT/checkpoints/$(basename "$CKPT_PN2_CLEAN")"
PN2W_CK="$OUT/checkpoints/$(basename "$CKPT_PN2_W")"

for variant in pn2lite_clean pn2lite_weights; do
  if [[ "$variant" == "pn2lite_clean" ]]; then CK="$PN2C_CK"; fi
  if [[ "$variant" == "pn2lite_weights" ]]; then CK="$PN2W_CK"; fi

  for split in report val test; do
    tiles=()
    if [[ "$split" == "report" ]]; then tiles=("${REPORT_PLY[@]}"); fi
    if [[ "$split" == "val" ]]; then tiles=("${VAL_TILES[@]}"); fi
    if [[ "$split" == "test" ]]; then tiles=("${TEST_TILES[@]}"); fi

    for tile in "${tiles[@]}"; do
      if [[ "$split" == "test" ]]; then PLY="$DATA/test/$tile"; else PLY="$DATA/train/$tile"; fi
      run_eval "$variant" "$split" "$tile" \
        "PYTHONUNBUFFERED=1 python -u eval_tile_pointnet2_lite.py --ply $PLY --ckpt $CK --voxel $VOX --max_points $MAXP --block_points 1024 --seed $SEED --use_global auto --use_intensity auto"
    done
  done
done

echo "[4/5] Parsing logs into a single CSV summary ..."
python - <<'PY'
import os, re, glob, csv

ROOT = "outputs/plateau_v1/eval"

# Regex robusto para formatos típicos
rgx = {
    "OA": re.compile(r"\bOA\b\s*[=:]?\s*([0-9]*\.[0-9]+)"),
    "mIoU": re.compile(r"\bmIoU\b\s*[=:]?\s*([0-9]*\.[0-9]+)"),
    "mIoU_present": re.compile(r"\bmIoU_present\b\s*[=:]?\s*([0-9]*\.[0-9]+)"),
}

rows = []
for fp in sorted(glob.glob(os.path.join(ROOT, "*", "*", "*.txt"))):
    # outputs/plateau_v1/eval/<model>/<split>/<tile>.txt
    parts = fp.split(os.sep)
    model, split, tile = parts[-3], parts[-2], parts[-1].replace(".txt","")
    txt = open(fp, "r", errors="ignore").read()
    row = {"model": model, "split": split, "tile": tile}
    for k, r in rgx.items():
        m = r.search(txt)
        row[k] = float(m.group(1)) if m else None
    rows.append(row)

out = "outputs/plateau_v1/summary/plateau_v1_metrics.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model","split","tile","OA","mIoU","mIoU_present"])
    w.writeheader()
    w.writerows(rows)

print("Wrote:", out)
print("Rows:", len(rows))
PY

echo "[5/5] Done. Summary CSV: $OUT/summary/plateau_v1_metrics.csv"
