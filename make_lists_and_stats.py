"""
make_lists_and_stats_plateau.py

Estructura esperada:
  <dataset_dir>/train/*.ply
  <dataset_dir>/test/*.ply

Salida:
  <dataset_dir>/lists/train.txt
  <dataset_dir>/lists/test.txt
  <dataset_dir>/lists/plateau_stats.csv
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional
from plyfile import PlyData

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

def read_counts_and_hist(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    vert = ply["vertex"].data
    n = len(vert)
    key = _find_label_key(vert.dtype.names)
    has_labels = key is not None
    hist = {}
    if has_labels:
        y = np.asarray(vert[key], dtype=np.int64)
        y = y[(y >= 0) & (y < 256)]
        if y.size:
            vals, counts = np.unique(y, return_counts=True)
            hist = {int(v): int(c) for v, c in zip(vals, counts)}
    return n, has_labels, hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True,
                    help="Ruta a .../plateau_railway (contiene train/ y test/)")
    args = ap.parse_args()

    ds = Path(args.dataset_dir).expanduser().resolve()
    train_dir = ds / "train"
    test_dir  = ds / "test"
    lists_dir = ds / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise SystemExit(f"No existe: {train_dir}")
    if not test_dir.exists():
        raise SystemExit(f"No existe: {test_dir}")

    train_files = sorted(train_dir.glob("*.ply"))
    test_files  = sorted(test_dir.glob("*.ply"))

    (lists_dir / "train.txt").write_text("\n".join([p.name for p in train_files]), encoding="utf-8")
    (lists_dir / "test.txt").write_text("\n".join([p.name for p in test_files]), encoding="utf-8")

    rows = []
    for p in train_files:
        n, has_labels, hist = read_counts_and_hist(p)
        rows.append({
            "tile_name": p.name, "split":"train",
            "n_points": n, "has_labels": has_labels,
            "class_hist_json": json.dumps(hist, ensure_ascii=False)
        })
    for p in test_files:
        n, has_labels, hist = read_counts_and_hist(p)
        rows.append({
            "tile_name": p.name, "split":"test",
            "n_points": n, "has_labels": has_labels,
            "class_hist_json": json.dumps(hist, ensure_ascii=False)
        })

    out_csv = lists_dir / "plateau_stats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[OK] {lists_dir/'train.txt'}  ({len(train_files)} tiles)")
    print(f"[OK] {lists_dir/'test.txt'}   ({len(test_files)} tiles)")
    print(f"[OK] stats -> {out_csv}")

if __name__ == "__main__":
    main()

