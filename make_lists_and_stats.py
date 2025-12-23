"""
make_lists_and_stats.py

Crea listas de ficheros y un CSV de estadísticas para WHU-Railway3D (escena Urban).
- Detecta .ply en Urban/tiles/train y Urban/tiles/test
- Escribe Urban/lists/train.txt y Urban/lists/test.txt
- Genera Urban/lists/urban_stats.csv con:
  tile_name, split, n_points, has_labels, class_hist_json

Uso:
  python make_lists_and_stats.py --base data/raw/WHU-Railway3D
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional
from plyfile import PlyData

def _find_label_key(names) -> Optional[str]:
    # Incluye 'class' que es tu caso
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
    """Lee un .ply y devuelve n_points, has_labels(bool), hist(dict) si hay labels."""
    ply = PlyData.read(str(ply_path))
    vert = ply['vertex'].data
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
    ap.add_argument("--base", type=str, required=True,
                    help="Ruta base que contiene WHU-Railway3D/Urban")
    args = ap.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    urban = base_dir / "Urban"
    tiles_train = urban / "tiles" / "train"
    tiles_test  = urban / "tiles" / "test"
    lists_dir   = urban / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted([p for p in tiles_train.glob("*.ply")])
    test_files  = sorted([p for p in tiles_test.glob("*.ply")])

    # Escribimos listas
    (lists_dir / "train.txt").write_text("\n".join([p.name for p in train_files]), encoding="utf-8")
    (lists_dir / "test.txt").write_text("\n".join([p.name for p in test_files]), encoding="utf-8")

    # Estadísticas
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

    df = pd.DataFrame(rows)
    out_csv = lists_dir / "urban_stats.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Escrito {lists_dir/'train.txt'} y {lists_dir/'test.txt'}")
    print(f"[OK] Estadísticas en {out_csv}")
    print(df.head(8).to_string(index=False))

if __name__ == "__main__":
    main()
