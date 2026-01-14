from pathlib import Path
from plyfile import PlyData
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train","test"])
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    d = Path(args.dataset_dir).expanduser().resolve() / args.split
    files = sorted(d.glob("*.ply"))
    if not files:
        raise SystemExit(f"No se encontraron .ply en {d}")

    candidates = [
        "label","semantic","class","classification",
        "category","category_id","seg_label",
        "scalar_Label","scalar_label","Scalar_Label"
    ]

    for f in files[:args.n]:
        ply = PlyData.read(str(f))
        names = ply["vertex"].data.dtype.names
        label_key = next((k for k in candidates if k in names), None)
        print("\n==========")
        print("Archivo:", f.name)
        print("Columnas vertex:", names)
        print("Etiqueta detectada:", label_key)

if __name__ == "__main__":
    main()

