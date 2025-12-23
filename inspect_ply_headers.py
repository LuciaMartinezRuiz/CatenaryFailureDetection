# inspect_ply_headers.py
from pathlib import Path
from plyfile import PlyData

# Mira los primeros .ply de TRAIN (cambia la ruta si la tuya es distinta)
train_dir = Path("data/raw/WHU-Railway3D/Urban/tiles/train")
files = sorted(train_dir.glob("*.ply"))
if not files:
    raise SystemExit("No se encontraron .ply en 'Urban/tiles/train'")

# Candidatos de nombre para la columna de etiquetas
candidates = [
    "label","semantic","class","classification",
    "category","category_id","seg_label",
    "scalar_Label","scalar_label","Scalar_Label"
]

for f in files[:5]:  # inspecciona hasta 5 ficheros
    ply = PlyData.read(str(f))
    names = ply["vertex"].data.dtype.names
    label_key = next((k for k in candidates if k in names), None)

    print("\n==========")
    print("Archivo:", f.name)
    print("Columnas de 'vertex':", names)
    print("Columna de etiqueta detectada:", label_key)
