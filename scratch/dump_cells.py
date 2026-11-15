import json

with open("notebooks/deepvision_kaggle_gpu.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    print("=" * 60)
    print(f"CELL {i:02d} [{cell['cell_type'].upper()}]")
    print("=" * 60)
    print("".join(cell["source"]).strip())
    print()

