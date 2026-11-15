import json
import ast

with open("notebooks/deepvision_kaggle_gpu.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
code_cell_count = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        ast.parse(code)
        print(f"Cell {i:02d} [CODE]: Valid syntax ({len(cell['source'])} lines)")
        code_cell_count += 1

print(f"ALL {code_cell_count} CODE CELLS HAVE VALID PYTHON SYNTAX!")

