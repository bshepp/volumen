import json

with open(r"f:\kaggle\vesuvius_challenge\notebooks\submission.ipynb") as f:
    nb = json.load(f)

print(f"Notebook format: nbformat {nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Number of cells: {len(nb['cells'])}")

for i, cell in enumerate(nb["cells"]):
    ctype = cell["cell_type"]
    src = "".join(cell.get("source", []))
    preview = src[:80].replace("\n", " ")
    print(f"  Cell {i}: {ctype:>8} | {preview}...")

print("Notebook is valid JSON and well-formed.")
