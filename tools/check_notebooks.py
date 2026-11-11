"""Check notebook JSON structure without executing ML models."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
notebooks = sorted((root / "notebooks").glob("*.ipynb"))
if not notebooks:
    raise SystemExit("No notebooks found")
for p in notebooks:
    notebook = json.loads(p.read_text(encoding="utf-8"))
    assert notebook.get("nbformat", 0) >= 4, p
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells, p
    assert all(c.get("outputs", []) == [] for c in code_cells), p
    print(f"OK {p.name}: {len(code_cells)} code cells; outputs stripped")
