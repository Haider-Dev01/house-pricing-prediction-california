import json

notebook_path = "Main.ipynb"
try:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    print(f"Number of cells: {len(nb.get('cells', []))}")
    code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    print(f"Number of code cells: {len(code_cells)}")
    
    print("\n--- Summary of Code Cells ---")
    for idx, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        first_few_lines = "\n".join(source.split("\n")[:4])
        print(f"\n[Cell {idx+1}]")
        print(first_few_lines)
        if len(source.split("\n")) > 4:
            print("...")
except Exception as e:
    print(f"Error reading notebook: {e}")
