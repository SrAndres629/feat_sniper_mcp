import ast
import json
import os
from pathlib import Path

# Optimization: Ignore heavy directories
IGNORE_DIRS = {'.git', 'venv', '.venv', '__pycache__', '.vscode', 'archive', 'node_modules', 'site-packages'}

class CortexIndexer:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.index = {}

    def analyze_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                node = ast.parse(f.read())
            except SyntaxError:
                return
            except UnicodeDecodeError:
                return # Skip binary or weird files

        # Calculate relative path string
        try:
            # Handle potential path issues
            file_key = str(Path(file_path).relative_to(self.root_dir))
        except ValueError:
            file_key = str(file_path)

        self.index[file_key] = {"classes": {}, "functions": []}

        for item in node.body:
            if isinstance(item, ast.ClassDef):
                methods = [m.name for m in item.body if isinstance(m, ast.FunctionDef)]
                self.index[file_key]["classes"][item.name] = {
                    "methods": methods,
                    "docstring": ast.get_docstring(item)
                }
            elif isinstance(item, ast.FunctionDef):
                self.index[file_key]["functions"].append({
                    "name": item.name,
                    "docstring": ast.get_docstring(item)
                })

    def run(self):
        print(f"🧠 indexing {self.root_dir}...")
        for root, dirs, files in os.walk(self.root_dir):
            # Optimization: Modify dirs in-place to skip ignored ones
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
            
            for file in files:
                if file.endswith(".py") and file != "cortex_indexer.py":
                    self.analyze_file(os.path.join(root, file))
        
        with open("project_atlas.json", "w", encoding='utf-8') as f:
            json.dump(self.index, f, indent=4)
        print("✅ Fase 1.5: project_atlas.json generado exitosamente.")

if __name__ == "__main__":
    indexer = CortexIndexer(".")
    indexer.run()
