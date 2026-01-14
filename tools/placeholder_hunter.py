import os
import ast

class PlaceholderHunter(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.issues = []

    def visit_FunctionDef(self, node):
        # Detectar funciones vacías (solo con 'pass' o docstrings)
        body = [n for n in node.body if not isinstance(n, ast.Expr)] # Ignorar docstrings
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            self.issues.append(f"[VACÍO] Función '{node.name}' en línea {node.lineno} contiene solo 'pass'.")
        
        # Detectar retornos genéricos sin lógica previa
        if len(body) == 1 and isinstance(body[0], ast.Return):
            self.issues.append(f"[SKELETON] Función '{node.name}' en línea {node.lineno} solo hace un return directo.")

    def visit_Assign(self, node):
        # Detectar posibles valores hardcodeados críticos
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ['balance', 'max_risk', 'account_id']:
                if isinstance(node.value, (ast.Constant, ast.Num)):
                    self.issues.append(f"[HARDCODED] Variable crítica '{target.id}' tiene un valor fijo en línea {node.lineno}.")

def audit_project(root_dir):
    report = []
    print(f"Scanning root: {os.path.abspath(root_dir)}")
    for root, _, files in os.walk(root_dir):
        if ".venv" in root or "venv" in root or "__pycache__" in root: continue
        
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        hunter = PlaceholderHunter(path)
                        hunter.visit(tree)
                        if hunter.issues:
                            report.append(f"\n--- Archivo: {path} ---")
                            report.extend(hunter.issues)
                    except Exception as e:
                        print(f"Error parsing {path}: {e}")
                        continue
    
    with open("audit_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"✅ Auditoría completada. {len(report)} secciones detectadas en audit_report.txt")

if __name__ == "__main__":
    audit_project(".")
