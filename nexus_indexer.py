import os
import ast
import json
import logging

# Config
IGNORE_DIRS = {'.git', 'venv', '.venv', '__pycache__', '.vscode', 'archive', 'node_modules', 'site-packages'}
OUTPUT_FILE = "code_index.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NexusIndexer")

def parse_file(filepath):
    """Extracts classes, functions, and docstrings from a python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content)
        definitions = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.append({
                    "type": "function",
                    "name": node.name,
                    "doc": ast.get_docstring(node),
                    "lineno": node.lineno
                })
            elif isinstance(node, ast.ClassDef):
                definitions.append({
                    "type": "class",
                    "name": node.name,
                    "doc": ast.get_docstring(node),
                    "lineno": node.lineno
                })
        return definitions
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return []

def scan_project(root_dir):
    """Recursively scans the project and builds the index."""
    index = {}
    
    for root, dirs, files in os.walk(root_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, root_dir)
                
                defs = parse_file(path)
                if defs:
                    index[rel_path] = defs
                    
    return index

if __name__ == "__main__":
    logger.info("🧠 Starting Cognitive Cortex Indexing...")
    root = os.getcwd()
    knowledge_base = scan_project(root)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2)
        
    logger.info(f"✅ Indexing Complete. Knowledge stored in {OUTPUT_FILE}")
    logger.info(f"Indexed {len(knowledge_base)} files.")
