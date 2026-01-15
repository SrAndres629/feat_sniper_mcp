import ast
import os
import json
import logging
import logging
import warnings

# Filter noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from pydantic.warnings import PydanticDeprecatedSince212
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince212)
except ImportError:
    pass

from typing import Set, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EXCLUDED_DIRS = {".git", "__pycache__", "venv", ".venv", "tests", "docs", "tools", "data", "logs"}
PROJECT_ROOT = "app"
ROOT_FILES = ["mcp_server.py", "nexus_control.py", "nexus_auditor.py"]

def get_python_files(root_dir: str) -> List[str]:
    """Recursively find all Python files in the given directory."""
    py_files = []
    
    # 1. Add root files
    for rf in ROOT_FILES:
        if os.path.exists(rf):
            py_files.append(rf)
            
    # 2. Walk directory
    for root, dirs, files in os.walk(root_dir):
        # Filter excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

def get_module_name(file_path: str) -> str:
    """Convert file path to dot-notation module name."""
    # Normalize path separators
    file_path = file_path.replace("\\", "/")
    # Remove extension
    if file_path.endswith(".py"):
        file_path = file_path[:-3]
    # Replace slashes with dots
    return file_path.replace("/", ".")

def analyze_imports(file_path: str) -> List[str]:
    """Parse a file and extract all imported module names."""
    imports = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                imports.append(module)
                
    except Exception as e:
        logging.warning(f"Could not parse {file_path}: {e}")
    return imports

def build_architecture_map():
    """Scan the project and build the dependency map."""
    logging.info("🗺️  Starting Nexus Cartographer Scan...")
    
    if not os.path.exists(PROJECT_ROOT):
        logging.error(f"Project root '{PROJECT_ROOT}' not found.")
        return

    all_files = get_python_files(PROJECT_ROOT)
    module_map: Dict[str, Dict[str, Any]] = {}
    
    # 1. Map all modules and their imports
    for file_path in all_files:
        module_name = get_module_name(file_path)
        imports = analyze_imports(file_path)
        module_map[module_name] = {
            "file_path": file_path,
            "imports": imports,
            "imported_by": [] 
        }

    # 2. Determine "imported_by" (reverse dependencies)
    for module, data in module_map.items():
        for imp in data["imports"]:
            # Handle relative imports or sub-module imports
            # Check if 'imp' matches any known module
            # Logic: If module A imports B, add A to B.imported_by
            
            # Direct match
            if imp in module_map:
                module_map[imp]["imported_by"].append(module)
            else:
                # Check for partial matches (e.g. from app.core import config -> imports app.core)
                # This is a simplified check.
                pass

    # 3. Identify Orphans (Modules in 'app' that are never imported)
    # We exclude main entry points if known, but generally library code should be imported.
    orphans = []
    for module, data in module_map.items():
        # Exclude entry points (like main.py or __init__.py if top level) from being strictly orphans?
        # For this strict audit, we list anything with 0 importers.
        if not data["imported_by"]:
            # Check if it's likely a script or entry point
            if "main" in module or "__init__" in module:
                continue 
            orphans.append(module)

    output = {
        "modules": list(module_map.keys()),
        "dependency_graph": module_map,
        "orphans": orphans,
        "scan_timestamp": "NOW" # Placeholder
    }

    with open("architecture_map.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
        
    logging.info(f"✅ Map generated. Found {len(all_files)} modules.")
    logging.info(f"🔍 Orphans detected: {len(orphans)}")
    if orphans:
        for o in orphans:
            logging.warning(f"   - ORPHAN: {o}")

if __name__ == "__main__":
    build_architecture_map()
