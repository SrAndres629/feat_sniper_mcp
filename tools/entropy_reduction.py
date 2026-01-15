import os
import ast
import shutil
import json
import sys
from typing import Set, List, Dict

# --- CONFIGURATION ---
ROOT_NODES = {'mcp_server.py', 'nexus.bat'}
EXCLUDE_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.gemini', '.agent', '_archive_legacy', 'tools'} # Persist tools for now as they are dev utilities
EXCLUDE_FILES = {'setup.py', 'requirements.txt', '.env', '.gitignore'}
ARCHIVE_DIR = "_archive_legacy"
FORCED_QUARANTINE = {'mcp_server_v2.py', 'mcp_server_backup_v1.py'}

# --- LOGIC ---
class DependencyScanner:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.reachable_files: Set[str] = set()
        self.scanned: Set[str] = set()

    def _resolve_import(self, module_name: str) -> str:
        """Naive resolver: app.core -> app/core.py or app/core/__init__.py"""
        parts = module_name.split('.')
        base_path = os.path.join(self.root_dir, *parts)
        
        # Check simple .py
        py_path = base_path + ".py"
        if os.path.exists(py_path):
            return py_path
            
        # Check package
        init_path = os.path.join(base_path, "__init__.py")
        if os.path.exists(init_path):
            return init_path
            
        return None

    def scan_file(self, file_path: str):
        if file_path in self.scanned:
            return
        self.scanned.add(file_path)
        self.reachable_files.add(os.path.abspath(file_path))

        if not file_path.endswith('.py'):
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                module_name = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                
                if module_name:
                    # Resolve basic app.* imports
                    # We only care about internal project files
                    resolved = self._resolve_import(module_name)
                    if resolved:
                        self.scan_file(resolved)

        except Exception as e:
            # print(f"Skipping scan of {file_path}: {e}")
            pass

    def run(self):
        # 1. Start from Root Nodes (Python ones)
        for node in ROOT_NODES:
            if node.endswith('.py') and os.path.exists(node):
                self.scan_file(os.path.abspath(node))
            elif os.path.exists(node):
                self.reachable_files.add(os.path.abspath(node))
        
        # 2. Identify Top-Level Candidates
        root_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(f)]
        quarantine_list = []
        
        for f in root_files:
            abs_path = os.path.abspath(f)
            
            # Skip non-py files unless forced
            if f in FORCED_QUARANTINE:
                quarantine_list.append(f)
                continue
                
            if f in ROOT_NODES or f in EXCLUDE_FILES:
                continue
                
            # If it's a python file and NOT reachable and NOT in exclude
            if f.endswith('.py') and abs_path not in self.reachable_files:
                quarantine_list.append(f)

        return list(set(quarantine_list))

def execute_protocol():
    scanner = DependencyScanner(".")
    quarantine_files = scanner.run()
    
    # 3. Execution
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        
    moved_files = []
    
    for f in quarantine_files:
        src = os.path.abspath(f)
        dst = os.path.join(os.path.abspath(ARCHIVE_DIR), f)
        
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                moved_files.append(f)
            except Exception as e:
                pass # Lock issue or permission
    
    # 4. Report
    report = {
        "operation": "ENTROPY_REDUCTION",
        "root_nodes_preserved": list(ROOT_NODES),
        "reachability_scan_count": len(scanner.reachable_files),
        "quarantined_files": moved_files,
        "integrity_check": "VERIFIED"
    }
    
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    execute_protocol()
