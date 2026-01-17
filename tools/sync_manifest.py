import os
import hashlib
import json
from pathlib import Path

# --- Configuration ---
CRITICAL_EXTENSIONS = {".py", ".mq5", ".md", ".json"}
IGNORE_DIRS = {".git", "__pycache__", "venv", "node_modules", ".gemini"}
MANIFEST_PATH = Path(".ai/memory/manifest.json")

def calculate_hash(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def sync_manifest():
    print("🔄 IRON DOME: Synchronizing Active Memory Manifest...")
    
    manifest_data = {
        "protocol": "Iron Dome v3.5",
        "lock_status": "ENFORCED",
        "files_audit": []
    }
    
    for root, dirs, files in os.walk("."):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if not any(file.endswith(ext) for ext in CRITICAL_EXTENSIONS):
                continue
            
            # Skip the manifest itself
            if "manifest.json" in file:
                continue
                
            file_path = Path(root) / file
            rel_path = str(file_path.relative_to(Path("."))).replace("\\", "/")
            
            try:
                manifest_data["files_audit"].append({
                    "path": rel_path,
                    "hash": calculate_hash(file_path),
                    "size_bytes": file_path.stat().st_size
                })
            except Exception as e:
                print(f"Error hashing {rel_path}: {e}")

    # Ensure memory directory exists
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
        
    print(f"✅ IRON DOME: Manifest synchronized. {len(manifest_data['files_audit'])} critical files mapped.")

if __name__ == "__main__":
    sync_manifest()
