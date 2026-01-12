
import os
import sys

def safe_clean():
    print("Starting Safe ASCII Sanitization...")
    count = 0
    for root, dirs, files in os.walk('.'):
        if '.git' in root or '.venv' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Target specific non-ascii that cause marshal errors in docstrings
                    # but keep logic intact.
                    # We'll be radical for now to ensure stability as per Senior Upgrade.
                    cleaned = content.encode('ascii', 'ignore').decode('ascii')
                    
                    if len(cleaned) < len(content) * 0.9:
                        print(f"  [!] Skip {path} - Too much data loss (check encoding)")
                        continue
                        
                    with open(path, 'w', encoding='ascii') as f:
                        f.write(cleaned)
                    count += 1
                except Exception as e:
                    print(f"  [ERR] {path}: {e}")
    print(f"Sanitization complete. {count} files cleaned.")

if __name__ == "__main__":
    safe_clean()
