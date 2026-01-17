import os

def count_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except:
        return 0

results = []
for root, dirs, files in os.walk('.'):
    if '.venv' in dirs:
        dirs.remove('.venv')
    if '.git' in dirs:
        dirs.remove('.git')
        
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            lines = count_lines(path)
            if lines > 300:
                results.append((path, lines))

results.sort(key=lambda x: x[1], reverse=True)
print("| File Path | Lines |")
print("|-----------|-------|")
for path, lines in results:
    print(f"| {path} | {lines} |")
