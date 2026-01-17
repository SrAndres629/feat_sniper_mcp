import os
import re
import sys
from pathlib import Path

# --- Institutional Directives ---
PRIME_DIRECTIVES = {
    "NO_GHOSTING": [r"^\s*pass\b", r"\bTODO\b", r"\bFIXME\b"],
    "PHYSICS_FIRST": [r"@njit", r"np\."],
    "KELLY_LOCK": [r"calculate_dynamic_lot", r"risk_engine"]
}

# Files to ignore
IGNORE_FILES = ["enforce_constitution.py"]
IGNORE_DIRS = [".git", "__pycache__", ".ai", "venv", ".venv", "node_modules", "_archive_legacy", ".gemini", ".cursor"]
CRITICAL_CORE_PATHS = ["nexus_core/", "app/services/risk_engine.py", "app/ml/"]

def enforce_constitution():
    print("🛡️ IRON DOME: Executing Constitutional Enforcement...")
    violations = 0
    
    for root, dirs, files in os.walk("."):
        # Prune ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if not file.endswith((".py", ".mq5")):
                continue
            
            if file in IGNORE_FILES:
                continue
                
            file_path = Path(root) / file
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.splitlines()
                    
                    # 1. No-Ghosting Audit
                    for pattern in PRIME_DIRECTIVES["NO_GHOSTING"]:
                        # Use MULTILINE to ensure ^ works on every line
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            line_no = content[:match.start()].count("\n") + 1
                            match_text = match.group(0)
                            
                            # [LEVEL 64] False Positive Shield: Spanish 'todo' (meaning 'all')
                            # Match only if the preceding context is typical of a comment/code TODO
                            if "TODO" in match_text:
                                line_text = lines[line_no-1]
                                # If 'todo' is lowercase and followed by Spanish context, or not capitalized 'TODO:', skip
                                if "TODO:" not in line_text and "TODO " not in line_text:
                                    if any(word in line_text.lower() for word in ["mata todo", "para todo", "en todo"]):
                                        continue

                            print(f"❌ [VETOED] {file_path}:{line_no} -> Violation: '{pattern}' detected.")
                            violations += 1
                            
                    # 2. Nexus Core Optimization Audit
                    if "nexus_core" in str(file_path) and "@njit" not in content and "math_engine" in file:
                         print(f"⚠️ [WARNING] {file_path} -> Missing @njit optimization in core math.")
                         # Not a hard veto yet, but a strong warning
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if violations > 0:
        print(f"\n🚫 CONSTITUTIONAL BREACH: {violations} violations found. Commit ABORTED.")
        return False
    
    print("\n✅ IRON DOME: Codebase is constitutionally aligned. Level 64 Integrity LOCKED.")
    return True

if __name__ == "__main__":
    if not enforce_constitution():
        sys.exit(1)
