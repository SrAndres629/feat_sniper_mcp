import os
import subprocess
import time
import shutil

# --- CONFIGURATION ---
EDITOR_PATH = r"C:\Program Files\LiteFinance MT5 Terminal\metaeditor64.exe"
PROJECT_ROOT = os.getcwd()

# Data Folder (from sync_mql5.py)
MT5_TERMINAL_ID = "065434634B76DD288A1DDF20131E8DDB"
MT5_DATA_PATH = os.path.join(os.getenv("APPDATA"), "MetaQuotes", "Terminal", MT5_TERMINAL_ID, "MQL5")

# Files to Compile (Target paths in MT5 Data Folder)
TARGETS = [
    os.path.join(MT5_DATA_PATH, "Indicators", "FEAT", "FEAT_Visualizer.mq5"),
    os.path.join(MT5_DATA_PATH, "Experts", "FEAT", "UnifiedModel_Main.mq5")
]
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "compile_mql5.log")

def compile_file():
    print("--- MQL5 COMPILER AGENT ---")
    
    if not os.path.exists(EDITOR_PATH):
        print(f"❌ MetaEditor not found at: {EDITOR_PATH}")
        return

    for target_file in TARGETS:
        if not os.path.exists(target_file):
            print(f"❌ Target source file missing: {target_file}")
            continue

        print(f"🔨 Compiling: {os.path.basename(target_file)}")
        
        # Command Structure: metaeditor64.exe /compile:"path" /log:"path"
        cmd = [
            EDITOR_PATH,
            f'/compile:{target_file}',
            f'/log:{LOG_FILE}'
        ]
        
        try:
            # Run blocking
            subprocess.run(cmd, check=False)
            
            # Check Log
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-16') as f: 
                    log_content = f.read()
                    
                if "0 errors" in log_content:
                    print(f"   ✅ SUCCESS: {os.path.basename(target_file)}")
                else:
                    print(f"   ❌ FAILED: {os.path.basename(target_file)}")
                    print(log_content)
            else:
                print("⚠️ No log file generated.")
                
        except Exception as e:
            print(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    compile_file()
