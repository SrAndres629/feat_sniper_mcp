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
TARGET_FILE = os.path.join(MT5_DATA_PATH, "Indicators", "FEAT", "FEAT_Visualizer.mq5")
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "compile_mql5.log")

def compile_file():
    print("--- MQL5 COMPILER AGENT ---")
    
    if not os.path.exists(EDITOR_PATH):
        print(f"❌ MetaEditor not found at: {EDITOR_PATH}")
        return

    if not os.path.exists(TARGET_FILE):
        print(f"❌ Target source file missing: {TARGET_FILE}")
        print("   -> Did you run the sync phase?")
        print("   -> Running sync now...")
        try:
             # Run sync module
             import tools.sync_mql5 as syncer
             syncer.sync()
        except Exception as e:
             print(f"   -> Sync failed: {e}")
             return

    # Prepare Log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    print(f"🔨 Compiling: {os.path.basename(TARGET_FILE)}")
    print(f"   [Editor]: {EDITOR_PATH}")
    
    # Command Structure: metaeditor64.exe /compile:"path" /log:"path"
    cmd = [
        EDITOR_PATH,
        f'/compile:{TARGET_FILE}',
        f'/log:{LOG_FILE}'
    ]
    
    try:
        # Run blocking
        subprocess.run(cmd, check=False)
        
        # Check Log
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-16') as f: # MetaEditor logs are usually UTF-16
                log_content = f.read()
                
            print("\n--- COMPILATION LOG ---")
            print(log_content)
            
            if "0 errors" in log_content:
                print("✅ COMPILATION SUCCESS")
                # Check .ex5
                ex5_path = TARGET_FILE.replace(".mq5", ".ex5")
                if os.path.exists(ex5_path):
                    print(f"   [+] Output generated: {ex5_path}")
            else:
                print("❌ COMPILATION FAILED")
        else:
            print("⚠️ No log file generated.")
            
    except Exception as e:
        print(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    compile_file()
