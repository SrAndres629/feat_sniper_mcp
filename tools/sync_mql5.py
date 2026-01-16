import os
import shutil
import logging
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MQL5_Sync")

# Configuration
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
MT5_TERMINAL_ID = "065434634B76DD288A1DDF20131E8DDB"
MT5_BASE_PATH = os.path.join(os.getenv("APPDATA"), "MetaQuotes", "Terminal", MT5_TERMINAL_ID, "MQL5")

MANIFEST = {
    "FEAT_Sniper_Master_Core/FEAT_Visualizer.mq5": "Indicators",
    "FEAT_Sniper_Master_Core/UnifiedModel_Main.mq5": "Experts",
    "FEAT_Sniper_Master_Core/Include/UnifiedModel/CInterop.mqh": "Include/UnifiedModel",
    "FEAT_Sniper_Master_Core/Include/UnifiedModel/CVisuals.mqh": "Include/UnifiedModel",
    "FEAT_Sniper_Master_Core/Include/UnifiedModel/CFSM.mqh": "Include/UnifiedModel"
}

def sync():
    print(" [SYNC] Executing MQL5 Synchronization...")
    print(f"        Target: .../Terminal/{MT5_TERMINAL_ID}/MQL5")
    
    if not os.path.exists(MT5_BASE_PATH):
        print(f" [ERROR] MT5 Data Folder not found: {MT5_BASE_PATH}")
        return

    updates = 0
    warnings = 0

    for local_rel_path, target_subdir in MANIFEST.items():
        local_path = os.path.join(SOURCE_DIR, local_rel_path)
        target_dir = os.path.join(MT5_BASE_PATH, target_subdir)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        filename = os.path.basename(local_path)
        target_path = os.path.join(target_dir, filename)
        
        if not os.path.exists(local_path):
             print(f" [WARN] Source missing: {local_rel_path}")
             continue

        # Check for updates
        needs_update = False
        if not os.path.exists(target_path):
            needs_update = True
            reason = "New File"
        else:
            src_mtime = os.path.getmtime(local_path)
            dst_mtime = os.path.getmtime(target_path)
            if src_mtime > dst_mtime:
                needs_update = True
                reason = "Updated"
        
        if needs_update:
            try:
                shutil.copy2(local_path, target_path)
                updates += 1
                print(f"   [+] {filename} -> {target_subdir} ({reason})")
                
                # Compilation Check for .mq5
                if filename.endswith(".mq5"):
                    ex5_name = filename.replace(".mq5", ".ex5")
                    ex5_target = os.path.join(target_dir, ex5_name)
                    
                    if not os.path.exists(ex5_target):
                         print(f"   ⚠️  [MISSING BINARY] {ex5_name} not found. Open MetaEditor and compile!")
                         warnings += 1
                    elif os.path.getmtime(target_path) > os.path.getmtime(ex5_target):
                         print(f"   ⚠️  [OUTDATED BINARY] {ex5_name} is older than source. Recompile needed!")
                         warnings += 1
                    else:
                        print(f"   [OK] {ex5_name} verified.")
            except Exception as e:
                print(f"   [X] Failed to copy {filename}: {e}")
        else:
             # Check compilation even if no update needed
             if filename.endswith(".mq5"):
                ex5_name = filename.replace(".mq5", ".ex5")
                ex5_target = os.path.join(target_dir, ex5_name)
                
                if not os.path.exists(ex5_target):
                    print(f"   ⚠️  [MISSING BINARY] {ex5_name} not found. Open MetaEditor and compile!")
                    warnings += 1
                elif os.path.getmtime(target_path) > os.path.getmtime(ex5_target):
                    print(f"   ⚠️  [OUTDATED BINARY] {ex5_name} is older than source. Recompile needed!")
                    warnings += 1

    
    # MetaEditor Path
    METAEDITOR_PATH = r"C:\Program Files\LiteFinance MT5 Terminal\metaeditor64.exe"
    if not os.path.exists(METAEDITOR_PATH):
        # Fallback for standard install
        METAEDITOR_PATH = r"C:\Program Files\MetaTrader 5\metaeditor64.exe"

    if not os.path.exists(METAEDITOR_PATH):
        print(f" [WARN] MetaEditor not found. Auto-compilation disabled.")
        METAEDITOR_PATH = None

    def compile_mql5(file_path):
        if not METAEDITOR_PATH: return False
        try:
            print(f"   [BUILD] Compiling {os.path.basename(file_path)}...")
            import subprocess
            cmd = [METAEDITOR_PATH, f"/compile:{file_path}", "/log"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   [SUCCESS] Compiled {os.path.basename(file_path)}")
                return True
            else:
                print(f"   [ERROR] Compilation Failed for {file_path}")
                return False
        except Exception as e:
            print(f"   [ERROR] Compiler execution failed: {e}")
            return False

    if updates == 0:
        print(" [OK] Source files are up-to-date.")
    
    # Always check binary status and compile if needed
    for local_rel_path, target_subdir in MANIFEST.items():
        if not local_rel_path.endswith(".mq5"): continue
        
        filename = os.path.basename(local_rel_path)
        target_dir = os.path.join(MT5_BASE_PATH, target_subdir)
        target_path = os.path.join(target_dir, filename)
        ex5_name = filename.replace(".mq5", ".ex5")
        ex5_target = os.path.join(target_dir, ex5_name)
        
        should_compile = False
        if not os.path.exists(ex5_target):
            print(f"   [MISSING] {ex5_name} not found.")
            should_compile = True
        elif os.path.getmtime(target_path) > os.path.getmtime(ex5_target):
             print(f"   [OUTDATED] {ex5_name} needs rebuild.")
             should_compile = True
             
        if should_compile:
            compile_mql5(target_path)
        else:
             print(f"   [VERIFIED] {ex5_name} is current.")

if __name__ == "__main__":
    sync()
