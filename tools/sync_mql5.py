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
    "FEAT_Sniper_Master_Core/FEAT_Visualizer.mq5": "Indicators/FEAT",
    "FEAT_Sniper_Master_Core/UnifiedModel_Main.ex5": "Experts/FEAT"
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
                    if not os.path.exists(ex5_target) or os.path.getmtime(target_path) > os.path.getmtime(ex5_target):
                        print(f"   ⚠️  [COMPILE REQUIRED] {ex5_name} is outdated/missing in MT5!")
                        warnings += 1
            except Exception as e:
                print(f"   [X] Failed to copy {filename}: {e}")
        else:
            # Check compilation even if no update needed
             if filename.endswith(".mq5"):
                ex5_name = filename.replace(".mq5", ".ex5")
                ex5_target = os.path.join(target_dir, ex5_name)
                if not os.path.exists(ex5_target) or os.path.getmtime(target_path) > os.path.getmtime(ex5_target):
                    print(f"   ⚠️  [COMPILE REQUIRED] {ex5_name} needs compilation in MetaEditor.")
                    warnings += 1

    if updates == 0:
        print(" [OK] All MQL5 components are up-to-date.")
    
    if warnings > 0:
        print(f" [WARN] {warnings} Compilation Warnings detected. Open MetaEditor!")

if __name__ == "__main__":
    sync()
