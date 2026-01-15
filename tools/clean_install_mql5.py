import os
import shutil

# Identify correct terminal path from sync_mql5.py logic
# Hardcoded for now based on previous context 
MT5_TERMINAL_ID = "065434634B76DD288A1DDF20131E8DDB"
MT5_BASE_PATH = os.path.join(os.getenv("APPDATA"), "MetaQuotes", "Terminal", MT5_TERMINAL_ID, "MQL5")

LEGACY_PATHS = [
    os.path.join(MT5_BASE_PATH, "Indicators", "FEAT_Sniper"),
    os.path.join(MT5_BASE_PATH, "Experts", "FEAT_Sniper"),
    os.path.join(MT5_BASE_PATH, "Indicators", "FEAT_Sniper_Master_Core"), # Just in case
]

def clean():
    print("--- MQL5 CLEANUP AGENT ---")
    print(f"Target Base: {MT5_BASE_PATH}")
    
    if not os.path.exists(MT5_BASE_PATH):
        print("❌ MT5 Path not found.")
        return

    for path in LEGACY_PATHS:
        if os.path.exists(path):
            print(f"🗑️ REMOVING LEGACY: {path}")
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                print("   [OK] Deleted.")
            except Exception as e:
                print(f"   [ERROR] Failed to delete: {e}")
        else:
            print(f"   [SKIP] Not found: {os.path.basename(path)}")
            
    print("✅ Cleanup Complete.")

if __name__ == "__main__":
    clean()
