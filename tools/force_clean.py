
import os
import shutil

def force_delete(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"✅ DELETED FILE: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"✅ DELETED DIR: {path}")
        else:
            print(f"⚠️ NOT FOUND: {path}")
    except Exception as e:
        print(f"❌ ERROR deleting {path}: {e}")

def main():
    root = os.getcwd()
    
    # List of targets to destroy
    targets = [
        "app/ml/fuzzy_logic.py",
        "app/ml/fuzzy_engine.py",
        "nexus_training/ghost_test_spatial.py",
        "nexus_core/mtf_engine" 
    ]
    
    print("⚔️ STARTING PURGE PROTOCOL...")
    
    for target in targets:
        full_path = os.path.join(root, target)
        force_delete(full_path)
        
    print("🏁 PURGE COMPLETE.")

if __name__ == "__main__":
    main()
