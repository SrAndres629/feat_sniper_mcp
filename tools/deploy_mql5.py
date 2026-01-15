import os
import subprocess
import sys

def run_step(script_name, description):
    print(f"\n>>> [STEP] {description} ({script_name})")
    try:
        cmd = [sys.executable, f"tools/{script_name}"]
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"❌ STEP FAILED: {script_name}")
            sys.exit(ret)
    except Exception as e:
        print(f"❌ EXECUTION ERROR: {e}")
        sys.exit(1)

def deploy():
    print("==========================================")
    print("   FEAT NEXUS: MQL5 DEPLOYMENT PROTOCOL   ")
    print("==========================================")
    
    # 1. Cleanup
    run_step("clean_install_mql5.py", "Cleaning Legacy Artifacts")
    
    # 2. Sync
    run_step("sync_mql5.py", "Synchronizing Source Code")
    
    # 3. Compile
    run_step("compile_mql5.py", "Compiling Indicators")
    
    print("\n==========================================")
    print("✅ DEPLOYMENT SUCCESSFUL")
    print("==========================================")

if __name__ == "__main__":
    deploy()
