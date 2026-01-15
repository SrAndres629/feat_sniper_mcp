import shutil
import os
import sys

def switch_account(mode):
    """
    Switch between Real and Demo accounts by swapping .env files.
    
    Args:
        mode (str): 'real' or 'demo'
    """
    base_path = "c:\\Users\\acord\\OneDrive\\Desktop\\Bot\\feat_sniper_mcp"
    source = os.path.join(base_path, f".env.{mode}")
    target = os.path.join(base_path, ".env")
    
    if not os.path.exists(source):
        print(f"❌ Error: Configuration file {source} not found!")
        return
    
    try:
        shutil.copy(source, target)
        print(f"✅ Switched to {mode.upper()} account successfully.")
        print(f"   Source: {source}")
        print(f"   Target: {target}")
    except Exception as e:
        print(f"❌ Error switching account: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['real', 'demo']:
        print("Usage: python switch_account.py [real|demo]")
        sys.exit(1)
    
    switch_account(sys.argv[1])
