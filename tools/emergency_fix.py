
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SRE] - %(message)s')
logger = logging.getLogger("EmergencyFix")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def purge_weights():
    """Step 1: BRAIN WIPE"""
    logger.info(">>> STEP 1: BRAIN WIPE (Purging incompatible weights)...")
    dirs_to_purge = [
        os.path.join(BASE_DIR, "app", "ml", "weights"),
        os.path.join(BASE_DIR, "app", "ml", "checkpoints")
    ]
    
    deleted_count = 0
    for folder in dirs_to_purge:
        if not os.path.exists(folder):
            continue
        
        files = glob.glob(os.path.join(folder, "*.pth"))
        for f in files:
            try:
                os.remove(f)
                logger.info(f"Deleted: {f}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")
    
    if deleted_count == 0:
        logger.info("No .pth files found. Brain is already clean.")
    else:
        logger.info(f"Purged {deleted_count} corrupt weight files.")

def patch_mcp_server():
    """Step 2: CODE PATCH (NexusState)"""
    logger.info(">>> STEP 2: PATCHING mcp_server.py...")
    target_file = os.path.join(BASE_DIR, "mcp_server.py")
    
    if not os.path.exists(target_file):
        logger.error(f"File not found: {target_file}")
        return

    with open(target_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    inside_nexus_state = False
    inside_init = False
    patch_applied = False

    for i, line in enumerate(lines):
        new_lines.append(line)
        
        if "class NexusState:" in line:
            inside_nexus_state = True
        
        if inside_nexus_state and "def __init__(self):" in line:
            # Check if next lines already have self.running
            # We look ahead a few lines
            already_has_running = False
            for lookahead in lines[i+1:i+20]:
                if "self.running = True" in lookahead:
                    already_has_running = True
                    break
            
            if not already_has_running:
                logger.info("Injecting 'self.running = True' into NexusState.__init__")
                new_lines.append("        self.running = True\n")
                patch_applied = True
            else:
                logger.info("'self.running = True' already present.")
    
    if patch_applied:
        with open(target_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info("mcp_server.py patched successfully.")
    else:
        logger.info("mcp_server.py checks out. No changes needed.")

def patch_trade_mgmt():
    """Step 3: EMOJI REMOVAL"""
    logger.info(">>> STEP 3: SANITIZING trade_mgmt.py...")
    target_file = os.path.join(BASE_DIR, "app", "skills", "trade_mgmt.py")
    
    if not os.path.exists(target_file):
        logger.error(f"File not found: {target_file}")
        return

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "🕒" in content:
        logger.info("Found incompatible clock emoji. Sanitizing...")
        new_content = content.replace("🕒", "[WAIT]")
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.info("trade_mgmt.py sanitized.")
    else:
        logger.info("trade_mgmt.py is clean (No emojis found).")

def main():
    purge_weights()
    patch_mcp_server()
    patch_trade_mgmt()
    print("\n✅ SISTEMA REPARADO. EJECUTA nexus.bat AHORA.")

if __name__ == "__main__":
    main()
