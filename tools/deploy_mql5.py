import os
import shutil
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MQL5_Deployer")

# Configuration
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root of project
MT5_TERMINAL_ID = "065434634B76DD288A1DDF20131E8DDB"
MT5_BASE_PATH = os.path.join(os.getenv("APPDATA"), "MetaQuotes", "Terminal", MT5_TERMINAL_ID, "MQL5")

# Map local files to target directories (Experts, Indicators, etc.)
# KEY: Local Path (relative to project root), VALUE: Target Subdirectory in MQL5/
MANIFEST = {
    "FEAT_Sniper_Master_Core/FEAT_Visualizer.mq5": "Indicators/FEAT",
    "FEAT_Sniper_Master_Core/UnifiedModel_Main.ex5": "Experts/FEAT"
}

def deploy():
    logger.info("📂 [NEXUS] MQL5 Auto-Deploy Sequence Initiated...")
    
    if not os.path.exists(MT5_BASE_PATH):
        logger.error(f"❌ MT5 Data Folder not found: {MT5_BASE_PATH}")
        logger.warning("   Please verify your MT5 installation or Terminal ID.")
        return

    deploy_count = 0
    
    for local_rel_path, target_subdir in MANIFEST.items():
        local_path = os.path.join(SOURCE_DIR, local_rel_path)
        
        # Determine full target directory
        target_dir = os.path.join(MT5_BASE_PATH, target_subdir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        target_path = os.path.join(target_dir, os.path.basename(local_path))
        
        if os.path.exists(local_path):
            try:
                shutil.copy2(local_path, target_path)
                deploy_count += 1
                logger.info(f"   ✅ Installed: {os.path.basename(local_path)} -> {target_subdir}")
            except Exception as e:
                logger.error(f"   ❌ Failed to copy {os.path.basename(local_path)}: {e}")
        else:
            logger.warning(f"   ⚠️ Source file missing: {local_rel_path}")

    if deploy_count > 0:
        logger.info(f"🚀 [SUCCESS] Updated {deploy_count} MQL5 components.")
        logger.info("   -> Please 'Refresh' indicators in MT5 Navigator.")
    else:
        logger.info("ℹ️  No components updated (Files missing or up to date).")

if __name__ == "__main__":
    deploy()
