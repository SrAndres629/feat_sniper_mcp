
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PURGE] - %(message)s')
logger = logging.getLogger(__name__)

def purge_directory(path):
    """Deletes all .pth files in the specified directory."""
    if not os.path.exists(path):
        logger.warning(f"Directory not found: {path}")
        return

    files = glob.glob(os.path.join(path, "*.pth"))
    if not files:
        logger.info(f"No .pth files found in {path}")
        return

    logger.info(f"Found {len(files)} files in {path}. Purging...")
    for file_path in files:
        try:
            os.remove(file_path)
            logger.info(f"Deleted: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "app", "ml", "weights")
    checkpoints_dir = os.path.join(base_dir, "app", "ml", "checkpoints")

    logger.info("Initiating BRAIN WIPE Protocol...")
    
    purge_directory(weights_dir)
    purge_directory(checkpoints_dir)

    logger.info("BRAIN WIPE Complete. Neural state reset to TABULA RASA.")

if __name__ == "__main__":
    main()
