
import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GARBAGE_FILES = [
    "test_error.txt",
    "violations.txt",
    "train_log.txt",
    "audit_lines.py",
    "dashboard.html" # Root dashboard is often legacy if using streamlit
]

GARBAGE_DIRS = [
    ".ai/logs",
    ".ai/context",
    "__pycache__"
]

def clean_garbage():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Clear Files
    for f in GARBAGE_FILES:
        f_path = os.path.join(root_dir, f)
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
                logging.info(f"Deleted garbage file: {f}")
            except Exception as e:
                logging.error(f"Failed to delete {f}: {e}")

    # 2. Clear Directories
    for d in GARBAGE_DIRS:
        d_path = os.path.join(root_dir, d)
        if os.path.exists(d_path):
            try:
                # We often want to clear content but keep the dir for some .ai paths
                # But for __pycache__ we delete entirely.
                if "__pycache__" in d:
                    shutil.rmtree(d_path)
                else:
                    for filename in os.listdir(d_path):
                        file_path = os.path.join(d_path, filename)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                logging.info(f"Cleaned garbage directory: {d}")
            except Exception as e:
                logging.error(f"Failed to clean {d}: {e}")

if __name__ == "__main__":
    logging.info("Starting AI Architectural Cleanup...")
    clean_garbage()
    logging.info("Cleanup complete. Sovereignty restored.")
