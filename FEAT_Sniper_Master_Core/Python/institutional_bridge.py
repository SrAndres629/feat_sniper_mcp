"""
institutional_bridge.py - Automated MQL5 <> Python Bridge
Monitors for new MQL5 data, triggers the ML/Optimization pipeline,
and automatically deploys updated parameters back to MT5.

Features:
- File System Watching (Watchdog)
- Auto-execution of run_pipeline.py
- Smart file moving/copying
- Audio/Visual alerts on completion

Usage:
    python institutional_bridge.py --watch-dir "C:/Path/To/MT5/Files" --deploy-dir "C:/Path/To/MT5/Common/Files"
"""

import time
import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import json

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    print("Installing watchdog...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BRIDGE] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class BridgeHandler(FileSystemEventHandler):
    def __init__(self, pipeline_script, deploy_dir):
        self.pipeline_script = pipeline_script
        self.deploy_dir = deploy_dir
        self.last_trigger = 0
        self.cooldown = 5  # Seconds between triggers to avoid duplicates

    def on_created(self, event):
        self._process_event(event)

    def on_modified(self, event):
        self._process_event(event)

    def _process_event(self, event):
        if event.is_directory:
            return
        
        filename = os.path.basename(event.src_path)
        
        # Only React to specific export files from UnifiedModel
        if not (filename.startswith("UnifiedModel_Export") and filename.endswith(".csv")):
            return

        # Cooldown check
        if time.time() - self.last_trigger < self.cooldown:
            return
        self.last_trigger = time.time()

        logging.info(f"⚡ Detected new data: {filename}")
        self.run_pipeline(event.src_path)

    def run_pipeline(self, data_path):
        try:
            # Parse symbol/timeframe from filename if standard format: UnifiedModel_Export_EURUSD_H1.csv
            # Fallback defaults
            symbol = "EURUSD"
            timeframe = "H1"
            
            parts = os.path.basename(data_path).split('_')
            if len(parts) >= 4:
                symbol = parts[2]
                timeframe = parts[3].replace('.csv', '')

            logging.info(f"🚀 Launching Pipeline for {symbol} {timeframe}...")
            
            # Execute Pipeline
            cmd = [
                sys.executable, 
                self.pipeline_script,
                "--symbol", symbol,
                "--timeframe", timeframe,
                "--data", data_path,
                "--output", os.path.dirname(self.pipeline_script) # Output to Python dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info("✅ Pipeline Success!")
                self.deploy_artifacts()
            else:
                logging.error(f"❌ Pipeline Failed:\n{result.stderr}")

        except Exception as e:
            logging.error(f"Critical Bridge Error: {str(e)}")

    def deploy_artifacts(self):
        """Deploy generated .txt files to MT5 Common/Files if configured."""
        if not self.deploy_dir or not os.path.exists(self.deploy_dir):
            logging.warning("⚠️ Deploy directory not configured or invalid. Skipping auto-deploy.")
            return

        source_dir = os.path.dirname(self.pipeline_script)
        artifacts = ["ml_thresholds.txt", "optuna_calibration.txt"]
        
        for art in artifacts:
            src = os.path.join(source_dir, art)
            if os.path.exists(src):
                dst = os.path.join(self.deploy_dir, art)
                try:
                    shutil.copy2(src, dst)
                    logging.info(f"📤 Deployed: {art} -> {self.deploy_dir}")
                except Exception as e:
                    logging.error(f"Failed to deploy {art}: {e}")

def main():
    # Configuration - Load from file or user interaction
    config_file = "bridge_config.json"
    config = {}
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    pipeline_script = os.path.abspath("run_pipeline.py")
    
    # Interactive Setup if missing config
    if "watch_dir" not in config:
        print("\n=== 🌉 Institutional Bridge Setup ===")
        print("Enter the path where MT5 exports CSV files.")
        print("Usually: C:\\Users\\...\\AppData\\Roaming\\MetaQuotes\\Terminal\\...\\MQL5\\Files")
        watch_dir = input("Watch Directory: ").strip().strip('"')
        
        print("\nEnter path to Deploy parameters (Common/Files).")
        print("Usually: C:\\Users\\...\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files")
        deploy_dir = input("Deploy Directory: ").strip().strip('"')
        
        config = {"watch_dir": watch_dir, "deploy_dir": deploy_dir}
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
    watch_dir = config["watch_dir"]
    deploy_dir = config["deploy_dir"]

    if not os.path.exists(watch_dir):
        logging.error(f"Watch directory does not exist: {watch_dir}")
        return

    observer = Observer()
    handler = BridgeHandler(pipeline_script, deploy_dir)
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    
    logging.info(f"👀 Bridge Active. Watching: {watch_dir}")
    logging.info(f"📂 Auto-Deploy Target: {deploy_dir}")
    logging.info("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
