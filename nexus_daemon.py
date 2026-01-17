import asyncio
import subprocess
import sys
import os
import time
import logging
import signal
from datetime import datetime

# Import the Engine
from app.core.nexus_engine import engine
from app.core.config import settings

# --- DAEMON CONFIG ---
DAEMON_VERSION = "2.5 (Immortal Core)"
LOG_FILE = "logs/nexus_daemon.log"

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [CORE-DAEMON] | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("NexusDaemon")

class ImmortalDaemon:
    def __init__(self):
        self.mcp_process = None
        self.dashboard_process = None
        self.running = True
        self.venv_python = os.path.join(".venv", "Scripts", "python.exe")
        if not os.path.exists(self.venv_python):
            self.venv_python = sys.executable

    async def run_engine(self):
        """Launches the primary Trading Engine as a persistent task."""
        try:
            await engine.initialize()
            logger.info("🛡️ TRADING ENGINE: Actively monitoring XAUUSD.")
        except Exception as e:
            logger.critical(f"🔥 Engine Fallback: {e}")
            self.running = False

    def setup_env(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONWARNINGS"] = "ignore"
        return env

    async def supervise_children(self):
        """Monitors and restarts Diplomatic Interface and Dashboard."""
        env = self.setup_env()
        
        while self.running:
            # 1. MCP Interface
            if not self.mcp_process or self.mcp_process.poll() is not None:
                logger.info("📻 Launching Diplomatic Interface (MCP)...")
                self.mcp_process = subprocess.Popen(
                    [self.venv_python, "mcp_server.py"],
                    env=env, creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            
            # 2. Streamlit Dashboard
            if not self.dashboard_process or self.dashboard_process.poll() is not None:
                logger.info("🎨 Launching Visual Cortex (Dashboard)...")
                self.dashboard_process = subprocess.Popen(
                    [self.venv_python, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.headless", "true"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            
            await asyncio.sleep(10) # Supervision cycle

    async def main_loop(self):
        logger.info(f"⚔️ NEXUS DAEMON v{DAEMON_VERSION} INITIALIZED ⚔️")
        
        # Launch Engine as a background task
        engine_task = asyncio.create_task(self.run_engine())
        # Launch Supervision
        supervision_task = asyncio.create_task(self.supervise_children())
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self):
        logger.info("🛑 Initiating Global Shutdown...")
        self.running = False
        await engine.shutdown()
        if self.mcp_process: self.mcp_process.terminate()
        if self.dashboard_process: self.dashboard_process.terminate()
        logger.info("👋 Shutdown Complete.")

if __name__ == "__main__":
    daemon = ImmortalDaemon()
    try:
        asyncio.run(daemon.main_loop())
    except KeyboardInterrupt:
        pass
