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
    format='%(asctime)s | [%(name)s] | %(levelname)s | %(message)s',
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
        """Monitors and restarts Diplomatic Interface and Dashboard with absolute paths."""
        env = self.setup_env()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        mcp_script = os.path.join(script_dir, "mcp_server.py")
        dashboard_script = os.path.join(script_dir, "dashboard", "war_room.py")

        while self.running:
            # 1. MCP Interface
            if not self.mcp_process or self.mcp_process.poll() is not None:
                if self.mcp_process and self.mcp_process.poll() is not None:
                    logger.warning("📻 MCP Interface died. Restarting...")
                
                logger.info(f"📻 Launching Diplomatic Interface (MCP) from {mcp_script}")
                self.mcp_process = subprocess.Popen(
                    [self.venv_python, mcp_script],
                    env=env,
                    cwd=script_dir
                )
            
            # 2. Streamlit Dashboard
            if not self.dashboard_process or self.dashboard_process.poll() is not None:
                if self.dashboard_process and self.dashboard_process.poll() is not None:
                    logger.warning("🎨 Visual Cortex (Dashboard) died. Restarting...")

                logger.info(f"🎨 Launching Visual Cortex (Dashboard) from {dashboard_script}")
                self.dashboard_process = subprocess.Popen(
                    [self.venv_python, "-m", "streamlit", "run", dashboard_script, "--server.port", "8501"],
                    env=env, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    cwd=script_dir
                )
                # Auto-open dashboard in browser
                asyncio.create_task(self._open_browser_delayed("http://localhost:8501"))
            
            await asyncio.sleep(10) # Supervision cycle

    async def _open_browser_delayed(self, url: str):
        """Opens the browser after a short delay to ensure the server is up."""
        await asyncio.sleep(5)
        import webbrowser
        webbrowser.open(url)

    def cleanup_zombies(self):
        """Kills any lingering mcp_server or streamlit processes using surgical image filters."""
        try:
            logger.info("🧟 Hunting Zombie Processes...")
            # More specific taskkill filters to avoid collateral damage
            subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq *mcp_server*", "/T"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.run(["taskkill", "/F", "/FI", "WINDOWTITLE eq *streamlit*", "/T"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            
            # Also check by command line via wmic if needed, but taskkill by window title is usually enough for these scripts
        except Exception as e:
            logger.debug(f"Zombie hunt had minor friction: {e}")

    async def main_loop(self):
        logger.info(f"⚔️ NEXUS DAEMON v{DAEMON_VERSION} INITIALIZED ⚔️")
        
        # Kill Zombies first
        self.cleanup_zombies()

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
    # Windows Selector Policy Fix
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    daemon = ImmortalDaemon()
    try:
        asyncio.run(daemon.main_loop())
    except KeyboardInterrupt:
        pass
