"""
FEAT SNIPER: Background Workers
================================
Manages async background tasks for long-running operations.
"""

import asyncio
import subprocess
import sys
import os
import json
import logging
from datetime import datetime
from typing import Optional, Callable
from queue import Queue
from threading import Thread

logger = logging.getLogger("API.Workers")


class SimulationWorker:
    """
    Manages simulation processes in the background.
    Provides status tracking and log streaming.
    """
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status = {
            "running": False,
            "current_episode": 0,
            "total_episodes": 0,
            "current_balance": 20.0,
            "start_time": None,
            "last_update": None
        }
        self.log_queue: Queue = Queue()
        self.status_file = "data/simulation_status.json"
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Determine Python executable
        venv_python = os.path.join(".venv", "Scripts", "python.exe")
        self.python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Script paths
        self.script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.sim_script = os.path.join(self.script_dir, "nexus_training", "simulate_warfare.py")
    
    async def start(self, episodes: int = 5, mode: str = "adversarial") -> bool:
        """Starts a simulation in the background."""
        if self.is_running():
            logger.warning("Simulation already running")
            return False
        
        logger.info(f"🎓 Starting simulation: {episodes} episodes, mode={mode}")
        
        self.status = {
            "running": True,
            "current_episode": 0,
            "total_episodes": episodes,
            "current_balance": 20.0,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat()
        }
        
        # Launch subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        self.process = subprocess.Popen(
            [self.python_exe, self.sim_script, "--episodes", str(episodes)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=self.script_dir,
            text=True,
            bufsize=1
        )
        
        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_process())
        
        return True
    
    async def stop(self) -> bool:
        """Stops the running simulation."""
        if not self.is_running():
            logger.warning("No simulation running to stop")
            return False
        
        logger.info("🛑 Stopping simulation...")
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        
        self.status["running"] = False
        self.status["last_update"] = datetime.now().isoformat()
        
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        
        return True
    
    def is_running(self) -> bool:
        """Check if simulation is currently running."""
        if self.process and self.process.poll() is None:
            return True
        return False
    
    def get_status(self) -> dict:
        """Get current simulation status."""
        # Try to read from status file for latest data
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    file_status = json.load(f)
                    self.status.update(file_status)
            except:
                pass
        
        # Update running flag based on actual process state
        self.status["running"] = self.is_running()
        self.status["last_update"] = datetime.now().isoformat()
        
        return self.status
    
    async def _monitor_process(self):
        """Monitor subprocess and capture output."""
        if not self.process:
            return
        
        try:
            while self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.log_queue.put({
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "message": line.strip(),
                        "source": "simulation"
                    })
                await asyncio.sleep(0.1)
            
            # Process finished
            self.status["running"] = False
            self.status["last_update"] = datetime.now().isoformat()
            logger.info("Simulation process completed")
            
        except asyncio.CancelledError:
            logger.info("Simulation monitor cancelled")
        except Exception as e:
            logger.error(f"Error monitoring simulation: {e}")
    
    def get_logs(self, max_entries: int = 100) -> list:
        """Get recent log entries."""
        logs = []
        while not self.log_queue.empty() and len(logs) < max_entries:
            logs.append(self.log_queue.get_nowait())
        return logs


class WorkerManager:
    """
    Central manager for all background workers.
    """
    
    def __init__(self):
        self.simulation = SimulationWorker()
        self.start_time = datetime.now()
    
    def get_uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()


# Singleton instance
worker_manager = WorkerManager()
