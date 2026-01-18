import os
import sys
import signal
import time
import logging
import atexit
from pathlib import Path
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("Lifecycle.Manager")

# =============================================================================
# MODELS
# =============================================================================

class ProcessState(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"
    UNKNOWN = "unknown"

@dataclass
class ProcessInfo:
    service_name: str
    pid: int
    state: ProcessState
    pid_file: str
    is_ours: bool = True

# =============================================================================
# UTILS
# =============================================================================

def is_windows() -> bool:
    return sys.platform.startswith('win')

def is_process_alive(pid: int) -> bool:
    if pid <= 0: return False
    try:
        if is_windows():
            try:
                import psutil
                return psutil.pid_exists(pid)
            except ImportError:
                import ctypes
                h = ctypes.windll.kernel32.OpenProcess(0x1000, 0, pid)
                if h:
                    ctypes.windll.kernel32.CloseHandle(h)
                    return True
                return False
        else:
            os.kill(pid, 0)
            return True
    except: return False

def get_pid_path(service: str, pid_dir: str = ".pid") -> Path:
    Path(pid_dir).mkdir(exist_ok=True)
    return Path(pid_dir) / f"{service}.pid"

# =============================================================================
# SIGNALS
# =============================================================================

def terminate_process(pid: int, timeout: int = 5):
    if not is_process_alive(pid): return
    try:
        if is_windows():
            os.kill(pid, signal.CTRL_BREAK_EVENT if hasattr(signal, "CTRL_BREAK_EVENT") else signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
        
        # Wait for graceful exit
        for _ in range(timeout * 10):
            if not is_process_alive(pid): return
            time.sleep(0.1)
            
        # Hard kill if still alive
        if is_process_alive(pid):
            logger.warning(f"Process {pid} unresponsive, forcing SIGKILL")
            if is_windows():
                import subprocess
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGKILL)
    except Exception as e:
        logger.error(f"Failed to terminate {pid}: {e}")

# =============================================================================
# MANAGER
# =============================================================================

class LifecycleManager:
    def __init__(self, pid_dir: str = ".pid"):
        self.pid_dir = pid_dir
        self.managed_services: List[str] = []
        atexit.register(self.cleanup_all)

    def register_service(self, name: str) -> bool:
        pid = os.getpid()
        path = get_pid_path(name, self.pid_dir)
        if path.exists():
            try:
                old_pid = int(path.read_text().strip())
                if is_process_alive(old_pid) and old_pid != pid:
                    logger.error(f"Service {name} already running with PID {old_pid}")
                    return False
            except: pass
        path.write_text(str(pid))
        if name not in self.managed_services: self.managed_services.append(name)
        logger.info(f"Service {name} registered with PID {pid}")
        return True

    def get_service_info(self, name: str) -> ProcessInfo:
        path = get_pid_path(name, self.pid_dir)
        if not path.exists(): return ProcessInfo(name, 0, ProcessState.STOPPED, str(path))
        try:
            pid = int(path.read_text().strip())
            alive = is_process_alive(pid)
            return ProcessInfo(name, pid, ProcessState.RUNNING if alive else ProcessState.STALE, str(path))
        except: return ProcessInfo(name, 0, ProcessState.UNKNOWN, str(path))

    def stop_service(self, name: str):
        info = self.get_service_info(name)
        if info.state == ProcessState.RUNNING:
            terminate_process(info.pid)
        path = Path(info.pid_file)
        if path.exists(): path.unlink()
        if name in self.managed_services: self.managed_services.remove(name)

    def cleanup_all(self):
        for service in list(self.managed_services):
            self.stop_service(service)
