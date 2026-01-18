import logging
import subprocess
import asyncio
import sys
import anyio
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Terminator")

class TerminalTerminator:
    """
    Autonomous MT5 terminal recovery module.
    When soft reconnection fails repeatedly, escalates to process kill and restart.
    """
    def __init__(self, max_soft_failures: int = 3):
        self.max_soft_failures = max_soft_failures
        self.consecutive_failures = 0
        self.hard_resets_count = 0
        logger.info(f"[TERMINATOR] Initialized (hard-reset after {max_soft_failures} failures)")

    def record_failure(self) -> bool:
        self.consecutive_failures += 1
        logger.warning(f"[TERMINATOR] Failure {self.consecutive_failures}/{self.max_soft_failures}")
        return self.consecutive_failures >= self.max_soft_failures

    def record_success(self):
        if self.consecutive_failures > 0:
            logger.info(f"[TERMINATOR] Connection restored after {self.consecutive_failures} failures")
        self.consecutive_failures = 0

    async def execute_hard_reset(self) -> bool:
        if sys.platform != "win32":
            logger.warning("[TERMINATOR] Hard-reset only supported on Windows")
            return False

        logger.critical("[TERMINATOR] 💀 EXECUTING HARD-RESET - Killing MT5 Process")
        self.hard_resets_count += 1

        try:
            result = await anyio.to_thread.run_sync(
                lambda: subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True, text=True)
            )
            if result.returncode == 0:
                logger.info("[TERMINATOR] MT5 process terminated successfully")
            else:
                logger.warning(f"[TERMINATOR] taskkill returned: {result.stderr}")

            await asyncio.sleep(2)
            self.consecutive_failures = 0
            return True
        except Exception as e:
            logger.error(f"[TERMINATOR] Hard-reset failed: {e}")
            return False

    def get_status(self) -> dict:
        return {
            "consecutive_failures": self.consecutive_failures,
            "hard_resets_count": self.hard_resets_count,
            "max_soft_failures": self.max_soft_failures,
        }
