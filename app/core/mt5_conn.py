"""
MT5 Connection Manager - Deadlock-Free Architecture (TerminalTerminator)
=========================================================================
Singleton connection manager for MetaTrader 5 with thread-safe,
non-blocking execution and deadlock-free reconnection logic.

[PROJECT ATLAS] Vibranium Grade Upgrades:
- TerminalTerminator: Autonomous hard-reset (taskkill) on persistent failure
- Separated health check from reconnection logic
- Pre-check terminal state BEFORE acquiring lock
- Exponential backoff with hard-reset escalation
"""

import asyncio
import logging
import threading
import subprocess
import sys
from typing import Any, Callable, TypeVar, Optional

import anyio

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

from app.core.config import settings

try:
    from app.core.observability import obs_engine, tracer, resilient
except ImportError:
    obs_engine = None
    tracer = None

    def resilient(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger("MT5_Bridge.Core")

T = TypeVar("T")


# =============================================================================
# TERMINAL TERMINATOR - Autonomous Hard-Reset
# =============================================================================


class TerminalTerminator:
    """
    Autonomous MT5 terminal recovery module.

    When soft reconnection fails repeatedly, this class escalates to:
    1. Kill MT5 process (taskkill /F /IM terminal64.exe)
    2. Wait for process death
    3. Restart terminal via path
    4. Wait for initialization

    This is the nuclear option - used only after 3 consecutive failures.
    """

    def __init__(self, max_soft_failures: int = 3):
        self.max_soft_failures = max_soft_failures
        self.consecutive_failures = 0
        self.hard_resets_count = 0
        logger.info(
            f"[TERMINATOR] Initialized (hard-reset after {max_soft_failures} failures)"
        )

    def record_failure(self) -> bool:
        """
        Record a connection failure.

        Returns:
            True if hard-reset is needed (threshold exceeded)
        """
        self.consecutive_failures += 1
        logger.warning(
            f"[TERMINATOR] Failure {self.consecutive_failures}/{self.max_soft_failures}"
        )
        return self.consecutive_failures >= self.max_soft_failures

    def record_success(self):
        """Record successful connection - reset failure counter."""
        if self.consecutive_failures > 0:
            logger.info(
                f"[TERMINATOR] Connection restored after {self.consecutive_failures} failures"
            )
        self.consecutive_failures = 0

    async def execute_hard_reset(self) -> bool:
        """
        Execute hard-reset sequence:
        1. Kill MT5 process
        2. Wait for death
        3. Optionally restart

        Returns:
            True if process was killed successfully
        """
        if sys.platform != "win32":
            logger.warning("[TERMINATOR] Hard-reset only supported on Windows")
            return False

        logger.critical("[TERMINATOR] 💀 EXECUTING HARD-RESET - Killing MT5 Process")
        self.hard_resets_count += 1

        try:
            # Kill MT5 terminal process
            result = await anyio.to_thread.run_sync(
                lambda: subprocess.run(
                    ["taskkill", "/F", "/IM", "terminal64.exe"],
                    capture_output=True,
                    text=True,
                )
            )

            if result.returncode == 0:
                logger.info("[TERMINATOR] MT5 process terminated successfully")
            else:
                logger.warning(f"[TERMINATOR] taskkill returned: {result.stderr}")

            # Wait for process to fully die
            await asyncio.sleep(2)

            # Reset failure counter for fresh start
            self.consecutive_failures = 0

            return True

        except Exception as e:
            logger.error(f"[TERMINATOR] Hard-reset failed: {e}")
            return False

    def get_status(self) -> dict:
        """Return terminator status for diagnostics."""
        return {
            "consecutive_failures": self.consecutive_failures,
            "hard_resets_count": self.hard_resets_count,
            "max_soft_failures": self.max_soft_failures,
        }


T = TypeVar("T")


class MT5Connection:
    """Singleton connection manager for MetaTrader 5.

    Provides a thread-safe (threading.Lock) and non-blocking (anyio.to_thread)
    environment for interacting with the synchronous MT5 C-API.

    [VIBRANIUM] Integrated with TerminalTerminator for autonomous recovery.
    """

    _instance: Optional["MT5Connection"] = None
    _lock = threading.Lock()
    _mt5_lock = threading.Lock()
    _watchdog_task: Optional[asyncio.Task] = None

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MT5Connection, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._connected = False
        self._needs_reconnect = False
        self._terminator = TerminalTerminator(max_soft_failures=3)
        logger.debug("MT5Connection core component initialized.")

    @property
    def connected(self) -> bool:
        """Thread-safe connection status check."""
        return self._connected and MT5_AVAILABLE

    async def startup(self) -> bool:
        """Initializes MT5 terminal connectivity and starts the watchdog.

        Enters passive mode in Linux/Docker environments.

        Returns:
            bool: True if initialization was successful or skipped (Docker).
        """
        if not MT5_AVAILABLE:
            if settings.HEADLESS_MODE:
                logger.warning(
                    "MT5 not found but HEADLESS_MODE is ON. Starting in PASSIVE MODE."
                )
                return True

            logger.critical(
                "FATAL: MetaTrader5 library not found and HEADLESS_MODE is False."
            )
            logger.critical(
                "Please install MetaTrader5 or enable HEADLESS_MODE in config."
            )
            return False

        success = await self.execute(self._initialize_mt5)
        if success:
            self._connected = True
            self.start_watchdog()
        return success

    def start_watchdog(self):
        """Inicia la tarea de monitoreo en segundo plano."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._connection_watchdog())
            logger.info("Watchdog de conexión MT5 iniciado.")

    async def _connection_watchdog(self):
        """
        [VIBRANIUM] Periodic task that verifies and restores connection.

        Escalation Path:
        1. Soft reconnect (normal MT5 init)
        2. If fails 3x -> TerminalTerminator hard-reset
        3. After hard-reset, retry soft connect
        """
        backoff = 10
        max_backoff = 60

        while True:
            await asyncio.sleep(backoff)
            try:
                is_connected = await self._check_health_async()

                if not is_connected:
                    logger.warning(
                        f"Watchdog: MT5 connection lost. Attempting recovery..."
                    )
                    self._connected = False

                    # Attempt soft reconnect
                    success = await self._try_reconnect_async()

                    if success:
                        self._terminator.record_success()
                        logger.info("Watchdog: Connection restored.")
                        self._connected = True
                        backoff = 10
                    else:
                        # Check if we need hard-reset
                        need_hard_reset = self._terminator.record_failure()

                        if need_hard_reset:
                            logger.critical("Watchdog: Escalating to HARD-RESET")
                            await self._terminator.execute_hard_reset()
                            # After hard-reset, try to reconnect
                            await asyncio.sleep(5)  # Wait for MT5 to stabilize
                            success = await self._try_reconnect_async()
                            if success:
                                self._connected = True
                                backoff = 10
                        else:
                            backoff = min(max_backoff, backoff * 2)
                else:
                    self._terminator.record_success()
                    backoff = 10
                    logger.debug("Watchdog: Connection OK.")

            except Exception as e:
                logger.error(f"Critical error in MT5 watchdog: {e}")
                backoff = min(max_backoff, backoff * 2)

    async def _check_health_async(self) -> bool:
        """
        [P0 FIX] Health check that does NOT trigger reconnection.

        Simply checks if terminal_info() returns valid data.
        """
        if not MT5_AVAILABLE:
            return False

        try:
            result = await anyio.to_thread.run_sync(self._health_check_sync)
            return result
        except Exception:
            return False

    def _health_check_sync(self) -> bool:
        """
        [P0 FIX] Synchronous health check - NO LOCK NEEDED.

        mt5.terminal_info() is thread-safe for read operations.
        We deliberately don't acquire the lock here to avoid contention.
        """
        if not MT5_AVAILABLE:
            return False
        try:
            info = mt5.terminal_info()
            return info is not None
        except Exception:
            return False

    async def _try_reconnect_async(self) -> bool:
        """
        [P0 FIX] Safe reconnection - runs the full init sequence.

        This method acquires the lock for the ENTIRE reconnection process,
        but is called from OUTSIDE any locked context, preventing deadlock.
        """
        if not MT5_AVAILABLE:
            return False

        try:
            result = await anyio.to_thread.run_sync(self._reconnect_sync)
            return result
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    def _reconnect_sync(self) -> bool:
        """
        [P0 FIX] Synchronous reconnection WITH lock protection.

        This is the ONLY place where _initialize_mt5 is called with lock held.
        It's never called from within _locked_execution.
        """
        with self._mt5_lock:
            return self._initialize_mt5()

    def _initialize_mt5(self) -> bool:
        """Internal initialization logic (Blocking).

        [P0 FIX] This method is now ONLY called from:
        1. startup() via execute() - initial connection
        2. _reconnect_sync() - recovery from watchdog

        It is NEVER called recursively from _locked_execution.

        Returns:
            bool: True if connection established.
        """
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 is not installed in this environment.")
            return False

        logger.info(f"Connecting to MT5 (Server: {settings.MT5_SERVER})...")

        init_params = {
            "login": settings.MT5_LOGIN,
            "password": settings.MT5_PASSWORD,
            "server": settings.MT5_SERVER,
        }

        if settings.MT5_PATH:
            init_params["path"] = settings.MT5_PATH

        # [FIX] Filter out None values to avoid "Invalid argument" error
        init_params = {k: v for k, v in init_params.items() if v is not None}

        # Ensure login is int if present
        if "login" in init_params:
            try:
                init_params["login"] = int(init_params["login"])
            except ValueError:
                logger.error(f"Invalid login format: {init_params['login']}")
                return False

        # [P0 FIX] Wrap shutdown in try-except for cold start safety
        try:
            mt5.shutdown()  # Force clean state
        except Exception:
            pass  # Ignore errors on cold start

        if not mt5.initialize(**init_params):
            err_code, err_msg = mt5.last_error()
            logger.error(f"MT5 Init Critical Failure: {err_msg} (Code: {err_code})")
            return False

        # [PRE-FLIGHT CHECK] Account & Data Verification
        try:
            account_info = mt5.account_info()
            if account_info:
                acc_type = (
                    "REAL"
                    if account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL
                    else "DEMO"
                )
                logger.info("")
                logger.info("[STATUS] CONEXION MT5: OK")
                logger.info(f"[ACCOUNT] TYPE: {acc_type} (Login: {account_info.login})")
                logger.info(f"[BALANCE] CURRENT: ${account_info.balance:.2f}")
                logger.info(f"[LEVERAGE] 1:{account_info.leverage}")

            # History Sync Check (500 bars)
            ticks = mt5.copy_rates_from_pos(settings.SYMBOL, mt5.TIMEFRAME_M1, 0, 500)
            if ticks is None or len(ticks) < 500:
                logger.warning(
                    f"[HISTORY] WARNING: ONLY FOUND {len(ticks) if ticks else 0} BARs"
                )
            else:
                logger.info(f"[HISTORY] TICKS LOADED: {len(ticks)}")

        except Exception as e:
            logger.error(f"Pre-Flight Check Failed: {e}")

        logger.info("[OK] MetaTrader 5 Connection Established.")
        return True

    async def shutdown(self):
        """Cierra la conexión con MT5 y detiene el watchdog."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            logger.info("Watchdog de conexión MT5 detenido.")
        logger.info("Cerrando conexión con MT5...")
        if MT5_AVAILABLE:
            await self.execute(mt5.shutdown)
        self._connected = False
        logger.info("[OK] MT5 desconectado.")

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Punto de entrada maestro para ejecutar cualquier función de mt5.
        Instrumentado con OTel y Métricas de Latencia.
        En Docker/Linux retorna None gracefully.
        """
        # Bypass completo si MT5 no está disponible
        if not MT5_AVAILABLE:
            op_name = func.__name__ if hasattr(func, "__name__") else "mt5_call"
            logger.debug(
                f"[Docker Mode] Operación '{op_name}' ignorada - MT5 no disponible."
            )
            return None

        import time

        start_time = time.time()
        op_name = func.__name__ if hasattr(func, "__name__") else "mt5_call"

        # Use tracer if available
        if tracer:
            with tracer.start_as_current_span(f"mt5_{op_name}") as span:
                if "symbol" in kwargs:
                    span.set_attribute("symbol", kwargs["symbol"])
                elif args and isinstance(args[0], str) and len(args[0]) < 10:
                    span.set_attribute("symbol", args[0])

                try:
                    result = await anyio.to_thread.run_sync(
                        self._locked_execution, func, *args, **kwargs
                    )

                    duration = time.time() - start_time
                    if obs_engine:
                        obs_engine.track_latency(op_name, "GLOBAL", duration)

                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise e
        else:
            # No tracer available
            try:
                result = await anyio.to_thread.run_sync(
                    self._locked_execution, func, *args, **kwargs
                )
                return result
            except Exception as e:
                logger.error(f"MT5 execution error: {e}")
                raise e

    def _locked_execution(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        [P0 FIX] Ejecuta la función dentro del lock de MT5.

        CRITICAL CHANGE: Removed recursive _initialize_mt5 call.
        If terminal is not active, we set a flag for async reconnection
        instead of attempting reconnection within the lock.
        """
        if not MT5_AVAILABLE:
            logger.error("[Failsafe] _locked_execution called without MT5.")
            return None

        with self._mt5_lock:
            # [P0 FIX] Pre-check without reconnection - just log and return None
            # The watchdog will handle reconnection asynchronously
            if (
                not mt5.terminal_info()
                and func != self._initialize_mt5
                and func != mt5.initialize
            ):
                logger.warning(
                    f"MT5 terminal not active during {func.__name__ if hasattr(func, '__name__') else 'call'}. "
                    "Watchdog will attempt reconnection."
                )
                self._connected = False
                return None

            return func(*args, **kwargs)

    # =========================================================================
    # ATOMIC EXECUTION - Race Condition Fix
    # =========================================================================

    async def execute_atomic(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Executes a compound function atomically within the MT5 lock.

        Unlike execute(), which acquires/releases the lock per-call, this method
        holds the lock for the ENTIRE duration of func, allowing func to make
        multiple MT5 calls without race conditions.

        Use case: Fetch tick + send order in a single atomic block.

        Args:
            func: A function that internally calls MT5 APIs (e.g., symbol_info_tick, order_send)
            *args, **kwargs: Arguments passed to func

        Returns:
            The return value of func
        """
        if not MT5_AVAILABLE:
            op_name = getattr(func, "__name__", "atomic_fn")
            logger.debug(
                f"[Docker Mode] Atomic op '{op_name}' ignored - MT5 unavailable."
            )
            return None

        import time

        start_time = time.time()
        op_name = f"atomic_{getattr(func, '__name__', 'fn')}"

        if tracer:
            with tracer.start_as_current_span(f"mt5_{op_name}") as span:
                try:
                    result = await anyio.to_thread.run_sync(
                        self._atomic_execution, func, *args, **kwargs
                    )

                    duration = time.time() - start_time
                    if obs_engine:
                        obs_engine.track_latency(op_name, "GLOBAL", duration)
                    span.set_attribute("atomic", True)

                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise e
        else:
            try:
                result = await anyio.to_thread.run_sync(
                    self._atomic_execution, func, *args, **kwargs
                )
                return result
            except Exception as e:
                logger.error(f"Atomic execution error: {e}")
                raise e

    def _atomic_execution(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        [P0 FIX] Executes func while holding the MT5 lock for its entire duration.

        CRITICAL CHANGE: Like _locked_execution, NO recursive reconnection.
        If terminal not active, return None and let watchdog handle it.
        """
        if not MT5_AVAILABLE:
            logger.error("[Failsafe] _atomic_execution called without MT5.")
            return None

        with self._mt5_lock:
            # [P0 FIX] Pre-check without reconnection attempt
            if not mt5.terminal_info():
                logger.warning("Atomic exec: Terminal not active. Returning None.")
                self._connected = False
                return None

            return func(*args, **kwargs)

    async def get_account_info(self) -> dict:
        """Get account information from MT5."""
        if not MT5_AVAILABLE or not self._connected:
            return {"status": "offline"}

        try:
            info = await self.execute(mt5.account_info)
            if info:
                return {
                    "status": "online",
                    "balance": info.balance,
                    "equity": info.equity,
                    "margin": info.margin,
                    "free_margin": info.margin_free,
                    "profit": info.profit,
                    "leverage": info.leverage,
                }
            return {"status": "error", "message": "No account info returned"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Instancia global exportable
mt5_conn = MT5Connection()
