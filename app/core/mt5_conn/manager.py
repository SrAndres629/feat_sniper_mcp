import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar
import anyio

from app.core.config import settings
from .utils import mt5, MT5_AVAILABLE, T, obs_engine, tracer
from .terminator import TerminalTerminator

logger = logging.getLogger("MT5_Bridge.Manager")

class MT5Connection:
    """Singleton connection manager for MetaTrader 5."""
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
        if self._initialized: return
        self._initialized = True
        self._connected = False
        self._terminator = TerminalTerminator(max_soft_failures=3)
        logger.debug("MT5Connection core component initialized.")

    @property
    def connected(self) -> bool:
        return self._connected and MT5_AVAILABLE

    async def startup(self) -> bool:
        if not MT5_AVAILABLE:
            if settings.HEADLESS_MODE: return True
            logger.critical("FATAL: MetaTrader5 library not found.")
            return False
        success = await self.execute(self._initialize_mt5)
        if success:
            self._connected = True
            self.start_watchdog()
        return success

    def start_watchdog(self):
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._connection_watchdog())

    async def _connection_watchdog(self):
        backoff = 10
        while True:
            await asyncio.sleep(backoff)
            try:
                if not await self._check_health_async():
                    self._connected = False
                    if await self._try_reconnect_async():
                        self._terminator.record_success()
                        self._connected = True
                        backoff = 10
                    else:
                        if self._terminator.record_failure():
                            await self._terminator.execute_hard_reset()
                            await asyncio.sleep(5)
                            if await self._try_reconnect_async(): self._connected = True
                        backoff = min(60, backoff * 2)
                else:
                    self._terminator.record_success()
                    backoff = 10
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                backoff = min(60, backoff * 2)

    async def _check_health_async(self) -> bool:
        if not MT5_AVAILABLE: return False
        try:
            return await anyio.to_thread.run_sync(self._health_check_sync)
        except: return False

    def _health_check_sync(self) -> bool:
        return mt5.terminal_info() is not None if MT5_AVAILABLE else False

    async def _try_reconnect_async(self) -> bool:
        if not MT5_AVAILABLE: return False
        try:
            return await anyio.to_thread.run_sync(self._reconnect_sync)
        except Exception: return False

    def _reconnect_sync(self) -> bool:
        with self._mt5_lock: return self._initialize_mt5()

    def _initialize_mt5(self) -> bool:
        if not MT5_AVAILABLE: return False
        params = {"login": int(settings.MT5_LOGIN), "password": settings.MT5_PASSWORD, "server": settings.MT5_SERVER}
        if settings.MT5_PATH: params["path"] = settings.MT5_PATH
        params = {k: v for k, v in params.items() if v is not None}
        
        try: mt5.shutdown()
        except: pass

        if not mt5.initialize(**params):
            logger.error(f"MT5 Init Failure: {mt5.last_error()}")
            return False

        account = mt5.account_info()
        if account: logger.info(f"[STATUS] MT5 OK (Login: {account.login}) | ${account.balance:.2f}")
        return True

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        if not MT5_AVAILABLE: return None
        op_name = getattr(func, "__name__", "mt5_call")
        
        async def _run():
            start = time.time()
            res = await anyio.to_thread.run_sync(self._locked_execution, func, *args, **kwargs)
            if obs_engine: obs_engine.track_latency(op_name, "GLOBAL", time.time() - start)
            return res

        if tracer:
            with tracer.start_as_current_span(f"mt5_{op_name}"): return await _run()
        return await _run()

    def _locked_execution(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self._mt5_lock:
            if not mt5.terminal_info() and func not in [self._initialize_mt5, mt5.initialize]:
                self._connected = False
                return None
            return func(*args, **kwargs)

    async def execute_atomic(self, func: Callable[..., T], *args, **kwargs) -> T:
        if not MT5_AVAILABLE: return None
        return await anyio.to_thread.run_sync(self._atomic_execution, func, *args, **kwargs)

    def _atomic_execution(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self._mt5_lock:
            if not mt5.terminal_info(): return None
            return func(*args, **kwargs)

    async def get_account_info(self) -> dict:
        if not self.connected: return {"status": "offline"}
        info = await self.execute(mt5.account_info)
        return {"status": "online", "balance": info.balance, "equity": info.equity} if info else {"status": "error"}
