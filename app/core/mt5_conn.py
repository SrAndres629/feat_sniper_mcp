import asyncio
import logging
import threading
from typing import Any, Callable, TypeVar, Optional

import anyio
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    
from app.core.config import settings

from app.core.observability import obs_engine, tracer, resilient

logger = logging.getLogger("MT5_Bridge.Core")

T = TypeVar("T")

class MT5Connection:
    """Singleton connection manager for MetaTrader 5.
    
    Provides a thread-safe (threading.Lock) and non-blocking (anyio.to_thread)
    environment for interacting with the synchronous MT5 C-API.
    
    Attributes:
        _instance: Singleton instance.
        _lock: Singleton lifecycle lock.
        _mt5_lock: API access synchronization lock.
        _watchdog_task: Background connectivity monitor.
    """
    _instance: Optional['MT5Connection'] = None
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
        logger.debug("MT5Connection core component initialized.")

    async def startup(self) -> bool:
        """Initializes MT5 terminal connectivity and starts the watchdog.
        
        Enters passive mode in Linux/Docker environments.
        
        Returns:
            bool: True if initialization was successful or skipped (Docker).
        """
        if not MT5_AVAILABLE:
            logger.info("Linux/Docker environment detected. Skipping local MT5 init. Entering Passive Mode.")
            return True
            
        success = await self.execute(self._initialize_mt5)
        if success:
            self.start_watchdog()
        return success

    def start_watchdog(self):
        """Inicia la tarea de monitoreo en segundo plano."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._connection_watchdog())
            logger.info("Watchdog de conexin MT5 iniciado.")

    async def _connection_watchdog(self):
        """Tarea peridica que verifica y restaura la conexin con Backoff Exponencial."""
        backoff = 10
        max_backoff = 60
        
        while True:
            await asyncio.sleep(backoff) 
            try:
                is_connected = await self.execute(lambda: mt5.terminal_info() is not None)
                if not is_connected:
                    logger.warning(f"Watchdog: Conexin MT5 perdida. Reintentando en {backoff}s...")
                    success = await self.execute(self._initialize_mt5)
                    
                    if success:
                        logger.info("Watchdog: Conexin restaurada.")
                        backoff = 10 # Reset
                    else:
                        backoff = min(max_backoff, backoff * 2) # Incrementar espera
                else:
                    backoff = 10 # Reset si estamos bien
                    logger.debug("Watchdog: Conexin OK.")
                    
            except Exception as e:
                logger.error(f"Error crtico en watchdog de MT5: {e}")
                backoff = min(max_backoff, backoff * 2)

    @resilient(max_retries=2, failure_threshold=2, recovery_timeout=10)
    def _initialize_mt5(self) -> bool:
        """Internal initialization logic (Blocking).
        
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
            "server": settings.MT5_SERVER
        }
        
        if settings.MT5_PATH:
            init_params["path"] = settings.MT5_PATH

        # [FIX] Filter out None values to avoid "Invalid argument" error
        # If login is None, we don't send it, letting MT5 use stored profile
        init_params = {k: v for k, v in init_params.items() if v is not None}
        
        # Ensure login is int if present
        if "login" in init_params:
            try:
                init_params["login"] = int(init_params["login"])
            except ValueError:
                logger.error(f"Invalid login format: {init_params['login']}")
                return False

        mt5.shutdown() # Force clean state

        if not mt5.initialize(**init_params):
            err_code, err_msg = mt5.last_error()
            logger.error(f"MT5 Init Critical Failure: {err_msg} (Code: {err_code})")
            return False
            
        logger.info(" MetaTrader 5 Connection Established.")
        return True

    async def shutdown(self):
        """Cierra la conexin con MT5 y detiene el watchdog."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            logger.info("Watchdog de conexin MT5 detenido.")
        logger.info("Cerrando conexin con MT5...")
        if MT5_AVAILABLE:
            await self.execute(mt5.shutdown)
        logger.info(" MT5 desconectado.")

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Punto de entrada maestro para ejecutar cualquier funcin de mt5.
        Instrumentado con OTel y Mtricas de Latencia.
        En Docker/Linux retorna None gracefully.
        """
        # Bypass completo si MT5 no est disponible
        if not MT5_AVAILABLE:
            op_name = func.__name__ if hasattr(func, "__name__") else "mt5_call"
            logger.debug(f"[Docker Mode] Operacin '{op_name}' ignorada - MT5 no disponible.")
            return None
        
        from app.core.observability import obs_engine, tracer
        import time

        start_time = time.time()
        op_name = func.__name__ if hasattr(func, "__name__") else "mt5_call"
        
        with tracer.start_as_current_span(f"mt5_{op_name}") as span:
            if "symbol" in kwargs:
                span.set_attribute("symbol", kwargs["symbol"])
            elif args and isinstance(args[0], str) and len(args[0]) < 10:
                span.set_attribute("symbol", args[0])

            try:
                result = await anyio.to_thread.run_sync(self._locked_execution, func, *args, **kwargs)
                
                duration = time.time() - start_time
                obs_engine.track_latency(op_name, "GLOBAL", duration)
                
                return result
            except Exception as e:
                span.record_exception(e)
                raise e

    def _locked_execution(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Envuelve la ejecucin de la funcin dentro del lock de MT5."""
        # Este punto NUNCA debera alcanzarse si MT5 no est disponible
        # debido al bypass en execute(), pero lo dejamos como failsafe.
        if not MT5_AVAILABLE:
            logger.error("[Failsafe] _locked_execution llamado sin MT5. Esto no debera pasar.")
            return None
            
        with self._mt5_lock:
            # Una verificacin extra de seguridad
            if not mt5.terminal_info() and func != self._initialize_mt5 and func != mt5.initialize:
                logger.warning("Llamada detectada sin terminal activo, intentando re-init...")
                self._initialize_mt5()
            
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
            op_name = getattr(func, '__name__', 'atomic_fn')
            logger.debug(f"[Docker Mode] Atomic op '{op_name}' ignored - MT5 unavailable.")
            return None
        
        from app.core.observability import obs_engine, tracer
        import time

        start_time = time.time()
        op_name = f"atomic_{getattr(func, '__name__', 'fn')}"
        
        with tracer.start_as_current_span(f"mt5_{op_name}") as span:
            try:
                result = await anyio.to_thread.run_sync(
                    self._atomic_execution, func, *args, **kwargs
                )
                
                duration = time.time() - start_time
                obs_engine.track_latency(op_name, "GLOBAL", duration)
                span.set_attribute("atomic", True)
                
                return result
            except Exception as e:
                span.record_exception(e)
                raise e
    
    def _atomic_execution(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Executes func while holding the MT5 lock for its entire duration.
        
        The function 'func' can make multiple MT5 API calls internally,
        all protected by a single lock acquisition.
        """
        if not MT5_AVAILABLE:
            logger.error("[Failsafe] _atomic_execution called without MT5.")
            return None
            
        with self._mt5_lock:
            # Pre-check terminal state
            if not mt5.terminal_info():
                logger.warning("Atomic exec: Terminal not active, attempting re-init...")
                self._initialize_mt5()
            
            return func(*args, **kwargs)

# Instancia global exportable
mt5_conn = MT5Connection()
