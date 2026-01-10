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

logger = logging.getLogger("MT5_Bridge.Core")

T = TypeVar("T")

class MT5Connection:
    """
    Gestor de conexión Singleton para MetaTrader 5.
    
    Proporciona un entorno seguro para hilos (threading.Lock) y no bloqueante
    (anyio.to_thread) para interactuar con la librería síncrona de MT5.
    """
    _instance = None
    _lock = threading.Lock()
    _mt5_lock = threading.Lock()  # El candado real para las llamadas a la API de MT5
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
        logger.info("MT5Connection Singleton inicializado.")

    async def startup(self) -> bool:
        """
        Inicializa la conexión con el terminal MT5 e inicia el watchdog.
        En Linux/Docker, entra en modo pasivo sin lanzar errores.
        """
        if not MT5_AVAILABLE:
            logger.info("Entorno Linux/Docker detectado. Saltando inicialización local de MT5. Esperando conexión ZMQ...")
            return True
            
        success = await self.execute(self._initialize_mt5)
        if success:
            self.start_watchdog()
        return success

    def start_watchdog(self):
        """Inicia la tarea de monitoreo en segundo plano."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._connection_watchdog())
            logger.info("Watchdog de conexión MT5 iniciado.")

    async def _connection_watchdog(self):
        """Tarea periódica que verifica y restaura la conexión con Backoff Exponencial."""
        backoff = 10
        max_backoff = 60
        
        while True:
            await asyncio.sleep(backoff) 
            try:
                is_connected = await self.execute(lambda: mt5.terminal_info() is not None)
                if not is_connected:
                    logger.warning(f"Watchdog: Conexión MT5 perdida. Reintentando en {backoff}s...")
                    success = await self.execute(self._initialize_mt5)
                    
                    if success:
                        logger.info("Watchdog: Conexión restaurada.")
                        backoff = 10 # Reset
                    else:
                        backoff = min(max_backoff, backoff * 2) # Incrementar espera
                else:
                    backoff = 10 # Reset si estamos bien
                    logger.debug("Watchdog: Conexión OK.")
                    
            except Exception as e:
                logger.error(f"Error crítico en watchdog de MT5: {e}")
                backoff = min(max_backoff, backoff * 2)

    def _initialize_mt5(self) -> bool:
        """Lógica interna de inicialización (bloqueante)."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 no está instalado en este entorno. Abortando inicialización.")
            return False
            
        logger.info(f"Intentando conectar a MT5 (Server: {settings.MT5_SERVER})...")
        
        init_params = {
            "login": settings.MT5_LOGIN,
            "password": settings.MT5_PASSWORD,
            "server": settings.MT5_SERVER
        }
        
        if settings.MT5_PATH:
            init_params["path"] = settings.MT5_PATH

        # Forzar cierre previo por si acaso
        mt5.shutdown()

        if not mt5.initialize(**init_params):
            error = mt5.last_error()
            logger.error(f"Error crítico al inicializar MT5: {error[1]} (Código: {error[0]})")
            return False
            
        logger.info("✅ Conexión con MT5 establecida exitosamente.")
        return True

    async def shutdown(self):
        """Cierra la conexión con MT5 y detiene el watchdog."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            logger.info("Watchdog de conexión MT5 detenido.")
        logger.info("Cerrando conexión con MT5...")
        if MT5_AVAILABLE:
            await self.execute(mt5.shutdown)
        logger.info("🛑 MT5 desconectado.")

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Punto de entrada maestro para ejecutar cualquier función de mt5.
        Instrumentado con OTel y Métricas de Latencia.
        En Docker/Linux retorna None gracefully.
        """
        # Bypass completo si MT5 no está disponible
        if not MT5_AVAILABLE:
            op_name = func.__name__ if hasattr(func, "__name__") else "mt5_call"
            logger.debug(f"[Docker Mode] Operación '{op_name}' ignorada - MT5 no disponible.")
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
        """Envuelve la ejecución de la función dentro del lock de MT5."""
        # Este punto NUNCA debería alcanzarse si MT5 no está disponible
        # debido al bypass en execute(), pero lo dejamos como failsafe.
        if not MT5_AVAILABLE:
            logger.error("[Failsafe] _locked_execution llamado sin MT5. Esto no debería pasar.")
            return None
            
        with self._mt5_lock:
            # Una verificación extra de seguridad
            if not mt5.terminal_info() and func != self._initialize_mt5 and func != mt5.initialize:
                logger.warning("Llamada detectada sin terminal activo, intentando re-init...")
                self._initialize_mt5()
            
            return func(*args, **kwargs)

# Instancia global exportable
mt5_conn = MT5Connection()
