"""
LIFECYCLE MANAGER - Surgical Process Management
================================================
Gestión de procesos profesional basada en archivos PID.

[SENIOR DEVOPS / SRE] Este módulo reemplaza los comandos genéricos de "kill all"
por una lógica de gestión de estado profesional, esencial para entornos HFT.

Features:
- Gestión de archivos PID por servicio
- Cleanup quirúrgico (SIGTERM gracioso → SIGKILL forzado)
- Prevención de instancias duplicadas
- Compatibilidad cross-platform (Windows/Unix)
- Context managers para ciclo de vida automático
"""

import os
import sys
import signal
import logging
import time
import atexit
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger("feat.core.lifecycle_manager")


# =============================================================================
# CONFIGURATION
# =============================================================================

PID_DIRECTORY = ".pid"
GRACEFUL_SHUTDOWN_TIMEOUT = 5  # Segundos antes de SIGKILL


# =============================================================================
# PROCESS STATE
# =============================================================================

class ProcessState(Enum):
    """Estados posibles de un proceso gestionado."""
    RUNNING = "running"
    STOPPED = "stopped"
    STALE = "stale"  # PID file existe pero proceso no está vivo
    UNKNOWN = "unknown"


@dataclass
class ProcessInfo:
    """Información de un proceso gestionado."""
    service_name: str
    pid: int
    state: ProcessState
    pid_file: str
    is_ours: bool = True  # True si el PID pertenece a nuestra aplicación


# =============================================================================
# CROSS-PLATFORM UTILITIES
# =============================================================================

def is_windows() -> bool:
    """Detecta si estamos en Windows."""
    return sys.platform.startswith('win')


def is_process_alive(pid: int) -> bool:
    """
    Verifica si un proceso con el PID dado está vivo.
    Compatible con Windows y Unix.
    """
    if pid <= 0:
        return False
    
    try:
        if is_windows():
            # Windows: usar psutil si disponible, o kernel32
            try:
                import psutil
                return psutil.pid_exists(pid)
            except ImportError:
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
                return False
        else:
            # Unix: signal 0 no mata pero verifica existencia
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False
    except Exception as e:
        logger.debug(f"Error checking process {pid}: {e}")
        return False


def kill_process(pid: int, force: bool = False) -> bool:
    """
    Termina un proceso de forma cross-platform.
    
    Args:
        pid: Process ID a terminar
        force: Si True, usa SIGKILL inmediatamente
    
    Returns:
        True si el proceso fue terminado
    """
    if not is_process_alive(pid):
        return True
    
    try:
        if is_windows():
            # Windows: usar taskkill
            import subprocess
            if force:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], 
                             capture_output=True, timeout=5)
            else:
                subprocess.run(["taskkill", "/PID", str(pid)], 
                             capture_output=True, timeout=5)
        else:
            # Unix: SIGTERM o SIGKILL
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, sig)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to kill process {pid}: {e}")
        return False


def get_process_name(pid: int) -> Optional[str]:
    """
    Obtiene el nombre del proceso para validación.
    Usa psutil si está disponible.
    """
    try:
        import psutil
        proc = psutil.Process(pid)
        return proc.name()
    except ImportError:
        return None
    except Exception:
        return None


# =============================================================================
# LIFECYCLE MANAGER - Singleton
# =============================================================================

class LifecycleManager:
    """
    Gestor de Ciclo de Vida de Procesos.
    
    ¿Por qué PID files en lugar de pkill?
    --------------------------------------
    1. pkill python mata TODO, incluyendo tu IDE y otros bots
    2. PID files permiten gestión quirúrgica de cada servicio
    3. Permiten detectar instancias duplicadas antes de arrancar
    4. Habilitación de microservicios: reiniciar un servicio sin afectar otros
    """
    
    _instance: Optional['LifecycleManager'] = None
    
    def __new__(cls) -> 'LifecycleManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._pid_dir = Path(PID_DIRECTORY)
        self._registered_services: Dict[str, int] = {}
        self._shutdown_handlers: Dict[str, Callable] = {}
        
        # Crear directorio PID si no existe
        self._ensure_pid_directory()
        
        logger.info(f"[LIFECYCLE] Manager initialized. PID directory: {self._pid_dir.absolute()}")
    
    def _ensure_pid_directory(self) -> None:
        """Crea el directorio de PIDs si no existe."""
        try:
            self._pid_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"[LIFECYCLE] Failed to create PID directory: {e}")
    
    def _get_pid_file(self, service_name: str) -> Path:
        """Retorna la ruta del archivo PID para un servicio."""
        return self._pid_dir / f"{service_name}.pid"
    
    def register_process(self, service_name: str, shutdown_handler: Optional[Callable] = None) -> bool:
        """
        Registra el proceso actual como un servicio.
        
        Args:
            service_name: Nombre único del servicio (ej: "mcp_server", "inference_api")
            shutdown_handler: Función optional a llamar antes del shutdown
        
        Returns:
            True si el registro fue exitoso
        
        Raises:
            RuntimeError si ya existe una instancia corriendo de este servicio
        """
        pid_file = self._get_pid_file(service_name)
        current_pid = os.getpid()
        
        # 1. Verificar si ya hay una instancia corriendo
        existing = self.get_process_info(service_name)
        if existing and existing.state == ProcessState.RUNNING:
            raise RuntimeError(
                f"[LIFECYCLE] Service '{service_name}' already running with PID {existing.pid}. "
                f"Use cleanup_process() first or ensure only one instance runs."
            )
        
        # 2. Limpiar PID file stale si existe
        if existing and existing.state == ProcessState.STALE:
            logger.warning(f"[LIFECYCLE] Cleaning stale PID file for '{service_name}'")
            self._remove_pid_file(service_name)
        
        # 3. Escribir nuevo PID file
        try:
            with open(pid_file, 'w') as f:
                f.write(str(current_pid))
            
            self._registered_services[service_name] = current_pid
            
            if shutdown_handler:
                self._shutdown_handlers[service_name] = shutdown_handler
            
            # Registrar cleanup automático al exit
            atexit.register(self._auto_cleanup, service_name)
            
            logger.info(f"[LIFECYCLE] ✅ Registered '{service_name}' [PID: {current_pid}]")
            return True
            
        except PermissionError as e:
            logger.error(f"[LIFECYCLE] Permission denied writing PID file: {e}")
            return False
        except Exception as e:
            logger.error(f"[LIFECYCLE] Failed to register '{service_name}': {e}")
            return False
    
    def _auto_cleanup(self, service_name: str) -> None:
        """Cleanup automático al exit del proceso."""
        self._remove_pid_file(service_name)
    
    def _remove_pid_file(self, service_name: str) -> bool:
        """Elimina el archivo PID de un servicio."""
        pid_file = self._get_pid_file(service_name)
        try:
            if pid_file.exists():
                pid_file.unlink()
            return True
        except Exception as e:
            logger.error(f"[LIFECYCLE] Failed to remove PID file: {e}")
            return False
    
    def get_process_info(self, service_name: str) -> Optional[ProcessInfo]:
        """
        Obtiene información de un servicio registrado.
        
        Returns:
            ProcessInfo o None si no hay PID file
        """
        pid_file = self._get_pid_file(service_name)
        
        if not pid_file.exists():
            return None
        
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Verificar si el proceso está vivo
            if is_process_alive(pid):
                # Verificar si es nuestra aplicación (si psutil disponible)
                proc_name = get_process_name(pid)
                is_python = proc_name and 'python' in proc_name.lower() if proc_name else True
                
                return ProcessInfo(
                    service_name=service_name,
                    pid=pid,
                    state=ProcessState.RUNNING,
                    pid_file=str(pid_file),
                    is_ours=is_python
                )
            else:
                # PID file existe pero proceso no está vivo = STALE
                return ProcessInfo(
                    service_name=service_name,
                    pid=pid,
                    state=ProcessState.STALE,
                    pid_file=str(pid_file),
                    is_ours=False
                )
                
        except ValueError:
            logger.error(f"[LIFECYCLE] Invalid PID in file: {pid_file}")
            return ProcessInfo(
                service_name=service_name,
                pid=0,
                state=ProcessState.UNKNOWN,
                pid_file=str(pid_file),
                is_ours=False
            )
        except Exception as e:
            logger.error(f"[LIFECYCLE] Error reading PID file: {e}")
            return None
    
    def cleanup_process(self, service_name: str, force: bool = False) -> bool:
        """
        Limpieza quirúrgica de un servicio.
        
        Flujo:
        1. Leer PID del archivo
        2. Verificar que el proceso esté vivo y sea nuestro
        3. SIGTERM (cierre gracioso)
        4. Esperar timeout
        5. SIGKILL si no responde
        6. Eliminar PID file
        
        Args:
            service_name: Nombre del servicio a limpiar
            force: Si True, usa SIGKILL inmediatamente
        
        Returns:
            True si el cleanup fue exitoso
        """
        info = self.get_process_info(service_name)
        
        if info is None:
            logger.info(f"[LIFECYCLE] No PID file for '{service_name}' - nothing to clean")
            return True
        
        if info.state == ProcessState.STALE:
            logger.info(f"[LIFECYCLE] Removing stale PID file for '{service_name}'")
            return self._remove_pid_file(service_name)
        
        if info.state != ProcessState.RUNNING:
            self._remove_pid_file(service_name)
            return True
        
        pid = info.pid
        logger.info(f"[LIFECYCLE] Stopping '{service_name}' [PID: {pid}]...")
        
        # 1. Llamar shutdown handler si existe
        if service_name in self._shutdown_handlers:
            try:
                self._shutdown_handlers[service_name]()
            except Exception as e:
                logger.warning(f"[LIFECYCLE] Shutdown handler failed: {e}")
        
        # 2. Cierre gracioso (SIGTERM)
        if not force:
            kill_process(pid, force=False)
            
            # Esperar a que termine
            for i in range(GRACEFUL_SHUTDOWN_TIMEOUT):
                time.sleep(1)
                if not is_process_alive(pid):
                    logger.info(f"[LIFECYCLE] ✅ '{service_name}' stopped gracefully")
                    self._remove_pid_file(service_name)
                    return True
            
            logger.warning(f"[LIFECYCLE] '{service_name}' did not stop gracefully, forcing...")
        
        # 3. Cierre forzado (SIGKILL)
        kill_process(pid, force=True)
        time.sleep(0.5)
        
        if not is_process_alive(pid):
            logger.info(f"[LIFECYCLE] ✅ '{service_name}' stopped (forced)")
            self._remove_pid_file(service_name)
            return True
        else:
            logger.error(f"[LIFECYCLE] ❌ Failed to stop '{service_name}'")
            return False
    
    def cleanup_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Limpia todos los servicios registrados.
        
        Returns:
            Dict[service_name, success]
        """
        results = {}
        
        # Limpiar todos los PID files encontrados
        if self._pid_dir.exists():
            for pid_file in self._pid_dir.glob("*.pid"):
                service_name = pid_file.stem
                results[service_name] = self.cleanup_process(service_name, force)
        
        return results
    
    def list_services(self) -> Dict[str, ProcessInfo]:
        """Lista todos los servicios con PID files."""
        services = {}
        
        if self._pid_dir.exists():
            for pid_file in self._pid_dir.glob("*.pid"):
                service_name = pid_file.stem
                info = self.get_process_info(service_name)
                if info:
                    services[service_name] = info
        
        return services
    
    @contextmanager
    def managed_service(self, service_name: str, shutdown_handler: Optional[Callable] = None):
        """
        Context manager para ciclo de vida automático.
        
        Uso:
            with lifecycle_manager.managed_service("my_service"):
                run_service()
        """
        try:
            self.register_process(service_name, shutdown_handler)
            yield
        finally:
            self._remove_pid_file(service_name)
            if service_name in self._registered_services:
                del self._registered_services[service_name]
    
    def get_status(self) -> Dict:
        """Retorna estado para diagnóstico."""
        services = self.list_services()
        return {
            "pid_directory": str(self._pid_dir.absolute()),
            "registered_count": len(services),
            "services": {
                name: {
                    "pid": info.pid,
                    "state": info.state.value,
                    "is_ours": info.is_ours
                }
                for name, info in services.items()
            }
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

lifecycle_manager = LifecycleManager()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def register_service(service_name: str, shutdown_handler: Optional[Callable] = None) -> bool:
    """Registra el proceso actual como un servicio."""
    return lifecycle_manager.register_process(service_name, shutdown_handler)

def cleanup_service(service_name: str, force: bool = False) -> bool:
    """Limpia un servicio específico."""
    return lifecycle_manager.cleanup_process(service_name, force)

def cleanup_all_services(force: bool = False) -> Dict[str, bool]:
    """Limpia todos los servicios."""
    return lifecycle_manager.cleanup_all(force)

def list_running_services() -> Dict[str, ProcessInfo]:
    """Lista servicios en ejecución."""
    return lifecycle_manager.list_services()
