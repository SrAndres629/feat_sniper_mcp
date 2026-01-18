from contextlib import contextmanager
from .manager import LifecycleManager, ProcessState, ProcessInfo

lifecycle_manager = LifecycleManager()

@contextmanager
def service_lifecycle(name: str):
    if lifecycle_manager.register_service(name):
        try:
            yield
        finally:
            lifecycle_manager.stop_service(name)
    else:
        raise RuntimeError(f"Could not start service {name}: already running.")

__all__ = ["lifecycle_manager", "service_lifecycle", "ProcessState", "ProcessInfo"]
