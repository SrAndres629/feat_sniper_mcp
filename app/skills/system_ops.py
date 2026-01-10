"""
System Operations Skill - Auto-Diagnóstico y Control del Host
==============================================================
Permite que la IA monitoree y gestione la salud del sistema.
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("MT5_Bridge.SystemOps")


async def system_health_check() -> Dict[str, Any]:
    """
    Revisa la salud del servidor (CPU, RAM, Disco).
    Útil para auto-diagnóstico antes de operaciones pesadas.
    """
    try:
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count()
        
        # Memory
        mem = psutil.virtual_memory()
        
        # Disk
        disk = psutil.disk_usage('/')
        
        # Network (bytes sent/received)
        net = psutil.net_io_counters()
        
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "cpu": {
                "percent": cpu_percent,
                "cores": cpu_count,
                "status": "warning" if cpu_percent > 80 else "ok"
            },
            "memory": {
                "percent": mem.percent,
                "used_gb": round(mem.used / (1024**3), 2),
                "total_gb": round(mem.total / (1024**3), 2),
                "status": "warning" if mem.percent > 85 else "ok"
            },
            "disk": {
                "percent": disk.percent,
                "used_gb": round(disk.used / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2),
                "status": "critical" if disk.percent > 90 else ("warning" if disk.percent > 80 else "ok")
            },
            "network": {
                "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
                "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2)
            }
        }
        
        # Overall status
        if health["disk"]["status"] == "critical":
            health["status"] = "critical"
        elif health["cpu"]["status"] == "warning" or health["memory"]["status"] == "warning":
            health["status"] = "warning"
        
        logger.info(f"System health check: {health['status']}")
        return health
        
    except ImportError:
        return {
            "status": "unavailable",
            "error": "psutil not installed",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


async def get_process_info() -> Dict[str, Any]:
    """
    Obtiene información sobre los procesos del contenedor.
    """
    try:
        import psutil
        
        current = psutil.Process()
        
        return {
            "pid": current.pid,
            "name": current.name(),
            "status": current.status(),
            "memory_mb": round(current.memory_info().rss / (1024**2), 2),
            "cpu_percent": current.cpu_percent(interval=0.1),
            "threads": current.num_threads(),
            "uptime_seconds": round((datetime.now() - datetime.fromtimestamp(current.create_time())).total_seconds()),
            "children": len(current.children())
        }
    except Exception as e:
        return {"error": str(e)}


async def get_environment_info() -> Dict[str, Any]:
    """
    Obtiene información del entorno de ejecución.
    """
    import sys
    import platform
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "hostname": platform.node(),
        "docker_mode": os.environ.get("DOCKER_MODE", "false"),
        "chroma_dir": os.environ.get("CHROMA_PERSIST_DIR", "not set"),
        "cwd": os.getcwd()
    }


async def list_running_tasks() -> Dict[str, Any]:
    """
    Lista las tareas asyncio activas.
    Útil para debugging de procesos colgados.
    """
    tasks = asyncio.all_tasks()
    return {
        "count": len(tasks),
        "tasks": [
            {
                "name": t.get_name(),
                "done": t.done(),
                "cancelled": t.cancelled()
            }
            for t in list(tasks)[:20]  # Limit to 20
        ]
    }


async def cleanup_cache() -> Dict[str, Any]:
    """
    Limpia caches de Python para liberar memoria.
    """
    import gc
    
    # Force garbage collection
    collected = gc.collect()
    
    return {
        "garbage_collected": collected,
        "status": "cleaned"
    }


# Función peligrosa - requiere confirmación explícita
async def restart_internal_service(service: str) -> Dict[str, Any]:
    """
    (PELIGROSO) Intenta reiniciar servicios internos.
    Solo debe usarse como último recurso.
    
    Servicios válidos: zmq_bridge, rag_memory, watchdog
    """
    valid_services = ["zmq_bridge", "rag_memory", "watchdog"]
    
    if service not in valid_services:
        return {
            "status": "error",
            "message": f"Servicio inválido. Opciones: {valid_services}"
        }
    
    logger.warning(f"⚠️ Restart solicitado para: {service}")
    
    # Por seguridad, solo logueamos la intención
    # El restart real requeriría lógica específica por servicio
    return {
        "status": "logged",
        "service": service,
        "message": "Restart request logged. Manual intervention may be required.",
        "timestamp": datetime.utcnow().isoformat()
    }
