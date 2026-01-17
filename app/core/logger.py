import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import os

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "nexus_system.log"

def setup_logger(name="NexusCore"):
    # 1. Crear el logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 2. Limpiar handlers viejos (Vital)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 3. Propagar al root logger para permitir visibilidad en consola (warnings/errors)
    logger.propagate = True

    # 4. Handler de Archivo UNICAMENTE
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
        )
        file_handler.setFormatter(formatter)
        
        # Filtro Anti-Recursión Ouroboros
        class NoRecursionFilter(logging.Filter):
            def filter(self, record):
                return "SYSTEM_STDIO" not in record.name and "SYSTEM_STDIO" not in record.getMessage()
        
        file_handler.addFilter(NoRecursionFilter())
        logger.addHandler(file_handler)
    except Exception:
        # Failsafe
        logger.addHandler(logging.NullHandler())

    return logger

# Instancia global
logger = setup_logger()

# Función dummy para compatibilidad (ya no hace nada peligroso)
def hijack_streams(mcp_mode=False):
    pass 
