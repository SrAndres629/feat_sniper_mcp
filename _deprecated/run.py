import uvicorn
import os
from app.core.config import settings

if __name__ == "__main__":
    pid = os.getpid()
    print(f"🚀 Iniciando MT5 Neural Bridge en http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"🆔 PID del Proceso: {pid}")
    print(f"🔧 Modo Debug: {settings.DEBUG}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
