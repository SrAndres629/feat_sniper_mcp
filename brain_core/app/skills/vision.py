import base64
import io
import logging
import json
from typing import Dict, Any, Optional
try:
    import pyautogui
    GUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    GUI_AVAILABLE = False
from PIL import Image
import anyio

logger = logging.getLogger("MT5_Bridge.Skills.Vision")

async def capture_panorama(resize_factor: float = 0.75) -> Dict[str, Any]:
    """
    Captura la pantalla actual de la terminal MT5 de forma segura.
    """
    try:
        if not GUI_AVAILABLE:
            raise RuntimeError("Captura de pantalla no disponible en este entorno (sin GUI/PyAutoGUI).")
            
        # Ejecutamos la captura en un hilo para no bloquear el bucle de eventos
        screenshot = await anyio.to_thread.run_sync(pyautogui.screenshot)
        
        if screenshot is None:
            raise ValueError("La captura de pantalla devolvió un objeto nulo.")

        # Redimensionar para optimizar tokens y ancho de banda
        if resize_factor < 1.0:
            new_size = (int(screenshot.width * resize_factor), int(screenshot.height * resize_factor))
            screenshot = screenshot.resize(new_size, Image.Resampling.LANCZOS)
        
        # Guardar en buffer de memoria como JPEG (mejor compresión que PNG)
        buffer = io.BytesIO()
        screenshot.save(buffer, format="JPEG", quality=85)
        
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "width": screenshot.width,
            "height": screenshot.height,
            "format": "JPEG",
            "image_base64": img_str
        }
        
    except Exception as e:
        logger.error(f"Error en Vision Skill: {str(e)}")
        # Detectar errores comunes de pantalla bloqueada o permisos
        error_msg = str(e)
        suggestion = "Asegúrate de que la terminal MT5 esté maximizada y la pantalla no esté bloqueada."
        
        if "display" in error_msg.lower():
            error_msg = "Error de acceso al display (posible sesión RDP desconectada o pantalla bloqueada)."
        
        return {
            "status": "error",
            "error_code": "VISION_FAILURE",
            "message": error_msg,
            "suggestion": suggestion
        }
