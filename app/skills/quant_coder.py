import logging
import os
import subprocess
from typing import Dict, Any
try:
    import MetaTrader5 as mt5
except ImportError:
    from unittest.mock import MagicMock
    mt5 = MagicMock()
from app.core.mt5_conn import mt5_conn
from app.models.schemas import MQL5CodeRequest
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Skills.QuantCoder")

async def create_native_indicator(req: MQL5CodeRequest) -> Dict[str, Any]:
    """
    Toma código MQL5, lo guarda en la carpeta de indicadores de MT5 e intenta compilarlo.
    """
    # 1. Obtener información del terminal para saber dónde guardar
    term_info = await mt5_conn.execute(mt5.terminal_info)
    if not term_info:
        return {"status": "error", "message": "No se pudo obtener la ruta del terminal MT5."}

    data_path = term_info.data_path
    indicators_path = os.path.join(data_path, "MQL5", "Indicators")
    
    if not os.path.exists(indicators_path):
        return {"status": "error", "message": f"No se encontró la carpeta de indicadores en {indicators_path}"}

    # 2. Guardar el archivo .mq5
    filename = f"{req.name}.mq5"
    if not filename.endswith(".mq5"):
        filename += ".mq5"
        
    full_path = os.path.join(indicators_path, filename)
    
    try:
        with open(full_path, "w", encoding="utf-16") as f: # MT5 prefiere UTF-16 para MQL5
            f.write(req.code)
        logger.info(f"Archivo MQL5 guardado en: {full_path}")
    except Exception as e:
        return {"status": "error", "message": f"Fallo al escribir el archivo: {str(e)}"}

    # 3. Intentar Compilación (Opcional pero recomendado)
    # Buscamos el metaeditor64.exe en la ruta del terminal
    terminal_path = term_info.path
    metaeditor_path = os.path.join(terminal_path, "metaeditor64.exe")
    
    compilation_status = "Skipped"
    if req.compile and os.path.exists(metaeditor_path):
        try:
            logger.info(f"Iniciando compilación con {metaeditor_path}...")
            # Comando: metaeditor64.exe /compile:"ruta_al_archivo" /log:"ruta_al_log"
            log_path = full_path + ".log"
            result = subprocess.run(
                [metaeditor_path, f"/compile:{full_path}", f"/log:{log_path}"],
                capture_output=True,
                text=True
            )
            
            # Leer el log generado por MetaEditor
            compile_log = "Log file not created."
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-16") as f_log:
                        compile_log = f_log.read()
                except Exception as e_read:
                    compile_log = f"Error reading log: {str(e_read)}"
            
            compilation_status = f"MetaEditor executed. Log output:\n{compile_log}"
            
        except Exception as e:
            compilation_status = f"Compilation Execution Error: {str(e)}"
    elif req.compile:
        compilation_status = "MetaEditor64.exe not found in terminal path (or path not configured). Code saved but not compiled."

    return {
        "status": "success",
        "message": f"Indicador '{req.name}' creado correctamente.",
        "path": full_path,
        "compilation": compilation_status,
        "note": "Revisa el output del log para confirmar si hubo errores de sintaxis."
    }
