import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger("MT5_MCP_Server.CustomLoader")

def register_custom_skills(mcp):
    """
    Escanea la carpeta 'indicadores propios' y registra cada .mq5 como una skill de lectura de datos.
    """
    base_dir = Path(os.getcwd())
    indicators_dir = base_dir / "FEAT_Sniper_Master_Core"
    config_path = indicators_dir / "Python" / "bridge_config.json"
    
    if not indicators_dir.exists():
        logger.warning(f"Directorio de indicadores no encontrado: {indicators_dir}")
        return

    # Cargar configuración de rutas de MT5
    watch_dir = None
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                watch_dir = Path(config.get("watch_dir", ""))
        except Exception as e:
            logger.error(f"Error cargando bridge_config.json: {e}")

    if not watch_dir or not watch_dir.exists():
        logger.warning("No se pudo determinar el watch_dir de MT5. Las skills personalizadas podrían no funcionar.")

    # Buscar archivos .mq5
    mq5_files = list(indicators_dir.glob("*.mq5"))
    logger.info(f"Detectados {len(mq5_files)} indicadores propios.")

    for mq5_file in mq5_files:
        indicator_name = mq5_file.stem
        register_indicator_tool(mcp, indicator_name, watch_dir)

    # Registrar herramientas de pipeline Python
    python_dir = indicators_dir / "Python"
    if python_dir.exists():
        register_python_pipeline_tools(mcp, python_dir)

def register_indicator_tool(mcp, indicator_name: str, watch_dir: Optional[Path]):
    """
    Registra dinámicamente una herramienta MCP para un indicador específico.
    """
    tool_name = f"skill_{indicator_name.lower().replace('.', '_')}"
    
    # Definir la función de la herramienta
    async def indicator_tool(symbol: str, timeframe: str = "H1", n_rows: int = 100):
        """
        Obtiene los datos exportados por el indicador personalizado.
        - symbol: Símbolo del activo (ej: EURUSD)
        - timeframe: Temporalidad (ej: H1, M15)
        - n_rows: Número de registros más recientes a retornar.
        """
        if not watch_dir:
            return {"status": "error", "message": "MT5 Files directory not configured in bridge_config.json"}
        
        # Patrón de búsqueda: IndicatorName_Export_Symbol_Timeframe.csv
        search_patterns = [
            f"{indicator_name}_Export_{symbol}_{timeframe}.csv",
            f"{indicator_name}_{symbol}_{timeframe}.csv",
            f"UnifiedModel_Export_{symbol}_{timeframe}.csv" if "Unified" in indicator_name else None
        ]
        
        csv_path = None
        for pattern in filter(None, search_patterns):
            potential_path = watch_dir / pattern
            if potential_path.exists():
                csv_path = potential_path
                break
        
        if not csv_path:
            return {
                "status": "error", 
                "message": f"No se encontró exportación para {indicator_name} ({symbol} {timeframe}) en {watch_dir}."
            }
        
        try:
            df = pd.read_csv(csv_path)
            data = df.tail(n_rows).to_dict(orient="records")
            return {
                "status": "success",
                "indicator": indicator_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "count": len(data),
                "data": data
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    mcp.tool(name=tool_name)(indicator_tool)
    logger.info(f"Skill registrada: {tool_name}")

def register_python_pipeline_tools(mcp, python_dir: Path):
    """
    Registra herramientas para ejecutar el pipeline de análisis de Python.
    """
    import subprocess
    import sys

    @mcp.tool(name="skill_run_unified_analysis")
    async def run_unified_analysis(symbol: str = "EURUSD", timeframe: str = "H1", data_path: Optional[str] = None):
        """
        Ejecuta el pipeline institucional completo: Ingesta, ML Training, Optimización y Dashboard.
        - symbol: Símbolo para el análisis.
        - timeframe: Temporalidad.
        - data_path: (Opcional) Ruta al CSV específico. Si no se da, usará la exportación más reciente.
        """
        pipeline_script = python_dir / "run_pipeline.py"
        if not pipeline_script.exists():
            return {"status": "error", "message": "run_pipeline.py no encontrado."}

        cmd = [sys.executable, str(pipeline_script), "--symbol", symbol, "--timeframe", timeframe, "--output", str(python_dir)]
        if data_path:
            cmd.extend(["--data", data_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(python_dir))
            if result.returncode == 0:
                # Buscar dashboard generado
                dashboard_path = python_dir / "dashboard.html"
                return {
                    "status": "success",
                    "output": result.stdout,
                    "dashboard_available": dashboard_path.exists(),
                    "dashboard_path": str(dashboard_path) if dashboard_path.exists() else None
                }
            else:
                return {"status": "error", "message": result.stderr, "stdout": result.stdout}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @mcp.tool(name="skill_get_analysis_report")
    async def get_analysis_report():
        """
        Devuelve el estado de los últimos resultados del pipeline (thresholds y métricas ML).
        """
        report_path = python_dir / "ml_thresholds.json"
        calib_path = python_dir / "optuna_calibration.json"
        
        results = {}
        if report_path.exists():
            with open(report_path, 'r') as f:
                results["ml_metrics"] = json.load(f)
        if calib_path.exists():
            with open(calib_path, 'r') as f:
                results["thresholds"] = json.load(f)
                
        if not results:
            return {"status": "error", "message": "No hay reportes disponibles. Ejecute skill_run_unified_analysis primero."}
            
        return {"status": "success", "results": results}

    logger.info("Pipeline skills registradas: skill_run_unified_analysis, skill_get_analysis_report")
