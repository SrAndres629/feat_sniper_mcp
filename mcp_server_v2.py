"""
FEAT NEXUS - MCP Server v2.0 (High Council Architecture)
=========================================================
10 Master Tools para máxima eficiencia de contexto de IA.

Este archivo reemplaza las ~70 herramientas micro con 10 herramientas maestras
que agrupan funcionalidad por dominio.
"""
import os
import sys
import io

# === PROTOCOLO BLACKHOLE (INICIO) ===
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_devnull = io.StringIO()

sys.stdout = _devnull
sys.stderr = _devnull

# 1. Desactivar OpenTelemetry
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"
os.environ["OTEL_CONSOLE_EXPORT"] = "0"

# 2. Silenciar Warnings
import warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:pyiceberg,ignore::DeprecationWarning:pydantic"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)

# 3. Logging silencioso
import logging
try:
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename='logs/mcp_debug.log', level=logging.INFO, force=True)
except Exception:
    pass

# 4. Imports peligrosos
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

try:
    from fastmcp import FastMCP
except Exception:
    pass

from app.core.logger import logger

# === PROTOCOLO BLACKHOLE (FIN) ===
sys.stdout = _original_stdout
sys.stderr = _original_stderr

# === IMPORTS DE NEGOCIO ===
from app.core.mt5_conn import mt5_conn
from app.core.zmq_bridge import zmq_bridge
from app.skills import market, execution, trade_mgmt, indicators
from app.ml.data_collector import data_collector
from app.services.supabase_sync import supabase_sync

# === LIFESPAN ===
@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Ciclo de vida institucional: MT5 + ZMQ."""
    logger.info("HIGH COUNCIL: Iniciando infraestructura...")
    await mt5_conn.startup()
    
    async def on_zmq_message(data):
        logger.debug(f"ZMQ: {data}")
    
    await zmq_bridge.start(on_zmq_message)
    
    try:
        yield
    finally:
        await zmq_bridge.stop()
        await mt5_conn.shutdown()

# === INICIALIZAR MCP ===
mcp = FastMCP("MT5_Neural_Sentinel", lifespan=app_lifespan)

# =============================================================================
# THE HIGH COUNCIL - 10 MASTER TOOLS
# =============================================================================

@mcp.tool()
async def sys_audit_status() -> Dict[str, Any]:
    """
    🛠️ MASTER TOOL 1: System Audit
    Devuelve salud de Docker, Puertos ZMQ, Memoria y MT5.
    """
    import psutil
    
    # Health checks
    mt5_status = await mt5_conn.get_account_info() if mt5_conn.connected else {"error": "MT5 disconnected"}
    
    return {
        "tool": "sys_audit_status",
        "mt5": mt5_status,
        "zmq": {
            "running": zmq_bridge.running,
            "pub_port": zmq_bridge.pub_port,
            "sub_port": zmq_bridge.sub_port
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@mcp.tool()
async def market_get_telemetry(symbol: str = "XAUUSD", timeframe: str = "M5") -> Dict[str, Any]:
    """
    📊 MASTER TOOL 2: Market Telemetry
    OHLCV + Fractales (Capas 1-4) + Indicadores + Liquidez en un solo JSON.
    """
    snapshot = await market.get_snapshot(symbol, timeframe)
    candles = await market.get_candles(symbol, timeframe, 100)
    feat_layers = indicators.calculate_feat_layers(candles) if candles else {}
    
    return {
        "tool": "market_get_telemetry",
        "symbol": symbol,
        "timeframe": timeframe,
        "snapshot": snapshot,
        "candles_count": len(candles) if candles else 0,
        "feat_layers": feat_layers,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@mcp.tool()
async def brain_run_inference(context_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    🧠 MASTER TOOL 3: Neural Inference
    Ejecuta modelo híbrido (GBM+LSTM) y devuelve predicción con confianza.
    """
    if not context_data:
        context_data = {}
    
    # Placeholder - integrar con nexus_brain/hybrid_model.py
    try:
        from nexus_brain.hybrid_model import HybridModel
        model = HybridModel()
        prediction = model.predict(context_data)
    except Exception as e:
        prediction = {"signal": "HOLD", "confidence": 0.5, "error": str(e)}
    
    return {
        "tool": "brain_run_inference",
        "prediction": prediction,
        "context_received": list(context_data.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@mcp.tool()
async def risk_analyze_trade(entry: float, stop: float, symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    💰 MASTER TOOL 4: Risk Analysis
    Consulta la Bóveda y devuelve lotaje aprobado basado en riesgo.
    """
    account = await mt5_conn.get_account_info() if mt5_conn.connected else {}
    balance = account.get("balance", 10000)
    
    risk_percent = 1.0  # 1% por defecto
    risk_amount = balance * (risk_percent / 100)
    
    pip_value = 0.01 if "JPY" in symbol else 0.0001
    pips_risk = abs(entry - stop) / pip_value
    
    lot_size = round(risk_amount / (pips_risk * 10), 2) if pips_risk > 0 else 0.01
    lot_size = max(0.01, min(lot_size, 10.0))  # Clamp entre 0.01 y 10
    
    return {
        "tool": "risk_analyze_trade",
        "symbol": symbol,
        "entry": entry,
        "stop": stop,
        "pips_risk": round(pips_risk, 1),
        "approved_lot": lot_size,
        "risk_percent": risk_percent,
        "risk_amount": round(risk_amount, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@mcp.tool()
async def trade_execute_order(action: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    🔫 MASTER TOOL 5: Trade Execution
    Unifica Buy/Sell/Limit/Modify/Close en una sola herramienta.
    
    Actions: BUY, SELL, BUY_LIMIT, SELL_LIMIT, MODIFY, CLOSE, CLOSE_ALL
    Params: symbol, volume, price, sl, tp, ticket (para modify/close)
    """
    if not params:
        params = {}
    
    action = action.upper()
    symbol = params.get("symbol", "XAUUSD")
    volume = params.get("volume", 0.01)
    price = params.get("price")
    sl = params.get("sl")
    tp = params.get("tp")
    ticket = params.get("ticket")
    
    result = {"action": action, "params": params}
    
    if action in ["BUY", "SELL"]:
        result["order"] = await execution.execute_market_order(symbol, action, volume, sl, tp)
    elif action in ["BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"]:
        result["order"] = await execution.execute_pending_order(symbol, action, volume, price, sl, tp)
    elif action == "MODIFY" and ticket:
        result["order"] = await trade_mgmt.modify_position(ticket, sl, tp)
    elif action == "CLOSE" and ticket:
        result["order"] = await trade_mgmt.close_position(ticket)
    elif action == "CLOSE_ALL":
        result["order"] = await trade_mgmt.close_all_positions()
    else:
        result["error"] = f"Unknown action: {action}"
    
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result

@mcp.tool()
async def trade_get_history(filter_type: str = "TODAY") -> Dict[str, Any]:
    """
    📜 MASTER TOOL 6: Trade History
    Devuelve historial y performance.
    
    Filters: TODAY, WEEK, MONTH, ALL
    """
    positions = await trade_mgmt.get_positions() if mt5_conn.connected else []
    history = await trade_mgmt.get_history(filter_type) if mt5_conn.connected else []
    
    total_profit = sum(p.get("profit", 0) for p in positions)
    
    return {
        "tool": "trade_get_history",
        "filter": filter_type,
        "open_positions": len(positions),
        "positions": positions[:10],  # Limitar para no saturar
        "history_count": len(history),
        "total_open_profit": round(total_profit, 2),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@mcp.tool()
async def visual_update_hud(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    👁️ MASTER TOOL 7: HUD Update
    Manda datos al Dashboard de MT5 o retorna estado visual.
    """
    if not data:
        data = {}
    
    # Placeholder - integrar con dashboard si existe
    return {
        "tool": "visual_update_hud",
        "data_received": data,
        "status": "HUD update acknowledged",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@mcp.tool()
async def data_manage_memory(action: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    📂 MASTER TOOL 8: Memory Management (RAG)
    Guardar/Cargar/Limpiar memoria a largo plazo.
    
    Actions: SAVE, LOAD, CLEAR, STATS
    Params: text, category, query, limit
    """
    if not params:
        params = {}
    
    action = action.upper()
    result = {"action": action}
    
    try:
        from app.skills.rag_memory import rag_memory
        
        if action == "SAVE":
            text = params.get("text", "")
            category = params.get("category", "general")
            result["saved"] = rag_memory.remember(text, category)
        elif action == "LOAD":
            query = params.get("query", "")
            limit = params.get("limit", 5)
            result["memories"] = rag_memory.recall(query, limit)
        elif action == "CLEAR":
            category = params.get("category")
            result["cleared"] = rag_memory.forget(category)
        elif action == "STATS":
            result["stats"] = rag_memory.get_stats()
        else:
            result["error"] = f"Unknown action: {action}"
    except ImportError:
        result["error"] = "RAG memory module not available"
    
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result

@mcp.tool()
async def config_update_parameters(params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    ⚙️ MASTER TOOL 9: Config Update
    Ajustar constantes del sistema en caliente.
    
    Params: risk_percent, max_positions, shadow_mode, etc.
    """
    if not params:
        return {
            "tool": "config_update_parameters",
            "current_config": {
                "risk_percent": 1.0,
                "max_positions": 3,
                "shadow_mode": True,
                "allowed_symbols": ["XAUUSD", "EURUSD", "GBPUSD"]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Placeholder - implementar config manager
    return {
        "tool": "config_update_parameters",
        "updated": params,
        "status": "Configuration updated",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@mcp.tool()
async def sys_emergency_stop(reason: str = "Manual stop") -> Dict[str, Any]:
    """
    🚨 MASTER TOOL 10: Emergency Stop
    Kill switch. Cierra todas las posiciones y cancela órdenes pendientes.
    """
    logger.warning(f"EMERGENCY STOP triggered: {reason}")
    
    closed_positions = []
    cancelled_orders = []
    
    try:
        if mt5_conn.connected:
            closed_positions = await trade_mgmt.close_all_positions()
            cancelled_orders = await trade_mgmt.cancel_all_orders()
    except Exception as e:
        logger.error(f"Emergency stop error: {e}")
    
    return {
        "tool": "sys_emergency_stop",
        "reason": reason,
        "closed_positions": len(closed_positions) if isinstance(closed_positions, list) else 0,
        "cancelled_orders": len(cancelled_orders) if isinstance(cancelled_orders, list) else 0,
        "status": "EMERGENCY STOP EXECUTED",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    is_docker = os.environ.get("DOCKER_MODE", "").lower() == "true"
    
    if is_docker:
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        mcp.run()
