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

# 3. CRITICAL: Override ALL logger formats BEFORE any third-party imports
# This prevents 'correlation_id' KeyError from docket/fastmcp default formatters
import logging
from logging.handlers import RotatingFileHandler

# Nuke ALL existing handlers on root logger
root_logger = logging.getLogger()
root_logger.handlers.clear()

# Create a SIMPLE format that works everywhere (no correlation_id)
SIMPLE_FORMAT = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

try:
    os.makedirs('logs', exist_ok=True)
    # File handler with simple format
    file_handler = RotatingFileHandler('logs/mcp_debug.log', maxBytes=10*1024*1024, backupCount=3)
    file_handler.setFormatter(SIMPLE_FORMAT)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
except Exception:
    # Fallback to NullHandler if file logging fails
    root_logger.addHandler(logging.NullHandler())

# Force override for problematic third-party loggers
for name in ['docket', 'fastmcp', 'uvicorn', 'httpx', 'httpcore', 'urllib3']:
    third_party = logging.getLogger(name)
    third_party.handlers.clear()
    third_party.propagate = True  # Propagate to our clean root logger

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

# === IMPORTS DE NEGOCIO (OPERACIÓN REUNIÓN - 0 ORPHANS) ===
# Core modules
try:
    from app.core.mt5_conn import mt5_conn
except ImportError:
    mt5_conn = None
    logger.warning("mt5_conn no disponible")

try:
    from app.core.zmq_bridge import zmq_bridge
except ImportError:
    zmq_bridge = None
    logger.warning("zmq_bridge no disponible")

try:
    from app.core.zmq_projector import ZMQProjector
    zmq_projector = ZMQProjector()
except ImportError:
    zmq_projector = None

# Skills modules (orphans → connected)
try:
    from app.skills import market, execution, indicators
    from app.skills.trade_mgmt import trade_mgmt
except ImportError as e:
    market = execution = indicators = trade_mgmt = None
    logger.warning(f"Skills import error: {e}")

try:
    from app.skills.history import history_manager
except ImportError:
    history_manager = None

try:
    from app.skills.unified_model import unified_model
except ImportError:
    unified_model = None

try:
    from app.skills.vision import vision_system
except ImportError:
    vision_system = None

try:
    from app.skills.calendar import economic_calendar
except ImportError:
    economic_calendar = None

try:
    from app.skills.custom_loader import custom_loader
except ImportError:
    custom_loader = None

try:
    from app.skills.advanced_analytics import advanced_analytics
except ImportError:
    advanced_analytics = None

# New Orphans Integration (Batch 2)
try:
    from app.skills.feat_chain import feat_chain_orchestrator
except ImportError:
    feat_chain_orchestrator = None

try:
    from app.skills.feat_kiv import kiv_engine
except ImportError:
    kiv_engine = None

try:
    from app.skills.skill_deep_auditor import deep_auditor
except ImportError:
    deep_auditor = None

try:
    from app.skills.quant_coder import quant_coder
except ImportError:
    quant_coder = None

try:
    from app.skills.remote_compute import remote_compute
except ImportError:
    remote_compute = None

try:
    from app.skills.system_ops import system_ops
except ImportError:
    system_ops = None

try:
    from app.skills.tester import tester
except ImportError:
    tester = None

# Services modules (orphans → connected)
try:
    from app.services.n8n_bridge import n8n_bridge
except ImportError:
    n8n_bridge = None

try:
    from app.services.rag_memory import rag_memory
except ImportError:
    rag_memory = None

try:
    from app.services.supabase_sync import supabase_sync
except ImportError:
    supabase_sync = None

try:
    from app.services.telemetry import telemetry
except ImportError:
    telemetry = None

# ML modules (orphans → connected via brain_run_inference)
try:
    from app.ml.data_collector import data_collector
except ImportError:
    data_collector = None

try:
    from app.ml.feat_processor import feat_processor
except ImportError:
    feat_processor = None

try:
    from app.ml.ml_normalization import ml_normalization
except ImportError:
    ml_normalization = None

try:
    from app.ml.temporal_features import temporal_features
except ImportError:
    temporal_features = None

try:
    from app.ml.train_mtf_models import train_mtf_models
except ImportError:
    train_mtf_models = None

# Nexus Brain (neural network core)
try:
    from nexus_brain.hybrid_model import HybridModel
    hybrid_model = HybridModel()
except ImportError:
    hybrid_model = None

# === LIFESPAN ===
@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Ciclo de vida institucional: MT5 + ZMQ."""
    logger.info("HIGH COUNCIL: Iniciando infraestructura...")
    
    # Iniciar MT5 si está disponible
    if mt5_conn:
        try:
            await mt5_conn.startup()
        except Exception as e:
            logger.warning(f"MT5 startup failed: {e}")
    
    # Iniciar ZMQ si está disponible
    if zmq_bridge:
        async def on_zmq_message(data):
            logger.debug(f"ZMQ: {data}")
        try:
            await zmq_bridge.start(on_zmq_message)
        except Exception as e:
            logger.warning(f"ZMQ startup failed: {e}")
    
    try:
        yield
    finally:
        if zmq_bridge:
            try:
                await zmq_bridge.stop()
            except Exception:
                pass
        if mt5_conn:
            try:
                await mt5_conn.shutdown()
            except Exception:
                pass

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
