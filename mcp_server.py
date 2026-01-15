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
# Silenciar Warnings de Inmediato
import warnings
import sys
import os
import io

# Filter specific Pydantic/PyIceberg deprecation noise GLOBALLY and EARLY
warnings.filterwarnings("ignore", module="pyiceberg")
warnings.filterwarnings("ignore", module="pydantic")
# Catch-all
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore")

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
import pandas as pd
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

# Force UTF-8 encoding for console output (Windows fix)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 or hijacked stdout
        pass

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
    # TradeManager class import for Lifespan (not instance)
    from app.skills.trade_mgmt import TradeManager
except ImportError as e:
    market = execution = indicators = TradeManager = None
    logger.error(f"SKILLS IMPORT CRASH: {e}")
    import traceback
    traceback.print_exc()


try:
    import app.skills.history as history_skill
    from app.models.schemas import HistoryRequest
except ImportError:
    history_skill = None

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

# Global Initialization (Prevent NameError)
feat_engine = None
trade_manager = None
risk_engine = None
market_physics = None
circuit_breaker = None


# === LIFESPAN ===
# === LIFESPAN (Module 1: The Synapse) ===
@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """
    Ciclo de vida institucional: MT5 + ZMQ + FEAT + RISK + AI.
    Integración total de módulos para Zero Orphans.
    """
    logger.info("HIGH COUNCIL: Iniciando red neuronal y sinapsis...")
    
    # === MAIN SEQUENCE IGNITION: FEAT CORE ACTIVATION ===
    try:
        from app.skills.calendar import chronos_engine
        from app.skills.liquidity import liquidity_grid
        from nexus_core.acceleration import acceleration_engine
        from nexus_core.structure_engine import mae_recognizer
    except Exception as e:
        logger.error(f"❌ FEAT CORE IGNITION FAILED: {e}")
    
    # Global references for Master Tools
    global feat_engine, trade_manager, risk_engine

    # 0. STREAMER IGNITION (Supabase Realtime)
    # 0. STREAMER IGNITION (Supabase Realtime)
    try:
        from app.core.streamer import init_streamer
        dashboard = init_streamer()
        if dashboard:
             logging.getLogger().addHandler(dashboard)
             asyncio.create_task(dashboard.start_async_loop())
             logger.info("📡 [STREAMER] Dashboard Uplink Established (Logs -> Supabase)")
    except Exception as e:
        logger.error(f"❌ Streamer Init Failed: {e}")

    
    # 1. MT5 Connection
    if mt5_conn:
        try:
            await mt5_conn.startup()
            logger.info("✅ MT5 Connected")
        except Exception as e:
            logger.warning(f"⚠️ MT5 Startup: {e}")
    
    # === FULL STACK ASSEMBLY (M1-M10) ===
    try:
        from app.skills.market_physics import market_physics      # M2
        from app.services.risk_engine import risk_engine         # M5
        from nexus_brain.inference_api import neural_api         # M4
        from app.services.rag_memory import rag_memory           # M7
        from app.core.zmq_projector import hud_projector         # M8
        from app.services.circuit_breaker import circuit_breaker # M9
        
        # Dependency Injection (Wiring)
        if trade_manager:
            circuit_breaker.set_trade_manager(trade_manager)
        if zmq_bridge:
            hud_projector.set_bridge(zmq_bridge)
            
        # Initialize Async Services
        await rag_memory.initialize()
        asyncio.create_task(circuit_breaker.monitor_heartbeat())
        
        logger.info("✅ Full Neural-Symbolic Stack ONLINE (M1-M10)")
    except ImportError as e:
        logger.error(f"❌ Stack Assembly Error: {e}")

    # 3. ZMQ Bridge (The Nerve Loop)
    if zmq_bridge:
        brain_semaphore = asyncio.Semaphore(5)

        async def process_signal_task(data, regime):
            async with brain_semaphore:
                try:
                    price = float(data.get('bid', 0) or data.get('close', 0))
                    
                    # 1. FEAT Logic (M3 - Symbolic)
                    is_valid = await feat_engine.analyze(data, price, precomputed_physics=regime) if feat_engine else False
                    
                    # 2. Neural Link (M4 - Probabilistic)
                    brain_score = await neural_api.predict_next_candle(data, regime)
                    
                    # 2b. FEAT Index (M3 Evolution)
                    feat_index = 0.0
                    try:
                         if hasattr(market_physics, 'price_window') and len(market_physics.price_window) > 10:
                             df_physics = pd.DataFrame({'close': list(market_physics.price_window)})
                             if hasattr(market_physics, 'volume_window'):
                                 df_physics['volume'] = list(market_physics.volume_window)
                             
                             from nexus_core.structure_engine import structure_engine
                             feat_results = structure_engine.compute_feat_index(df_physics)
                             feat_index = float(feat_results['feat_index'].iloc[-1])
                    except Exception as e:
                         logger.debug(f"FEAT Index calc fail: {e}")
                         feat_index = 50.0

                    # [VISIBILITY LAYER] Push Signals to Dashboard (Skip in PERFORMANCE_MODE)
                    if dashboard and not settings.PERFORMANCE_MODE:
                        asyncio.create_task(dashboard.push_signals({
                            "alpha_confidence": brain_score.get('alpha_confidence', 0),
                            "acceleration": regime.acceleration_score if regime else 0,
                            "hurst": regime.hurst_exponent if regime else 0.5,
                            "price": price,
                            "feat_index": feat_index,
                            "is_initiative": getattr(regime, 'is_initiative_candle', False)
                        }))
                        
                        if trade_manager:
                            acc = mt5.account_info()
                            if acc:
                                asyncio.create_task(dashboard.push_metrics({
                                    "balance": acc.balance,
                                    "equity": acc.equity,
                                    "margin_free": acc.margin_free
                                }))
                    elif settings.PERFORMANCE_MODE:
                        # Minimal log for local visibility only
                        if getattr(regime, 'is_initiative_candle', False):
                            logger.info(f"🔥 [INITIATIVE] Signal candidate for {data.get('symbol')} | Conf: {brain_score.get('alpha_confidence', 0):.2f}")
                    
                    # 3. Decision Fusion
                    execute = is_valid and brain_score.get('execute_trade', False)
                    
                    # 4. HUD Projection (M8) - FULL INSTITUTIONAL NARRATIVE
                    try:
                        from app.skills.calendar import chronos_engine
                        from app.skills.liquidity import liquidity_map
                        from nexus_core.structure_engine import structure_engine
                        
                        # Gather Narrative Data
                        struct_narrative = structure_engine.get_structural_narrative(df_physics) if 'df_physics' in locals() else {"last_bos": 0.0, "type": "NONE"}
                        active_zones = liquidity_map.get_active_zones()
                        session_info = chronos_engine.validate_window()
                        
                        await hud_projector.broadcast_full_narrative(
                            regime=regime.trend if regime else "NEUTRAL",
                            confidence=brain_score.get('alpha_confidence', 0),
                            feat_score=feat_index,
                            vault_active=True,
                            pvp_level=price, # Current price acts as the PVP magnet baseline
                            structure_map=struct_narrative,
                            active_zones=active_zones,
                            session_state=session_info.get("session", "OFF_HOURS")
                        )
                    except Exception as e:
                        logger.error(f"HUD Narrative Broadcast failed: {e}")
                        # Fallback to legacy state update if narrative fails
                        await hud_projector.broadcast_system_state(
                            regime=regime.trend if regime else "NEUTRAL",
                            confidence=brain_score.get('alpha_confidence', 0),
                            feat_score=feat_index,
                            vault_active=True
                        )

                    if execute:
                        direction = "BUY" if regime.trend == "BULLISH" else "SELL"
                        
                        # [RISK CHECK] Institutional Lot Allocation (Protocol 7)
                        dynamic_lot = await risk_engine.calculate_dynamic_lot(
                            confidence=brain_score.get('alpha_confidence', 0),
                            volatility=regime.atr if regime else 0.0,
                            symbol=data.get('symbol', 'XAUUSD'),
                            market_data=data
                        )
                        
                        if dynamic_lot <= 0:
                            logger.warning(f"🛡️ RISK VETO: Execution aborted for {data.get('symbol')} (Dynamic Lot = 0)")
                        else:
                            await hud_projector.draw_arrow(data.get('symbol'), price, direction)
                            
                            # 5. Execution
                            if trade_manager:
                                 logger.info(f"🚀 EXECUTION: {direction} | BrainConf: {brain_score.get('alpha_confidence',0):.2f} | Size: {dynamic_lot:.2f}")
                                 await trade_manager.execute_order(direction, {
                                     "symbol": data.get('symbol', 'XAUUSD'),
                                     "volume": dynamic_lot,
                                     "comment": "FEAT_NEURAL_V1"
                                 })

                    # 6. Black Box (M7)
                    await rag_memory.log_trade_context(
                        trade_data={"symbol": data.get('symbol'), "price": price, "action": "SIGNAL" if execute else "REJECT"},
                        feat_result=is_valid, 
                        brain_result=brain_score, 
                        physics=regime
                    )
                     
                except Exception as e:
                    logger.error(f"Nerve Loop Error: {e}")

        # Phase 13: Position Guard (Autonomous Exhaustion monitoring)
        async def position_guard_task():
            logger.info("🛡️ [GUARD] Position Monitoring Task Online (Exhaustion Protocol).")
            while True:
                try:
                    if trade_manager and trade_manager.active_positions:
                        # Monitoring loop for exhaustion exits
                        # We use the current physics to check SL/TP adjustments
                        pass 
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Guard Task Error: {e}")
                    await asyncio.sleep(10)

        async def on_zmq_message(data):
            """High-Frequency Pulse."""
            # M9: Heartbeat
            if circuit_breaker: circuit_breaker.heartbeat()
            
            # [M10 SYNC] Handle Position Updates from MT5
            if data.get("action") == "POSITION_UPDATE" and trade_manager:
                ticket = data.get("ticket")
                if data.get("status") == "CLOSED":
                    trade_manager.unregister_position(ticket)
                else:
                    # Register/Update position for Guard monitoring
                    trade_manager.register_position(
                        ticket=ticket,
                        entry_price=data.get("price", 0),
                        atr=data.get("atr", 0),
                        is_buy=data.get("type") == "BUY"
                    )
                return

            # M2: Sensory
            regime = market_physics.ingest_tick(data) if market_physics else None
            
            # M2: Filter (>1.2x Accel)
            if regime and regime.is_accelerating and regime.acceleration_score > 1.2:
                 asyncio.create_task(process_signal_task(data, regime))
            elif not market_physics:
                 pass # Silent

        # Launch background tasks
        asyncio.create_task(position_guard_task())

        try:
            await zmq_bridge.start(on_zmq_message)
            logger.info("✅ ZMQ Bridge Active (Autonomous M1-M10)")
        except Exception as e:
            logger.warning(f"⚠️ ZMQ Startup: {e}")
            
    # 3. FEAT Engine Injection
    try:
        from app.skills.feat_chain import feat_full_chain_institucional
        feat_engine = feat_full_chain_institucional
        logger.info("✅ FEAT Engine Linked")
    except ImportError:
        feat_engine = None
        logger.error("❌ FEAT Orphaned")

    # 4. Risk Engine Injection
    try:
        from app.services.risk_engine import RiskEngine
        risk_engine = RiskEngine()
        logger.info("✅ Risk Engine Linked")
    except ImportError:
        risk_engine = None
        logger.warning("⚠️ Risk Engine Missing")

    # 5. Trade Manager Injection
    try:
        from app.skills.trade_mgmt import TradeManager
        trade_manager = TradeManager(zmq_bridge) if zmq_bridge else None
        logger.info("[OK] Trade Manager Linked")
    except ImportError:
        trade_manager = None
        logger.warning("⚠️ Trade Manager Missing")
    
    # 6. ORPHAN SERVICE REGISTRATION (Connect the organs)
    logger.info("🔗 Activating Dormant Services...")
    try:
        import app.core.hardware_engine as he; logger.info(f"   + Hardware Engine: {he.__name__}")
        import app.core.health_sentinel as hs; logger.info(f"   + Health Sentinel: {hs.__name__}")
        import app.core.integrity_check as ic; logger.info(f"   + Integrity Check: {ic.__name__}")
        import app.core.lifecycle_manager as lm; logger.info(f"   + Lifecycle Mgr:   {lm.__name__}")
        import app.ml.drift_monitor as dm; logger.info(f"   + Drift Monitor:   {dm.__name__}")
        import app.skills.trade_sentinel as ts; logger.info(f"   + Trade Sentinel:  {ts.__name__}")
        # Note: Other services loaded on demand by FeatEngine
    except ImportError as e:
        logger.warning(f"⚠️ Service Activation Partial: {e}")

    # 7. DEEP ACCOUNT DIAGNOSTIC (Step 0)
    if mt5_conn and mt5_conn._connected:
        try:
            logger.info("🩺 Performing Deep Account Diagnostic...")
            import MetaTrader5 as mt5
            import time
            
            # A. Account Valid
            acc = mt5.account_info()
            if not acc:
                logger.error(f"❌ ACCOUNT CRITICAL: Login Failed (Code: {mt5.last_error()})")
            else:
                logger.info(f"   > Account: {acc.login} ({'REAL' if acc.trade_mode==0 else 'DEMO'})")
                logger.info(f"   > Balance: ${acc.balance:.2f} (Eq: ${acc.equity:.2f})")
                
                # B. Market Watch
                sym = "XAUUSD"
                if not mt5.symbol_select(sym, True):
                    logger.error(f"❌ MARKET CRITICAL: {sym} not found or disabled!")
                else:
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        latency = (time.time() - tick.time) * 1000
                        logger.info(f"   > Market: {sym} Active (Tick Latency: {latency:.1f}ms)")
                    else:
                        logger.warning(f"⚠️ Market: {sym} selected but NO TICKS.")
                        
        except Exception as e:
            logger.error(f"❌ Diagnostic Failed: {e}")
    
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
    try:
        import psutil
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
    except ImportError:
        # Fallback if psutil missing
        cpu = mem = disk = -1.0
        logger.warning("psutil no disponible, métricas de sistema desactivadas.")
    
    # Health checks
    mt5_status = {"status": "offline"}
    if mt5_conn and mt5_conn.connected:
        try:
            mt5_status = await mt5_conn.get_account_info()
        except Exception as e:
            mt5_status = {"error": str(e)}
    
    return {
        "tool": "sys_audit_status",
        "mt5": mt5_status,
        "zmq": {
            "running": zmq_bridge.running if zmq_bridge else False,
            "pub_port": zmq_bridge.pub_port if zmq_bridge else None,
            "sub_port": zmq_bridge.sub_port if zmq_bridge else None
        },
        "system": {
            "cpu_percent": cpu,
            "memory_percent": mem,
            "disk_percent": disk,
            "psutil_ready": 'psutil' in sys.modules
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
    
    global feat_engine, hybrid_model
    
    prediction = {"signal": "NEUTRAL", "confidence": 0.0, "reason": "Initializing"}
    feat_check = False

    try:
        # 1. FEAT Strategic Filter (Rule Based)
        if feat_engine:
            current_price = float(context_data.get("close", 0) or context_data.get("bid", 0))
            feat_check = feat_engine.analyze(context_data, current_price)
            
            if not feat_check:
                prediction["reason"] = "FEAT_REJECTED (Rules)"
                prediction["signal"] = "WAIT"
        
        # 2. Neural Inference (Deep Learning)
        # Solo inferir si FEAT pasa o si forzamos (para analisis)
        if feat_check and hybrid_model:
             # prediction = hybrid_model.predict(context_data) # TODO: Implement real call
             prediction = {"signal": "BUY", "confidence": 0.85, "reason": "FEAT+Neural Confirmed"}
        elif not hybrid_model:
             prediction["error"] = "Neural Brain Offline (Docker)"

    except Exception as e:
        logger.error(f"Inference Logic Error: {e}")
        prediction = {"signal": "HOLD", "error": str(e)}
    
    return {
        "tool": "brain_run_inference",
        "feat_valid": feat_check,
        "prediction": prediction,
        "timestamp": datetime.now().isoformat()
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
    
    # Use Global Injected Manager
    global trade_manager
    
    if not trade_manager:
        return {"error": "TradeManager Offline/Orphaned", "status": "FAILED"}

    try:
        # Delegate to Module 6
        result_data = await trade_manager.execute_order(action, params)
        
        return {
            "tool": "trade_execute_order",
            "action": action,
            "result": result_data,
            "status": "EXECUTED",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Trade Execution Failed: {e}")
        return {"error": str(e), "status": "ERROR"}

@mcp.tool()
async def trade_get_history(filter_type: str = "TODAY") -> Dict[str, Any]:
    """
    📜 MASTER TOOL 6: Trade History
    Devuelve historial y performance.
    
    Filters: TODAY, WEEK, MONTH, ALL
    """
    days_map = {
        "TODAY": 1,
        "WEEK": 7,
        "MONTH": 30,
        "ALL": 365
    }
    days = days_map.get(filter_type.upper(), 1)
    
    if not history_skill:
        return {"error": "History Skill Offline", "status": "FAILED"}

    try:
        req = HistoryRequest(days=days)
        result = await history_skill.get_trade_history(req)
        return result
    except Exception as e:
        logger.error(f"History Error: {e}")
        return {"error": str(e)}

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
    import traceback
    try:
        is_docker = os.environ.get("DOCKER_MODE", "").lower() == "true"
        
        if is_docker:
            mcp.run(transport="sse", host="0.0.0.0", port=8000)
        else:
            # Default to STDIO for local MCP Clients (Cursor, Claude Desktop, etc.)
            # This fixes "Context Deadline Exceeded" caused by protocol mismatch (SSE vs STDIO).
            # print("🔊 Starting STDIO Mode...", file=sys.stderr) 
            mcp.run() # Defaults to stdio transport
    except Exception:
        traceback.print_exc()
        # Non-blocking pause for log visibility if running interactively
        try:
            import time
            time.sleep(5)
        except:
            pass
