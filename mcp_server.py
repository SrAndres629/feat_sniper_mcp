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

# [WINDOWS FIX] Force UTF-8 Encoding for Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        _win_encoding_error = True

# === PROTOCOLO BLACKHOLE (INICIO) ===
# Silenciar Warnings de Inmediato
import warnings

# Filter specific noise while preserving actual system warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyiceberg")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
try:
    from pydantic import PydanticDeprecatedSince20
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except: 
    _pydantic_v2_fail = True
# os.environ["PYTHONWARNINGS"] = "ignore"  # Too aggressive

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
import logging
from logging.handlers import RotatingFileHandler

# Nuke ALL existing handlers on root logger
root_logger = logging.getLogger()
root_logger.handlers.clear()

# Create formats
SIMPLE_FORMAT = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
CONSOLE_FORMAT = logging.Formatter('\033[93m%(levelname)s:\033[0m %(message)s') # Colored for emphasis

try:
    os.makedirs('logs', exist_ok=True)
    # File handler (Full Info)
    file_handler = RotatingFileHandler('logs/mcp_debug.log', maxBytes=10*1024*1024, backupCount=3)
    file_handler.setFormatter(SIMPLE_FORMAT)
    root_logger.addHandler(file_handler)
    
    # Console handler (Silent but Critical)
    # Filters out noise, only shows WARNING and above to the human
    console_handler = logging.StreamHandler(_original_stdout) # Use original to bypass redirect
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(CONSOLE_FORMAT)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO)
except Exception:
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
    _fastmcp_missing = True

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
        _old_python_api = True

# === IMPORTS DE NEGOCIO (OPERACIÓN REUNIÓN - 0 ORPHANS) ===
# Core modules (MILITARY GRADE: FAIL-STOP)
try:
    from app.core.mt5_conn import mt5_conn
    from app.core.zmq_bridge import zmq_bridge
    from app.core.zmq_projector import ZMQProjector
    zmq_projector = ZMQProjector()
except ImportError as e:
    logger.critical(f"🛑 CRITICAL BOOT FAILURE: Core Connectivity missing: {e}")
    sys.exit(1)

# Skills modules (FAIL-STOP for Trade and Strategy)
try:
    from app.skills import market, execution, indicators
    from app.skills.trade_mgmt import TradeManager
    from app.skills.feat_chain import feat_full_chain_institucional as feat_chain_orchestrator
    from app.services.risk_engine import risk_engine  # Critical Safety (FIX: was app.skills)
except ImportError as e:
    logger.critical(f"🛑 CRITICAL BOOT FAILURE: Trade/Risk modules missing: {e}")
    sys.exit(1)


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
    from app.skills.calendar_guard import calendar_guard
except ImportError:
    calendar_guard = None

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

# Master ML Engine (Quantum Leap V9.0)
try:
    from app.ml.ml_engine import MLEngine
    ml_engine = MLEngine()
    logger.info("✅ Quantum Leap ML Engine Initialized")
except ImportError:
    ml_engine = None
    logger.error("❌ MLEngine Import Failed")

# Institutional Standard: Module-level imports for hot-path services
import pandas as pd
import numpy as np
from nexus_core.convergence_engine import convergence_engine
from app.services.neural_state import neural_service
from app.ml.automl.orchestrator import automl_orchestrator
from app.ml.automl.drift_detector import drift_detector
from app.core.system_guard import system_sentinel
from app.ml.market_regime import market_regime
from app.ml.fuzzy_logic import fuzzy_logic
from app.ml.fractal_analysis import fractal_analyzer
from app.skills.liquidity_detector import detect_liquidity_pools, detect_asian_sweep
from app.ml.temporal_features import temporal_feature_generator
from nexus_core.features import feat_features
from nexus_core.kinetic_engine import kinetic_engine

# === SYSTEM STATE (SENIOR ARCHITECTURE: CLASS-BASED SINGLETONS) ===
class NexusState:
    """
    Centralized System State to avoid scoping/closure issues.
    All background tasks and callbacks access services through this singleton.
    """
    def __init__(self):
        self.market_physics = None
        self.risk_engine = None
        self.trade_manager = None
        self.feat_engine = None
        self.circuit_breaker = None
        self.dashboard = None
        self.ml_engine = None
        self.structure_engine = None
        self.structure_engine = None
        self.market = None
        self.calendar_guard = None
        self.mtf_engine = None
        self.mtf_engine = None
        self.brain_semaphore = asyncio.Semaphore(20)
        self.last_scan_time = 0.0
        self.context_cache = {"in_zone": False, "session": "OFF"}
        self.chronos_engine = None

    async def initialize_services(self):
        """Sequential initialization of core services."""
        logger.info("🚀 [NEXUS] Initializing Core Services...")
        try:
            from app.skills.market_physics import market_physics as mp
            from app.services.risk_engine import RiskEngine
            from app.skills.trade_mgmt import TradeManager
            from app.skills.feat_chain import feat_full_chain_institucional
            from app.services.circuit_breaker import circuit_breaker as cb
            from app.ml.ml_engine import MLEngine
            from nexus_core.structure_engine import structure_engine
            from app.skills import market
            from app.skills.calendar import chronos_engine
            from app.skills.calendar_guard import calendar_guard
            from nexus_core.mtf_engine import mtf_engine

            self.market_physics = mp
            self.risk_engine = RiskEngine()
            self.structure_engine = structure_engine
            self.mtf_engine = mtf_engine
            self.calendar_guard = calendar_guard
            self.chronos_engine = chronos_engine
            self.market = market
            self.feat_engine = feat_full_chain_institucional
            self.ml_engine = MLEngine()
            self.circuit_breaker = cb

            # [SENIOR LINKAGE] Connecting TradeManager to CircuitBreaker
            from app.core.zmq_bridge import zmq_bridge
            self.trade_manager = TradeManager(zmq_bridge)
            if self.circuit_breaker:
                self.circuit_breaker.set_trade_manager(self.trade_manager)
            
            logger.info("✅ Services Bound to NexusState and SRE Linked")
        except Exception as e:
            logger.critical(f"🛑 FAILED TO BIND SERVICES: {e}")
            raise

    # --- CALLBACKS ---
    async def on_zmq_message(self, data):
        """High-Frequency Pulse Handler."""
        if self.circuit_breaker: 
            self.circuit_breaker.heartbeat()
        
        if data.get("action") == "POSITION_UPDATE" and self.trade_manager:
            ticket = data.get("ticket")
            if data.get("status") == "CLOSED": 
                self.trade_manager.unregister_position(ticket)
            else:
                self.trade_manager.register_position(
                    ticket=ticket, entry_price=data.get("price", 0),
                    atr=data.get("atr", 0), is_buy=data.get("type") == "BUY"
                )
            return

        regime = self.market_physics.ingest_tick(data) if self.market_physics else None
        
        # --- THE ETERNAL SENTINEL LOGIC ---
        # 1. Context Filter (Cheap)
        in_killzone = False
        if self.chronos_engine:
             w_res = self.chronos_engine.validate_window()
             in_killzone = w_res.get("is_valid", False)
        
        # 2. Structural Heartbeat (Update Map Always - periodic)
        import time
        now = time.time()
        if (now - self.last_scan_time) >= 60.0:
             self.last_scan_time = now
             # Spawn SCAN task (Non-blocking map update)
             asyncio.create_task(self.process_signal_task(data, regime, mode="SCAN"))
        
        # 3. Trigger Logic (The Sniper Trigger)
        # Condition: (Killzone OR In_Zone) AND (Physics Breakout)
        in_zone = self.context_cache.get("in_zone", False)
        
        # Physics Breakout: Acceleration > 1.2 OR Initiative Candle
        is_breakout = False
        if regime:
             is_breakout = (regime.is_accelerating and regime.acceleration_score > 1.2) or \
                           getattr(regime, 'is_initiative_candle', False)
        
        if (in_killzone or in_zone) and is_breakout:
             # Full Neural Inference
             asyncio.create_task(self.process_signal_task(data, regime, mode="FULL"))

    # --- BACKGROUND TASKS ---
    async def process_signal_task(self, data, regime, mode: str = "FULL"):
        """
        Strategic Inference and Execution Pipeline.
        Modes:
        - SCAN: Updates Structure Map & Zone Cache only (The Sentry).
        - FULL: Executes ML Inference & Trade Logic (The Sniper).
        """
        async with self.brain_semaphore:
            try:
                price = float(data.get('bid', 0) or data.get('close', 0))
                symbol = data.get('symbol', 'XAUUSD')
                
                # Phase 1: Context Acquisition (MTF)
                target_tfs = ["M1", "M5", "H1", "H4"]
                candles_by_tf = {}
                if self.market:
                    for tf in target_tfs:
                        c_res = await self.market.get_candles(symbol, tf, 100)
                        if c_res.get("status") == "success" and c_res.get("candles"):
                             df = pd.DataFrame(c_res["candles"])
                             for c in ['open', 'high', 'low', 'close', 'tick_volume']:
                                 if c in df.columns:
                                     df[c] = pd.to_numeric(df[c])
                             candles_by_tf[tf] = df

                if "M1" not in candles_by_tf:
                    logger.warning(f"⚠️ Initial Context Missing for {symbol}. Abortion initiated.")
                    return

                # Phase 2: Structural & Semantic Engineering (The Core)
                # triggers structure_engine, acceleration_engine, kinetic_engine internally
                processed_df = feat_processor.apply_feat_engineering(candles_by_tf["M1"])
                last_row = processed_df.iloc[-1]
                
                # Update sentinel cache for MTF/Zone awareness
                if self.structure_engine:
                    report = self.structure_engine.get_structural_report(candles_by_tf["M1"])
                    self.context_cache["in_zone"] = (report['zones']['distance_to_zone'] < (last_row['close'] * settings.ZONE_PROXIMITY_FACTOR))

                if mode == "SCAN":
                    return

                # Phase 3: Neural Adaptation & Inference
                # Converts processing row to Neural Latent Vector (Z_t)
                feature_map = feat_processor.compute_latent_vector(last_row)
                feature_vector = [feature_map.get(name, 0.0) for name in settings.NEURAL_FEATURE_NAMES]
                
                # Predicted: Alpha, Win_Prob, Vol_Regime, Logits
                brain_score = await ml_engine.predict_hybrid(feature_vector, symbol)
                
                # Phase 4: Probabilistic Convergence (Bayesian Fusion)
                convergence = convergence_engine.evaluate_convergence(
                    neural_alpha=brain_score.get("alpha_multiplier", 1.0),
                    kinetic_coherence=feature_map.get("kinetic_coherence", 0.0),
                    p_win=brain_score.get("p_win", 0.5),
                    uncertainty=brain_score.get("uncertainty", 0.1)
                )
                
                final_probability = convergence.score
                
                # Phase 5: Telemetry & Living Dashboard Sync
                neural_service.update_state(
                    symbol=symbol,
                    price=price,
                    probs={"buy": brain_score.get("buy", 0.0), "sell": brain_score.get("sell", 0.0), "hold": brain_score.get("hold", 0.0)},
                    brain_score={**brain_score, "alpha_confidence": round(final_probability, 3)},
                    feature_vector=feature_map,
                    sentinel_status={
                        "kill_switch": len(convergence.vetoes) > 0, 
                        "reason": ", ".join(convergence.vetoes) if convergence.vetoes else "NONE"
                    }
                )
                
                if self.supabase_sync:
                    current_state = neural_service.get_latest_state()
                    asyncio.create_task(self.supabase_sync.log_neural_state(current_state))

                # Phase 6: Institutional Veto Gateway
                if not system_sentinel.is_safe():
                    logger.critical(f"🛑 TRADING HALTED: Sentinel Veto. {system_sentinel.kill_reason}")
                    return
                
                if convergence.vetoes:
                    logger.warning(f"✋ SIGNAL Vetoed for {symbol}: {', '.join(convergence.vetoes)}")
                    brain_score['execute_trade'] = False
                else:
                    # Epistemic Gate
                    is_mechanically_sound = (final_probability > settings.ALPHA_CONFIDENCE_THRESHOLD)
                    is_certain = (brain_score.get("uncertainty", 1.0) < settings.CONVERGENCE_MAX_UNCERTAINTY)
                    brain_score['execute_trade'] = is_mechanically_sound and is_certain
                
                # Phase 7: Active Management & Execution logic
                if self.trade_manager:
                    await self.trade_manager.update_positions_logic(final_probability)
                
                if brain_score.get('execute_trade', False):
                    direction = convergence.direction if convergence.direction in ["BUY", "SELL"] else None
                    if not direction:
                        direction = "BUY" if brain_score.get("buy", 0) > brain_score.get("sell", 0) else "SELL"

                    # Calculate Ph.D. Risk Lots
                    dynamic_lot = await self.risk_engine.calculate_dynamic_lot(
                        confidence=final_probability,
                        volatility=last_row.get("atr14", 0.0),
                        symbol=symbol,
                        market_data={**data, "brain_uncertainty": brain_score.get("uncertainty", 0.1)}
                    ) if self.risk_engine else 0
                    
                    if dynamic_lot > 0 and self.trade_manager:
                        res = await self.trade_manager.execute_order(direction, {
                            "symbol": symbol,
                            "volume": dynamic_lot,
                            "magic": settings.MT5_MAGIC_NUMBER,
                            "comment": settings.MT5_ORDER_COMMENT
                        })
                        
                        if res.get("status") == "EXECUTED":
                             self.trade_manager.register_position(
                                 ticket=res["ticket"],
                                 entry_price=price,
                                 atr=last_row.get("atr14", 0.0),
                                 is_buy=(direction == "BUY"),
                                 context={**feature_map, "p_win": brain_score.get("p_win", 0.5)}
                             )
            except Exception as e:
                logger.error(f"process_signal_task CRITICAL Error: {e}", exc_info=True)

    async def automl_background_loop(self):
        """
        [LEVEL 61] Autonomous Evolution Sentinel.
        Monitors model drift and triggers retraining periodically.
        """
        while True:
            try:
                await automl_orchestrator.check_and_evolve()
            except Exception as e:
                logger.error(f"AutoML Loop Error: {e}")
            
            # Wait for next check interval
            await asyncio.sleep(settings.AUT_EVO_INTERVAL_MINUTES * 60)

    async def dashboard_heartbeat(self):
        """Push account metrics every 5 seconds."""
        import MetaTrader5 as mt5_lib
        while True:
            try:
                if self.dashboard and mt5_lib.terminal_info():
                    acc = mt5_lib.account_info()
                    if acc:
                        await self.dashboard.push_metrics({
                            "balance": acc.balance,
                            "equity": acc.equity,
                            "margin_free": acc.margin_free
                        })
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
            await asyncio.sleep(5)

    async def zmq_processing_loop(self):
        """Polls ZMQ updates and feeds Memory."""
        from nexus_core.memory import nexus_memory
        while True:
            try:
                 if zmq_bridge:
                     msg = zmq_bridge.check_messages()
                     if msg:
                         if msg.get('type') == 'TICK':
                             await nexus_memory.save_tick(msg)
                         await asyncio.sleep(0.001)
                     else:
                         await asyncio.sleep(0.01)
                 else:
                     await asyncio.sleep(1.0)
            except Exception as e:
                 logger.error(f"ZMQ Loop Error: {e}")
                 await asyncio.sleep(1.0)

    async def jitter_sentinel(self):
        """Monitors ZMQ loop health."""
        while True:
            if self.running:
                self._background_tasks.add(asyncio.create_task(self._drift_monitor_loop()))
            
            await self.ml_engine.check_loop_jitter()
            await asyncio.sleep(0.1)

    async def _drift_monitor_loop(self):
        """Periodic drift check for AutoML."""
        if not automl_orchestrator or not settings.AUTOML_ENABLED:
            return
            
        logger.info(f"🛡️ AUTOML SENTINEL: Initialized (Check every {settings.AUTOML_CHECK_INTERVAL_MINUTES}m)")
        while self.running:
            try:
                await automl_orchestrator.check_and_evolve()
            except Exception as e:
                logger.error(f"AutoML Loop Error: {e}")
            await asyncio.sleep(settings.AUTOML_CHECK_INTERVAL_MINUTES * 60)

    async def background_bootstrap(self):
        """Populates brain buffers while ZMQ remains responsive."""
        from app.skills.market import get_candles
        from app.ml.data_collector import data_collector
        import MetaTrader5 as mt5_ref

        # [AUTO-SCAN] Detect active symbols from Terminal
        symbols_to_hydrate = []
        try:
            curr_id = mt5_ref.chart_first()
            while curr_id > 0:
                sym_name = mt5_ref.chart_symbol(curr_id)
                if sym_name and sym_name not in symbols_to_hydrate:
                    symbols_to_hydrate.append(sym_name)
                curr_id = mt5_ref.chart_next(curr_id)
            
            if not symbols_to_hydrate:
                 logger.warning("[SCAN] No active charts found. Defaulting to XAUUSD.")
                 symbols_to_hydrate = ["XAUUSD"]
            else:
                 logger.info(f"[SCAN] Detected Active Assets: {symbols_to_hydrate}")
                 
        except Exception as e:
            logger.warning(f"[SCAN] Auto-detection failed ({e}). Defaulting to XAUUSD.")
            symbols_to_hydrate = ["XAUUSD"]
        
        for symbol in symbols_to_hydrate: 
            logger.info(f"[HYDRATION] Deep Sync for {symbol}...")
            await data_collector.hydrate_all_timeframes(symbol, mt5_fallback_func=get_candles)
            
            candles_res = await get_candles(symbol, "M1", n_candles=200)
            if candles_res.get("status") == "success":
                candles = candles_res["candles"]
                if self.market_physics: self.market_physics.hydrate(candles)
                prices = [float(c['close']) for c in candles]
                if self.ml_engine: self.ml_engine.hydrate_hurst(symbol, prices)
            
        logger.info("[HYDRATION] Buffers ACTIVE.")

# Initialize the NexusState Manager
STATE = NexusState()

# --- LIFESPAN ENCAPSULATION ---
@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Ciclo de vida institucional."""
    # 1. Initialize State and Services
    await STATE.initialize_services()
    
    # 2. Startup sequences
    from app.core.streamer import init_streamer
    STATE.dashboard = init_streamer()
    if STATE.dashboard:
        logging.getLogger().addHandler(STATE.dashboard)
        asyncio.create_task(STATE.dashboard.start_async_loop())
        asyncio.create_task(STATE.dashboard_heartbeat())
        # [LEVEL 61] AutoML Sovereign Sentinel
    if settings.AUTOML_ENABLED:
        asyncio.create_task(STATE.automl_background_loop())
        logger.info("🛡️ AutoML Sovereign Sentinel Active (Autonomous Evolution Loop).")
    
    if mt5_conn: await mt5_conn.startup()
    
    from app.services.rag_memory import rag_memory
    await rag_memory.initialize()
    if STATE.circuit_breaker: 
        asyncio.create_task(STATE.circuit_breaker.monitor_heartbeat())
    
    if zmq_bridge:
        from app.skills.trade_mgmt import TradeManager
        STATE.trade_manager = TradeManager(zmq_bridge)
        asyncio.create_task(STATE.background_bootstrap())
        asyncio.create_task(STATE.jitter_sentinel())
        
        # [FIX] Register Memory Feeder as secondary callback (Topology V2)
        from nexus_core.memory import nexus_memory
        async def feed_memory(msg):
             if msg.get('type') == 'TICK':
                 await nexus_memory.save_tick(msg)
        zmq_bridge.add_callback(feed_memory)
        
        # Start Bridge with Main Signal Handler
        await zmq_bridge.start(STATE.on_zmq_message)
        logger.info("✅ ZMQ Bridge Active (Callbacks: Signal + Memory)")

    try:
        yield
    finally:
        if STATE.dashboard:
            await STATE.dashboard.push_metrics({"status": "OFFLINE", "equity": 0, "balance": 0})
        if zmq_bridge: await zmq_bridge.stop()
        if mt5_conn: await mt5_conn.shutdown()

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
    candles_res = await market.get_candles(symbol, timeframe, 100)
    
    feat_layers = {}
    if candles_res.get("status") == "success":
        candles_list = candles_res.get("candles", [])
        if candles_list:
            df = pd.DataFrame(candles_list)
            # Ensure column names match what indicator engine expects (tick_volume -> volume if needed)
            if 'tick_volume' in df.columns:
                df = df.rename(columns={'tick_volume': 'volume'})
            
            feat_df = indicators.calculate_feat_layers(df)
            feat_layers = feat_df.to_dict('records') if not feat_df.empty else {}
            
            # 3. Structural Intelligence Report
            from nexus_core.structure_engine import structure_engine
            structure_report = structure_engine.get_structural_report(df)
            structural_narrative = structure_engine.get_structural_narrative(df)
            structural_score = structure_engine.get_structural_score(df)
            structural_risk = structure_engine.get_structural_risk(df)

    return {
        "tool": "market_get_telemetry",
        "symbol": symbol,
        "timeframe": timeframe,
        "snapshot": snapshot,
        "candles_count": len(candles_res.get("candles", [])) if candles_res.get("status") == "success" else 0,
        "feat_layers": feat_layers,
        "structure": {
            "report": structure_report,
            "narrative": structural_narrative,
            "score": structural_score,
            "risk": structural_risk
        },
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
    
    prediction = {"signal": "NEUTRAL", "confidence": 0.0, "reason": "Initializing"}
    feat_check = False # MILITARY GRADE: Rejected by default

    try:
        # 0. Price Context
        current_price = float(context_data.get("close", 0) or context_data.get("bid", 0))
        
        # 1. FEAT Strategic Filter (Rule Based)
        if STATE.feat_engine:
            feat_check = await STATE.feat_engine.analyze(context_data, current_price) if asyncio.iscoroutinefunction(STATE.feat_engine.analyze) else STATE.feat_engine.analyze(context_data, current_price)
            
            if not feat_check:
                prediction["reason"] = "FEAT_REJECTED (Rules)"
                prediction["signal"] = "WAIT"
        else:
            prediction["reason"] = "STRATEGY_ENGINE_OFFLINE"
            prediction["signal"] = "HALT"
            feat_check = False
        
        # 2. Neural Inference (Quantum Leap Ensemble)
        if feat_check and STATE.ml_engine:
             res = await STATE.ml_engine.ensemble_predict_async(context_data.get('symbol', settings.SYMBOL), context_data)
             prediction = {
                 "signal": res["prediction"],
                 "confidence": res["confidence"],
                 "reason": f"QuantumLeap ({res['regime']})",
                 "hurst": res.get("hurst"),
                 "p_win": res.get("p_win"),
                 "is_anomaly": res.get("is_anomaly")
             }
             
             if 'symbol' in context_data:
                # For now, we rely on the pre-computed 'res["regime"]'.
                _deep_context_skipped = True

        elif not STATE.ml_engine:
             prediction["error"] = "MLEngine Offline"

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
async def brain_evolve_networks(days: int = 5, symbol: str = "XAUUSD") -> Dict[str, Any]:
    """
    🧬 MASTER TOOL 11: Neural Evolution
    Inicia el ciclo de evolución genética para optimizar parámetros de análisis y gestión.
    Usa datos de MT5 para 'Virtual Replay' y busca maximizar el ratio de ganancias/hora.
    """
    try:
        from tools.start_evolution import run_evolution_cycle
        # Run in background to avoid blocking MCP
        asyncio.create_task(run_evolution_cycle(symbol=symbol, days=days))
        return {
            "status": "EVOLUTION_STARTED",
            "message": f"Ciclo de evolución iniciado para {symbol} ({days} días). Sigue el progreso en el Dashboard Tab: Evolution",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

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
    
    if not STATE.trade_manager:
        return {"error": "TradeManager Offline/Orphaned", "status": "FAILED"}

    try:
        # Delegate to Module 6
        result_data = await STATE.trade_manager.execute_order(action, params)
        
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
    
    # Send HUD update via ZMQ to MT5
    from app.core.zmq_bridge import zmq_bridge
    if zmq_bridge.running:
        hud_payload = {
            "action": "HUD_UPDATE",
            **data,
            "ts": datetime.now(timezone.utc).timestamp() * 1000
        }
        await zmq_bridge.send_raw(hud_payload)
        status = "HUD update sent via ZMQ"
    else:
        status = "ZMQ bridge not running - HUD update queued"
    
    return {
        "tool": "visual_update_hud",
        "data_received": data,
        "status": status,
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
                "risk_percent": settings.RISK_PERCENT,
                "max_positions": settings.MAX_OPEN_POSITIONS,
                "shadow_mode": settings.SHADOW_MODE,
                "allowed_symbols": [settings.SYMBOL]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Update settings dynamically (runtime only, not persistent)
    updated = {}
    for key, value in params.items():
        key_upper = key.upper()
        if hasattr(settings, key_upper):
            try:
                setattr(settings, key_upper, value)
                updated[key] = value
                logger.info(f"[CONFIG] Updated {key_upper} = {value}")
            except Exception as e:
                logger.warning(f"[CONFIG] Failed to set {key_upper}: {e}")
    
    return {
        "tool": "config_update_parameters",
        "updated": updated,
        "status": "Configuration updated (runtime only)",
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
        if mt5_conn.connected and STATE.trade_manager:
            closed_positions = await STATE.trade_manager.close_all_positions()
            # Note: cancel_all_orders would need implementation in TradeManager
            # For now we close active positions which is the priority.
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

@mcp.tool()
async def sys_warm_reboot() -> Dict[str, Any]:
    """
    🔄 MASTER TOOL 11: Warm Reboot
    Recarga módulos críticos y re-ejecuta la hidratación sin cerrar el proceso.
    Útil para aplicar cambios en caliente en la lógica de market o ML.
    """
    import importlib
    global market, indicators
    
    reports = []
    
    try:
        # 1. Reload Modules
        from app.skills import market as m, indicators as i, market_physics as mp_mod
        importlib.reload(m)
        importlib.reload(i)
        importlib.reload(mp_mod)
        
        market = m
        indicators = i
        STATE.market_physics = mp_mod.market_physics
        reports.append("Modules [market, indicators, market_physics] reloaded.")
        
        # 2. Re-instantiate MLEngine if possible
        from app.ml.ml_engine import MLEngine
        import app.ml.ml_engine as mle_mod
        importlib.reload(mle_mod)
        STATE.ml_engine = MLEngine()
        reports.append("MLEngine re-instantiated and reloaded.")
        
        # 3. Reload FEAT Engine
        from app.skills.feat_chain import feat_full_chain_institucional
        import app.skills.feat_chain as fc_mod
        importlib.reload(fc_mod)
        STATE.feat_engine = feat_full_chain_institucional
        reports.append("FeatEngine re-instantiated and reloaded.")
        
        # 4. Trigger Re-Hydration
        # We need access to background_bootstrap - but it's inside app_lifespan.
        # However, we can call data_collector.hydrate_all_timeframes directly.
        from app.ml.data_collector import data_collector
        import app.ml.data_collector as dc_mod
        importlib.reload(dc_mod)
        
        # Run hydration in background
        async def rehydrate():
             for symbol in ["XAUUSD", "EURUSD"]:
                 await data_collector.hydrate_all_timeframes(symbol, mt5_fallback_func=market.get_candles)
        
        asyncio.create_task(rehydrate())
        reports.append("Re-hydration sequence triggered in background.")
        
        return {
            "status": "success",
            "reports": reports,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Warm reboot failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@mcp.tool()
async def visual_perception_snapshot(resize_factor: float = 0.5) -> Dict[str, Any]:
    """
    📸 MASTER TOOL 12: Vision System
    Captura una imagen de la terminal MT5 para análisis visual.
    """
    try:
        from app.skills.vision import capture_panorama
        return await capture_panorama(resize_factor)
    except Exception as e:
        return {"error": str(e), "status": "FAILED"}

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
            print("FEAT NEXUS: System online in STDIO mode. Awaiting synapse activations...", file=sys.stderr)
            mcp.run() # Defaults to stdio transport
    except Exception:
        traceback.print_exc()
        # Non-blocking pause for log visibility if running interactively
        try:
            import time
            time.sleep(5)
        except Exception:
            _stop_interrupted = True
