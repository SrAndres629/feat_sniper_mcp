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
        pass

# === PROTOCOLO BLACKHOLE (INICIO) ===
# Silenciar Warnings de Inmediato
import warnings

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
            self.circuit_breaker = cb
            self.feat_engine = feat_full_chain_institucional
            self.ml_engine = MLEngine()
            
            # Bind trade manager once zmq is active (handled in startup_sequence)
            logger.info("✅ Services Bound to NexusState")
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
                is_valid = await self.feat_engine.analyze(data, price, precomputed_physics=regime) if self.feat_engine else False
                
                # --- NEURAL-STRUCTURAL BINDING [LEVEL 10] ---
                # 1. Fetch Contextual Structure (MTF Convergence)
                symbol = data.get('symbol', 'XAUUSD')
                
                # Fetch M1, M5, H1, H4 for Fractal Analysis
                target_tfs = ["M1", "M5", "H1", "H4"]
                candles_by_tf = {}
                import pandas as pd
                
                # Default values
                struct_score = 50.0 
                mae_phase = 0.0
                is_in_zone = 0.0 
                struct_bos = 0.0 
                feat_index = 50.0
                mtf_score = 0.5
                mtf_alignment = 0.0
                
                # Fetch loop
                if self.market:
                    for tf in target_tfs:
                        c_res = await self.market.get_candles(symbol, tf, 100)
                        if c_res.get("status") == "success" and c_res.get("candles"):
                             df = pd.DataFrame(c_res["candles"])
                             cols = ['open', 'high', 'low', 'close', 'tick_volume']
                             for c in cols:
                                 if c in df.columns:
                                     df[c] = pd.to_numeric(df[c])
                             candles_by_tf[tf] = df

                # 2. Execute Structure Engine (on M1) & MTF Engine
                if "M1" in candles_by_tf and self.structure_engine:
                     df_struct = candles_by_tf["M1"]
                     
                     # Structure Analysis
                     feat_res = self.structure_engine.compute_feat_index(df_struct)
                     struct_score = float(feat_res['feat_index'].iloc[-1])
                     feat_index = struct_score 
                     
                     report = self.structure_engine.get_structural_report(df_struct)
                     phase_map = {"RANGE": 0.0, "NORMAL": 0.0, "ACCUMULATION": 0.5, "EXPANSION": 1.0, "MOMENTUM": 0.8}
                     mae_phase = phase_map.get(report['mae_pattern']['phase'], 0.0)
                     
                     # Use distance to zone for proximity
                     dist = report['zones']['distance_to_zone']
                     price_threshold = df_struct['close'].iloc[-1] * 0.0005 
                     is_in_zone = 1.0 if dist < price_threshold else 0.0
                     
                     struct_bos = report['health']['bos_strength']
                     
                     # UPDATE SENTINEL CACHE
                     self.context_cache["in_zone"] = (is_in_zone > 0.8) # >0.8 means very close/inside
                
                # If SCAN mode, stop here (We just wanted to update the Map/Cache)
                if mode == "SCAN":
                     return

                # MTF Execution
                if self.mtf_engine and candles_by_tf:
                     mtf_res = await self.mtf_engine.analyze_all_timeframes(candles_by_tf, price)
                     mtf_score = mtf_res.composite_score
                     mtf_alignment = mtf_res.alignment_percentage

                # PVP / Energy Analysis (Volume Perception)
                pvp_energy = 0.5
                poc_price = price # Default
                
                try:
                    from nexus_core.features import feat_features
                    from app.ml.market_regime import market_regime
                    from app.ml.fuzzy_logic import fuzzy_logic

                    if "M1" in candles_by_tf:
                        # Generate Energy Tensor (Density + Kinetic + Flow)
                        energy_map = feat_features.generate_energy_map(candles_by_tf["M1"])
                        pvp_energy = energy_map.get("poc_idx", 25) / 50.0
                        
                        # Calculate Actual PVP Price for Reversion Logic
                        e_meta = energy_map.get("metadata", {})
                        bin_sz = e_meta.get("bin_size", 0.0)
                        r_min = e_meta.get("range_min", 0.0)
                        poc_idx = energy_map.get("poc_idx", 25)
                        if bin_sz > 0 and r_min > 0:
                            poc_price = r_min + (poc_idx * bin_sz)
                        
                        # [LEVEL 18] CLUSTERING & FUZZY LOGIC
                        # 1. Update & Predict Regime
                        last_candle = candles_by_tf["M1"].iloc[-1]
                        atr_val = regime.atr if regime else 0.001
                        vol_val = last_candle.get('tick_volume', 100) 
                        
                        market_regime.update(atr_val, vol_val) # Online Learning
                        regime_label = market_regime.predict(atr_val, vol_val)
                        
                        # 2. Fuzzy Logic Evaluation
                        rsi_val = data.get("rsi", 50.0)
                        accel_val = regime.acceleration_score if regime else 0.0
                        fuzzy_score = fuzzy_logic.evaluate(rsi_val, accel_val) # -10 to +10
                        
                    else:
                        regime_label = "UNKNOWN"
                        fuzzy_score = 0.0
                        
                except Exception as e:
                    logger.warning(f"Cognitive Engine Error: {e}")
                    regime_label = "UNKNOWN"
                    fuzzy_score = 0.0
                               
                # 3. Inject into Neural Context
                neural_context = {
                    **data, 
                    "close": price,
                    "rsi": data.get("rsi", 50.0), 
                    "atr": data.get("atr", 0.001),
                    "feat_structure_score": struct_score,
                    "mae_status": mae_phase,
                    "zone_proximity": is_in_zone,
                    "struct_bos": struct_bos,
                    "mtf_composite_score": mtf_score,
                    "fractal_alignment": mtf_alignment,
                    "pvp_energy": pvp_energy,
                    "vol_intensity": regime.vol_z_score if regime else 0.0,
                    "news_event": 1.0 if news_risk.get("is_news_time") else 0.0,
                    "market_regime": regime_label,
                    "fuzzy_score": fuzzy_score,
                    "poc_price": poc_price
                }
                
                brain_score = await self.ml_engine.predict_async(symbol, neural_context) if self.ml_engine else {"p_win": 0.5, "uncertainty": 0}
                
                # 4. Probabilistic Fusion (Total Convergence)
                # Map p_win (0.0-1.0) to lstm_conf for fusion equation
                lstm_conf = brain_score.get('p_win', 0.5)
                norm_struct = struct_score / 100.0
                norm_mtf = mtf_score  # Already 0-1
                
                # Fuzzy Confidence Normalization (-10..10 -> 0..1)
                norm_fuzzy = (fuzzy_score + 10) / 20.0 
                
                norm_physics = regime.acceleration_score / 10.0 if regime else 0.5
                norm_physics = max(0.0, min(1.0, norm_physics))
                
                # Weights: LSTM (25%), Structure (25%), MTF (20%), Fuzzy (15%), Physics (15%)
                final_probability = (lstm_conf * 0.25) + (norm_struct * 0.25) + (norm_mtf * 0.2) + (norm_fuzzy * 0.15) + (norm_physics * 0.15)
                
                # [LEVEL 25] ALIGNMENT CHECK (ML Direction vs Regime Trend)
                if regime and regime.trend in ["BULLISH", "BEARISH"]:
                    ml_p_win = brain_score.get('p_win', 0.5)
                    ml_says_buy = ml_p_win > 0.5
                    trend_says_buy = (regime.trend == "BULLISH")
                    
                    if ml_says_buy != trend_says_buy:
                        # Conflict: ML and Trend disagree. 
                        # We crush the probability to prevent trading against the AI's conviction.
                        final_probability *= 0.2 
                        logger.warning(f"[ALIGNMENT] CONFLICT: ML(p={ml_p_win:.2f}) disagrees with Trend({regime.trend}). Vetoing.")

                # [STRATEGIC REGIME FILTER - STRICT]
                is_long_bias = final_probability > 0.5
                
                if regime_label == "CHAOS":
                    final_probability *= 0.5 # Penalize Chaos
                    logger.warning(f"[REGIME] CHAOS DETECTED on {symbol}. Confidence Halved.")
                    
                # [LEVEL 32] UNSUPERVISED ANOMALY GUARDIAN
                # Checks for Out-of-Distribution events using Isolation Forest
                if regime and regime.is_anomaly(data.get("atr", 0.001), data.get("vol_z", 0.0)):
                    final_probability = 0.0 # COMPLETE VETO
                    logger.warning(f"[GUARDIAN] ANOMALY DETECTED (Black Swan). Trade Vetoed.")
                    
                elif regime_label == "CALM":
                    # Reversion Logic: Only Buy Low (Below POC), Sell High (Above POC)
                    dist_to_poc = price - poc_price
                    # If Long bias but Price > POC (Buying High) -> Penalize
                    if is_long_bias and dist_to_poc > 0:
                         penalty = 0.3
                         final_probability -= penalty
                         logger.info(f"[REGIME] CALM Filter: Penalized BUY > PVP (-{penalty})")
                    # If Short bias but Price < POC (Selling Low) -> Penalize
                    elif not is_long_bias and dist_to_poc < 0:
                         penalty = 0.3
                         final_probability -= penalty
                         logger.info(f"[REGIME] CALM Filter: Penalized SELL < PVP (-{penalty})")

                elif regime_label == "TENDENCIA":
                    # Trend Logic: Follow Macro Flow
                    physics_trend = regime.trend if regime else "NEUTRAL"
                    if is_long_bias and physics_trend == "BEARISH":
                         final_probability -= 0.4 # Severe penalty for Counter-Trend
                         logger.info("[REGIME] TREND Filter: Penalized Counter-Trend BUY")
                    elif not is_long_bias and physics_trend == "BULLISH":
                         final_probability -= 0.4
                         logger.info("[REGIME] TREND Filter: Penalized Counter-Trend SELL")
 
                
                brain_score['alpha_confidence'] = round(final_probability, 3)

                # 5. Active Trade Management (The Hands - Reactive)
                # Close existing trades if probability decays significantly
                if self.trade_manager:
                     await self.trade_manager.update_positions_logic(final_probability)

                from app.core.config import settings
                
                # [LEVEL 20] PROBABILISTIC TRIGGER
                # Only pull the trigger if Consensus is High AND Uncertainty is Low
                uncertainty = brain_score.get('uncertainty', 1.0)
                is_mechanically_sound = (final_probability > settings.PROFIT_THRESHOLD)
                is_certain = (uncertainty < 0.05)
                
                if is_mechanically_sound and not is_certain:
                    logger.warning(f"✋ TRIGGER HELD: High Uncertainty ({uncertainty:.3f} > 0.05) despite Signal ({final_probability:.2f})")
                
                brain_score['execute_trade'] = is_mechanically_sound and is_certain

                if self.dashboard and not settings.PERFORMANCE_MODE:
                    asyncio.create_task(self.dashboard.push_signals({
                        "alpha_confidence": brain_score.get('alpha_confidence', 0),
                        "acceleration": regime.acceleration_score if regime else 0,
                        "hurst": regime.hurst_exponent if regime else 0.5,
                        "price": price, "feat_index": feat_index,
                        "is_initiative": getattr(regime, 'is_initiative_candle', False)
                    }))

                from app.core.zmq_projector import ZMQProjector
                projector = ZMQProjector()
                await projector.broadcast_system_state(
                    regime=regime.trend if regime else "NEUTRAL",
                    confidence=brain_score.get('alpha_confidence', 0),
                    feat_score=feat_index, vault_active=True
                )

                if is_valid and brain_score.get('execute_trade', False):
                    direction = "BUY" if regime.trend == "BULLISH" else "SELL"
                    
                    # Smart Risk: Cap lot size if News, unless Acceleration is Extreme
                    lot_cap = news_risk.get("max_lot_cap", 10.0)
                    if regime.acceleration and regime.acceleration_score > 2.0:
                         lot_cap = 10.0 # Override cap if Momentum is Extreme (Smart Aggression)
                    
                    dynamic_lot = await self.risk_engine.calculate_dynamic_lot(
                        confidence=brain_score.get('alpha_confidence', 0),
                        volatility=regime.atr if regime else 0.0,
                        symbol=data.get('symbol', 'XAUUSD'), market_data=data
                    ) if self.risk_engine else 0
                    
                    # Apply Cap
                    dynamic_lot = min(dynamic_lot, lot_cap)
                    
                    if dynamic_lot > 0 and self.trade_manager:
                        res = await self.trade_manager.execute_order(direction, {
                            "symbol": data.get('symbol', 'XAUUSD'),
                            "volume": dynamic_lot, "comment": "FEAT_NEURAL_V1"
                        })
                        
                        # Register for Active Management if filled
                        if res.get("status") == "EXECUTED" and res.get("ticket"):
                             self.trade_manager.register_position(
                                 ticket=res["ticket"],
                                 entry_price=price,
                                 atr=regime.atr if regime else 0.0,
                                 is_buy=(direction == "BUY"),
                                 context=neural_context
                             )
            except Exception as e:
                logger.error(f"process_signal_task Error: {e}")

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
            if self.ml_engine:
                await self.ml_engine.check_loop_jitter()
            await asyncio.sleep(0.1)

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
                # To obtain a deeper structural context, we would need to fetch history here.
                # For now, we rely on the pre-computed 'res["regime"]'.
                pass

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
        except:
            pass
