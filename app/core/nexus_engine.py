import asyncio
import logging
import time
import os
import pandas as pd
import numpy as np
import MetaTrader5 as mt5_ref
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.config import settings
from app.core.zmq_bridge import zmq_bridge
from app.ml.feat_processor import feat_processor
from nexus_core.features import feat_features
from app.services.circuit_breaker import circuit_breaker
from app.services.state_exporter import state_exporter
from nexus_core.microstructure.scanner import micro_scanner
from nexus_core.neural_health import neural_health

logger = logging.getLogger("nexus.engine")

class NexusEngine:
    """
    [LEVEL 64] THE IMMORTAL CORE
    ============================
    Independent trading engine responsible for:
    - MT5 Data Hydration & Tick Ingestion
    - Neural Inference (TCN-BiLSTM)
    - Execution & Risk Management
    - Sentinel Monitoring
    """
    def __init__(self):
        self.running = True
        self.last_scan_time = 0.0
        self.context_cache = {"in_zone": False, "session": "OFF"}
        self.brain_semaphore = asyncio.Semaphore(20)
        self._background_tasks = set()
        
        # Service Placeholders
        self.ml_engine = None
        self.risk_engine = None
        self.trade_manager = None
        self.structure_engine = None
        self.market_physics = None
        self.mtf_engine = None
        self.chronos_engine = None
        self.strategy_engine = None
        
        # Sentinels
        self.jitter_sentinel = None
        self.drift_sentinel = None

    async def initialize(self):
        """Sequential bootstrap of warfare assets."""
        logger.info("⚔️ NexusEngine: Initializing Immortal Core...")
        try:
            from app.skills.market_physics import market_physics as mp
            from app.services.risk import risk_engine as RiskEngine
            from app.skills.trade_mgmt import TradeManager
            from app.ml.ml_engine import MLEngine
            from nexus_core.structure_engine import structure_engine
            from app.skills import market
            from app.skills.calendar import chronos_engine
            from nexus_core.mtf_engine import mtf_engine
            from app.sentinels.jitter import JitterSentinel
            from app.sentinels.drift import DriftSentinel
            from app.ml.automl.orchestrator import automl_orchestrator
            from app.core.mt5_conn import mt5_conn

            # 1. MT5 Connection (THE BRIDGE)
            logger.info("🔌 Connecting to MetaTrader 5...")
            if not await mt5_conn.startup():
                logger.error("❌ Failed to connect to MT5. Check terminal path and credentials.")
                # We don't raise here to allow the engine to run in shadow/mock mode if needed, 
                # but for production parity, we log clearly.

            self.market_physics = mp
            self.risk_engine = RiskEngine
            self.structure_engine = structure_engine
            self.ml_engine = MLEngine()
            self.mtf_engine = mtf_engine
            self.chronos_engine = chronos_engine
            self.market = market
            self.trade_manager = TradeManager(zmq_bridge)
            
            # 2. Strategy Engine (THE TACTICIAN)
            from nexus_core.strategy_engine import StrategyEngine
            self.strategy_engine = StrategyEngine(self.risk_engine)
            
            # Link Circuit Breaker
            circuit_breaker.set_trade_manager(self.trade_manager)
            
            # Initialize Sentinels
            self.jitter_sentinel = JitterSentinel(self.ml_engine)
            self.drift_sentinel = DriftSentinel(automl_orchestrator)
            
            # Initialize ZMQ Bridge with Callback
            await zmq_bridge.start(callback=self.on_market_update)
            
            # Launch Background Loops
            # self._spawn_task(self.zmq_processing_loop()) # DEPRECATED: Replaced by ZMQ Callback
            self._spawn_task(self.jitter_sentinel.run_loop())
            self._spawn_task(self.drift_sentinel.run_loop())
            self._spawn_task(self.dashboard_heartbeat())
            self._spawn_task(self.state_export_loop())
            self._spawn_task(self.command_processor_loop())
            
            logger.info("✅ NexusEngine: All Systems Operational.")
        except Exception as e:
            logger.critical(f"🔥 NexusEngine: Initialization Failure: {e}")
            raise

    def _spawn_task(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def zmq_processing_loop(self):
        """Deprecated: Logic moved to on_market_update callback."""
        logger.info("📡 NexusEngine: ZMQ Bridge Callback Mode Active.")
        # Minimal keepalive if needed, but logic is now push-based
        while self.running:
             await asyncio.sleep(1)

    async def on_market_update(self, data):
        """Core signal handling logic."""
        # 1. Heartbeat
        circuit_breaker.heartbeat()
        if self.jitter_sentinel: self.jitter_sentinel.heartbeat()

        # 2. Position Updates (from MT5)
        if data.get("action") == "POSITION_UPDATE":
            self._handle_position_update(data)
            return

        # 3. Physics Ingestion
        regime = self.market_physics.ingest_tick(data) if self.market_physics else None
        
        # 4. Trigger Analysis
        in_killzone = self._check_killzone()
        in_zone = self.context_cache.get("in_zone", False)
        
        # Performance Scan (Periodic)
        now = time.time()
        if (now - self.last_scan_time) >= 60.0:
            self.last_scan_time = now
            self._spawn_task(self.execute_vision_protocol(data, regime, mode="SCAN"))
        
        # Sniper Trigger
        is_breakout = self._is_physics_breakout(regime)
        if (in_killzone or in_zone) and is_breakout:
            self._spawn_task(self.execute_vision_protocol(data, regime, mode="FULL"))

    def _handle_position_update(self, data):
        ticket = data.get("ticket")
        if data.get("status") == "CLOSED":
            self.trade_manager.unregister_position(ticket)
        else:
            self.trade_manager.register_position(
                ticket=ticket, 
                entry_price=data.get("price", 0),
                atr=data.get("atr", 0), 
                is_buy=data.get("type") == "BUY"
            )

    def _check_killzone(self) -> bool:
        if self.chronos_engine:
            return self.chronos_engine.validate_window().get("is_valid", False)
        return False

    def _is_physics_breakout(self, regime) -> bool:
        if not regime: return False
        return (regime.is_accelerating and regime.acceleration_score > 1.2) or \
               getattr(regime, 'is_initiative_candle', False)

    async def execute_vision_protocol(self, data, regime, mode: str = "FULL"):
        """High-Level Inference & Execution."""
        async with self.brain_semaphore:
            try:
                symbol = data.get('symbol', 'XAUUSD')
                price = float(data.get('bid', 0) or data.get('close', 0))
                
                # Context Awareness (MTF - Fractal Expansion)
                target_tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
                candles = {}
                for tf in target_tfs:
                    c_res = await self.market.get_candles(symbol, tf, 100)
                    if c_res.get("status") == "success":
                         df = pd.DataFrame(c_res["candles"])
                         # Clean/Numeric Conversion
                         for c in ['open', 'high', 'low', 'close', 'tick_volume']:
                             if c in df.columns: df[c] = pd.to_numeric(df[c])
                         candles[tf] = df

                if "M1" not in candles: return

                # Structural Mapping
                processed_df = feat_processor.process_dataframe(candles["M1"])
                last_row = processed_df.iloc[-1]
                
                if self.structure_engine:
                    report = self.structure_engine.get_structural_report(candles["M1"])
                    self.context_cache["in_zone"] = (report['zones']['distance_to_zone'] < (price * settings.ZONE_PROXIMITY_FACTOR))
                    
                # Fractal Coherence Logic
                alignment_map = {}
                coherence_score = 0.5
                if self.mtf_engine:
                    # Diagnose across all timeframes
                    fractal_report = self.mtf_engine.diagnose_market_fractals(candles)
                    alignment_map = fractal_report.get("alignment_map", {})
                    coherence_score = fractal_report.get("coherence", 0.5)
                    self.context_cache["alignment_map"] = alignment_map
                    self.context_cache["fractal_coherence"] = coherence_score

                # 4. Temporal Physics (Inter-Temporal Synthesis)
                temporal_physics = feat_features.extract_temporal_physics(candles)
                
                if mode == "SCAN": return

                # Neural Logic
                latent_vector = feat_processor.compute_latent_vector(last_row)
                feature_names = settings.NEURAL_FEATURE_NAMES
                input_vec = [latent_vector.get(name, 0.0) for name in feature_names]
                
                prediction = await self.ml_engine.predict_hybrid(input_vec, symbol)
                
                # Strategy Convergence (Institutional Logic)
                from nexus_core.convergence_engine import convergence_engine
                cv = convergence_engine.evaluate_convergence(
                    neural_alpha=prediction.get("alpha_multiplier", 1.0),
                    kinetic_coherence=latent_vector.get("kinetic_coherence", 0.0),
                    p_win=prediction.get("p_win", 0.5),
                    uncertainty=prediction.get("uncertainty", 0.1)
                )

                # 5. Strategic Intelligence (AI Decision)
                # This uses the PolicyNetwork (Strategic Cortex)
                if self.strategy_engine:
                    legs = self.strategy_engine.analyze_strategic_intent(
                        market_price=price,
                        neural_probs={
                            "scalp": prediction.get("buy", 0.0) if cv.direction == "BUY" else prediction.get("sell", 0.0),
                            "day": prediction.get("p_win", 0.5) * 0.8, # Approximations for now
                            "swing": prediction.get("p_win", 0.5) * 0.6
                        },
                        macro_context={
                            "direction": cv.direction,
                            "alignment_map": self.context_cache.get("alignment_map", {})
                        },
                        titanium_level=self.context_cache["in_zone"],
                        microstructure_state={
                            "ofi_z_score": last_row.get("ofi_z", 0.0),
                            "entropy_score": last_row.get("entropy", 0.5),
                            "fractal_coherence": coherence_score
                        },
                        physics_metrics={
                            "feat_force": latent_vector.get("feat_composite", 50.0),
                            "rvol": last_row.get("rvol", 1.0)
                        },
                        temporal_physics=temporal_physics
                    )
                    
                    # 6. Execution decision
                    if cv.score > settings.ALPHA_CONFIDENCE_THRESHOLD and not cv.vetoes and legs:
                        for leg in legs:
                            await self._execute_strategic_leg(symbol, leg, price, last_row, prediction, latent_vector)

            except Exception as e:
                logger.error(f"Vision Protocol Error: {e}", exc_info=True)

    async def _execute_strategic_leg(self, symbol, leg, price, row, prediction, features):
        """Executes a specific strategic leg (Scalp, Day, or Swing)."""
        if not leg or leg.volume <= 0: return

        res = await self.trade_manager.execute_order(leg.direction, {
            "symbol": symbol,
            "volume": leg.volume,
            "magic": settings.MT5_MAGIC_NUMBER,
            "comment": f"NEXUS_{leg.strategy_type.value}",
            "strategy": leg.strategy_type.value,
            "intent": leg.intent
        })
        
        if res.get("status") == "EXECUTED":
            self.trade_manager.register_position(
                ticket=res["ticket"],
                entry_price=price,
                atr=row.get("atr14", 0.0),
                is_buy=(leg.direction == "BUY"),
                context={
                    **features, 
                    "p_win": prediction.get("p_win", 0.5),
                    "strategy": leg.strategy_type.value,
                    "intent": leg.intent
                }
            )
            logger.info(f"🚀 {leg.strategy_type.value} EXECUTED: {leg.direction} {leg.volume} lot on {symbol} | Intent: {leg.intent}")

    async def _execute_trade(self, symbol, direction, price, row, prediction, features):
        # Legacy placeholder - replaced by _execute_strategic_leg
        pass

    async def dashboard_heartbeat(self):
        while self.running:
            try:
                if mt5_ref.terminal_info():
                    acc = mt5_ref.account_info()
                    if acc:
                        # StateExporter picks this up from State
                        pass
            except Exception: pass
            await asyncio.sleep(5)

    async def state_export_loop(self):
        while self.running:
            try:
                acc_info = await self.market.get_account_metrics() if self.market else {}
                # Use self.ml_engine status instead of neural_service
                latest_neural = self.ml_engine.get_status() if self.ml_engine else {}
                
                # Health & Resilience Metrics
                health_metrics = neural_health.get_health_metrics()
                micro_state = micro_scanner.get_dict()
                
                data = {
                    "account": {
                        "balance": float(acc_info.get("balance", 0)),
                        "equity": float(acc_info.get("equity", 0)),
                        "pnl": float(acc_info.get("profit", 0)),
                        "margin_level": float(acc_info.get("margin_level", 0))
                    },
                    "positions_count": len(self.trade_manager.active_positions) if self.trade_manager else 0,
                    "symbol": "XAUUSD",
                    "risk_factor": getattr(settings, "RISK_FACTOR", 1.0),
                    "circuit_breaker": {
                        "status": "CLOSED" if circuit_breaker.is_ok() else "TRIPPED",
                        "latency": circuit_breaker.get_last_latency()
                    },
                    "microstructure": micro_state,
                    "health": health_metrics,
                    "alignment_map": self.context_cache.get("alignment_map", {}),
                    "fractal_coherence": self.context_cache.get("fractal_coherence", 0.5),
                    **latest_neural
                }
                await state_exporter.export(data)
            except Exception as e:
                logger.error(f"State Export Error: {e}")
            await asyncio.sleep(1.0)

    async def command_processor_loop(self):
        cmd_file = "data/app_commands.json"
        while self.running:
            if os.path.exists(cmd_file):
                try:
                    with open(cmd_file, 'r') as f: commands = json.load(f)
                    os.remove(cmd_file)
                    for cmd in commands:
                        act = cmd.get("action")
                        p = cmd.get("params", {})
                        if act == "SET_RISK_FACTOR": settings.RISK_FACTOR = p.get("value", 1.0)
                        elif act == "PANIC_CLOSE_ALL" and self.trade_manager: await self.trade_manager.close_all_positions()
                        elif act == "RELOAD_MODELS" and self.ml_engine: await self.ml_engine.reload_weights()
                except Exception as e: logger.error(f"Command Error: {e}")
            await asyncio.sleep(0.5)

    async def shutdown(self):
        logger.info("👋 NexusEngine: Initiating Safe Shutdown...")
        self.running = False
        for task in self._background_tasks:
            task.cancel()
        if self.trade_manager:
            await self.trade_manager.cleanup()
        logger.info("✅ NexusEngine: Shutdown Complete.")

# Single Instance for the Process
engine = NexusEngine()
