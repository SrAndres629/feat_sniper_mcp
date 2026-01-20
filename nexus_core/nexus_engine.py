"""
FEAT SNIPER: NEXUS ENGINE (SOUL OF THE MACHINE)
================================================
The main orchestration loop that connects:
- Fractal Diagnosis (Market State)
- Spectral Physics (Wavelet + PVP)
- Neural Brain (Probability)
- Strategy Engine (Tactics)
- Money Manager (Capital)
- Executor (JIT Order Dispatch)

This is the "Heartbeat" of the autonomous trading system.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

# Core Imports
from nexus_core.money_management import risk_officer
from nexus_core.strategy_engine import StrategyEngine, TradeLeg, StrategyMode
from nexus_core.neural_health import neural_health

# Diagnosis Imports
from tools.fractal_diagnosis import diagnose_market_fractals
from nexus_core.kinetic_engine import KineticEngine
from nexus_core.fundamental_engine import FundamentalEngine
from nexus_core.fundamental_engine.risk_modulator import DEFCON

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("NexusEngine")

class EngineState(Enum):
    IDLE = "IDLE"
    SENSING = "SENSING"
    PERCEIVING = "PERCEIVING"
    DECIDING = "DECIDING"
    STRATEGIZING = "STRATEGIZING"
    EXECUTING = "EXECUTING"
    MANAGING = "MANAGING"

@dataclass
class MarketSnapshot:
    price: float
    fractal_coherence: float
    dominant_bias: str
    titanium_floor: bool
    neural_probs: dict
    recommendation: str
    microstructure: dict = None # Added for Strategic Cortex

# Microstructure Imports
from nexus_core.microstructure import micro_scanner
from app.core.mt5_conn.tick_listener import tick_listener

class NexusEngine:
    """
    The Autonomous Trading Brain.
    Implements the Sense -> Perceive -> Decide -> Strategize -> Execute -> Manage loop.
    """
    
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.state = EngineState.IDLE
        self.strategy_engine = StrategyEngine(risk_officer)
        self.kinetic_engine = KineticEngine()
        
        # MACRO SENTINEL: News awareness engine
        # Use real ForexFactory scraper in live mode, mock in demo
        calendar_provider = "mock" if demo_mode else "forexfactory"
        self.fundamental_engine = FundamentalEngine(calendar_provider=calendar_provider)
        self.current_macro_status = None  # Track for dashboard
        
        self.active_trades: List[TradeLeg] = []
        
        # Coherence Gate (From Fractal Diagnosis)
        self.min_coherence_for_twin = 0.75
        self.min_coherence_for_trade = 0.50
        
        logger.info("🚀 NEXUS ENGINE INITIALIZED")
        logger.info(f"   Mode: {'DEMO' if demo_mode else 'LIVE'}")
        logger.info(f"   Strategy: Twin Sniper Evolution")
        logger.info(f"   Account Phase: {risk_officer.phase_name}")
        logger.info(f"   📡 Macro Sentinel: {calendar_provider.upper()} mode")
        
    def sense(self) -> dict:
        """
        PHASE 1: SENSE
        Read market data and calculate Fractal Coherence.
        """
        self.state = EngineState.SENSING
        
        # Get Fractal Diagnosis (Multi-Timeframe Alignment)
        # In PROD, this fetches real data from the exchange
        fractal_result = diagnose_market_fractals(mock_mode=self.demo_mode)
        
        logger.info(f"🌀 Fractal Coherence: {fractal_result['coherence_score']*100:.1f}% | Bias: {fractal_result['dominant_bias']}")
        
        return fractal_result
    
    async def perceive(self, fractal_data: dict) -> MarketSnapshot:
        """
        PHASE 2: PERCEIVE
        Calculate Titanium Floor (PVP + EMA + Wavelet confluence).
        """
        self.state = EngineState.PERCEIVING
        
        # 1. Get REAL Microstructure State (via Scanner / Ticker)
        # Using Zero-Lag Live Scan
        micro_state_obj = micro_scanner.live_scan()
        
        # Fallback for display if buffer still warming up
        if micro_state_obj.entropy_score == 0.5 and self.demo_mode:
             micro_state_obj_dict = {
                "entropy_score": 0.35 if fractal_data['coherence_score'] > 0.6 else 0.75,
                "ofi_z_score": 1.5 if fractal_data['dominant_bias'] == 'BULLISH' else -0.5,
                "hurst": 0.65
             }
        else:
             micro_state_obj_dict = micro_scanner.get_dict()

        coherence = fractal_data['coherence_score']
        
        # 2. Physics / Kinetic Computation (Fuerza Real)
        df_dummy = await self._get_latest_ohlc() 
        kinetic_metrics = self.kinetic_engine.compute_kinetic_state(df_dummy)
        
        # Titanium Floor Detection 
        titanium = kinetic_metrics.get("absorption_state", 0.0) >= 2.0 # MONITORING or CONFIRMED
        
        # Mock Neural Probabilities (Mapping from fractal diagnosis)
        if fractal_data['dominant_bias'] == 'BULLISH':
            neural_probs = {'scalp': 0.85, 'day': 0.60, 'swing': 0.40}
        elif fractal_data['dominant_bias'] == 'BEARISH':
            neural_probs = {'scalp': 0.80, 'day': 0.55, 'swing': 0.30}
        else:
            neural_probs = {'scalp': 0.50, 'day': 0.40, 'swing': 0.20}
            
        snapshot = MarketSnapshot(
            price=df_dummy['close'].iloc[-1],
            fractal_coherence=coherence,
            dominant_bias=fractal_data['dominant_bias'],
            titanium_floor=titanium,
            neural_probs=neural_probs,
            recommendation=fractal_data['recommendation'],
            microstructure=micro_state_obj_dict
        )
        
        # Store kinetic metrics for strategy
        snapshot.physics_data = kinetic_metrics
        
        if titanium:
            logger.info(f"🔥 TITANIUM FLOOR DETECTED | Direction: {snapshot.dominant_bias}")
        
        return snapshot
    
    def decide(self, snapshot: MarketSnapshot) -> bool:
        """
        PHASE 3: DECIDE
        Should we trade? Gate based on Coherence, Titanium and Noise.
        """
        self.state = EngineState.DECIDING
        
        # Gate 0: MACRO KILL SWITCH (News/DEFCON Check) - HIGHEST PRIORITY
        macro_status = self.fundamental_engine.check_event_proximity(currencies=["USD", "XAU"])
        self.current_macro_status = macro_status  # Store for dashboard
        
        if macro_status["kill_switch"]:
            next_event = macro_status.get("next_event")
            event_name = next_event.event_name if next_event else "Unknown Event"
            minutes = macro_status.get("minutes_until", 0)
            logger.warning(f"🚨 DEFCON {macro_status['defcon'].name}: TRADING FROZEN")
            logger.warning(f"   📅 Event: {event_name} in {minutes:.0f} minutes")
            logger.warning(f"   ⛔ NO NEW TRADES - Kill Switch Active")
            return False
        
        # Log DEFCON status even when not frozen
        if macro_status["defcon"] != DEFCON.DEFCON_5:
            logger.info(f"📡 DEFCON {macro_status['defcon'].name} | Position Mult: {macro_status['position_multiplier']:.1%}")
        
        # Gate 1: Entropy Check (Noise Filter)
        # If entropy is too high, market is "drunk" - avoid trading.
        if snapshot.microstructure.get('entropy_score', 0.5) > 0.8:
            logger.warning(f"❌ TRADE BLOCKED: Market is DRUNK (High Entropy: {snapshot.microstructure['entropy_score']:.2f})")
            return False
            
        # Gate 2: Minimum Coherence
        if snapshot.fractal_coherence < self.min_coherence_for_trade:
            logger.warning(f"❌ TRADE BLOCKED: Low Coherence ({snapshot.fractal_coherence*100:.1f}% < 50%)")
            return False

        
        # Gate 3: Titanium Floor Required for Entry
        if not snapshot.titanium_floor:
            logger.warning("❌ TRADE BLOCKED: No Titanium Floor detected.")
            return False
        
        # Gate 4: Neutral Bias = No Trade
        if snapshot.dominant_bias == 'NEUTRAL':
            logger.warning("❌ TRADE BLOCKED: Market is NEUTRAL (No clear direction).")
            return False
            
        logger.info("✅ TRADE APPROVED: All gates passed.")
        return True
    
    def strategize(self, snapshot: MarketSnapshot) -> List[TradeLeg]:
        """
        PHASE 4: STRATEGIZE
        Determine Trade Structure (Single or Twin) based on Coherence.
        """
        self.state = EngineState.STRATEGIZING
        
        direction = "BUY" if snapshot.dominant_bias == "BULLISH" else "SELL"
        
        if snapshot.recommendation == 'TWIN_AGGRESSIVE':
            logger.info("♟️ STRATEGY: TWIN SNIPER (Cash Flow + Wealth Runner)")
        else:
            logger.info("♟️ STRATEGY: SINGLE SHOT (Standard Entry)")
            
        legs = self.strategy_engine.analyze_strategic_intent(
            market_price=snapshot.price,
            neural_probs=snapshot.neural_probs,
            macro_context={
                'direction': direction,
                'alignment_map': snapshot.alignment_map # We need to pass this
            },
            titanium_level=snapshot.titanium_floor,
            microstructure_state=snapshot.microstructure,
            physics_metrics=getattr(snapshot, 'physics_data', {})
        )
        
        for i, leg in enumerate(legs):
            logger.info(f"   Leg {i+1}: {leg.direction} | {leg.strategy_type.value} | Vol: {leg.volume} | {leg.intent}")
            
            # Log for Neural Health Monitoring (Drift Detection)
            # Use the max probability as the "Confidence" score
            confidence = max(snapshot.neural_probs.values()) if snapshot.neural_probs else 0.5
            neural_health.log_prediction(
                trade_id=f"T-{int(time.time())}-{i}", 
                confidence=confidence, 
                action=leg.strategy_type.value
            )
            
        return legs
    
    def execute(self, legs: List[TradeLeg]):
        """
        PHASE 5: EXECUTE
        Dispatch orders via MT5 Bridge.
        """
        self.state = EngineState.EXECUTING
        from app.core.mt5_conn import mt5_conn
        from app.core.mt5_conn.utils import mt5
        
        for leg in legs:
            if self.demo_mode:
                logger.info(f"📨 [DEMO] ORDER SENT: {leg.direction} {leg.volume} lots @ Market")
            else:
                # REAL EXECUTION BRIDGE
                async def _send():
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": "XAUUSD",
                        "volume": leg.volume,
                        "type": mt5.ORDER_TYPE_BUY if leg.direction == "BUY" else mt5.ORDER_TYPE_SELL,
                        "price": mt5.symbol_info_tick("XAUUSD").ask if leg.direction == "BUY" else mt5.symbol_info_tick("XAUUSD").bid,
                        "magic": 123456,
                        "comment": leg.intent,
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = await mt5_conn.execute(mt5.order_send, request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"✅ REAL ORDER EXECUTED: {leg.direction} {leg.volume} @ {result.price}")
                    else:
                        logger.error(f"❌ EXECUTION FAILED: {result.comment if result else 'Unknown Error'}")
                
                import asyncio
                asyncio.create_task(_send())
                
            self.active_trades.append(leg)

    async def _get_latest_ohlc(self) -> pd.DataFrame:
        """Helper to get OHLC data from MT5 for kinetic analysis."""
        from app.core.mt5_conn import mt5_conn
        from app.core.mt5_conn.utils import mt5
        
        symbol = "XAUUSD" # Default
        mt5_tf = mt5.TIMEFRAME_M1
        rates = await mt5_conn.execute(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, 100)
        
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
        
        # Fallback to synthetic ONLY if MT5 fails
        logger.warning("⚠️ MT5 OHLC Fetch Failed. Using synthetic data for kinetic pulse.")
        prices = np.linspace(2045, 2050, 100) + np.random.normal(0, 0.5, 100)
        return pd.DataFrame({
            'open': prices, 'high': prices + 0.5, 'low': prices - 0.5, 
            'close': prices, 'volume': np.random.randint(100, 1000, 100)
        })
            
    def manage(self, snapshot: MarketSnapshot):
        """
        PHASE 6: MANAGE
        Check for Dynamic Promotion (Scalp -> Swing).
        """
        self.state = EngineState.MANAGING
        
        # StrategyEngine has optimize_targets (inherited from previous implementation)
        promoted = self.strategy_engine.optimize_targets(
            active_trades=self.active_trades,
            fresh_probs=snapshot.neural_probs
        )
        
        if promoted:
            logger.info("⬆️ PROMOTION: Scalp position upgraded to SWING RUNNER!")
            
    def run_cycle(self):
        """
        Execute one full Sense->Perceive->Decide->Strategize->Execute->Manage cycle.
        """
        logger.info("=" * 50)
        logger.info("🔄 NEXUS CYCLE START")
        
        # 1. SENSE
        fractal_data = self.sense()
        
        # 2. PERCEIVE
        snapshot = await self.perceive(fractal_data)
        snapshot.alignment_map = fractal_data.get('alignment_map', {}) # Pass alignment along
        
        # 3. DECIDE
        should_trade = self.decide(snapshot)
        
        if should_trade:
            # 4. STRATEGIZE
            legs = self.strategize(snapshot)
            
            # 5. EXECUTE
            self.execute(legs)
            
        # 6. MANAGE (Always run for existing positions)
        if self.active_trades:
            self.manage(snapshot)
            
        logger.info(f"🔄 NEXUS CYCLE END | Active Trades: {len(self.active_trades)}")
        logger.info("=" * 50)
        
    async def run_loop(self, interval_seconds: int = 60):
        """
        Main execution loop (for live trading).
        """
        logger.info("🏁 NEXUS ENGINE STARTING MAIN LOOP...")
        
        # Start high-frequency tick ingestion
        tick_listener.start()
        
        try:
            while True:
                self.run_cycle()
                await asyncio.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("🛑 NEXUS ENGINE STOPPED BY USER")
            tick_listener.stop()

# Singleton
nexus = NexusEngine(demo_mode=True)

if __name__ == "__main__":
    # Run a single cycle for testing
    nexus.run_cycle()
