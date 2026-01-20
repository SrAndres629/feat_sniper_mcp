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

# Diagnosis Imports
from tools.fractal_diagnosis import diagnose_market_fractals

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

class NexusEngine:
    """
    The Autonomous Trading Brain.
    Implements the Sense -> Perceive -> Decide -> Strategize -> Execute -> Manage loop.
    """
    
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.state = EngineState.IDLE
        self.strategy_engine = StrategyEngine(risk_officer)
        self.active_trades: List[TradeLeg] = []
        
        # Coherence Gate (From Fractal Diagnosis)
        self.min_coherence_for_twin = 0.75
        self.min_coherence_for_trade = 0.50
        
        logger.info("🚀 NEXUS ENGINE INITIALIZED")
        logger.info(f"   Mode: {'DEMO' if demo_mode else 'LIVE'}")
        logger.info(f"   Strategy: Twin Sniper Evolution")
        logger.info(f"   Account Phase: {risk_officer.phase_name}")
        
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
    
    def perceive(self, fractal_data: dict) -> MarketSnapshot:
        """
        PHASE 2: PERCEIVE
        Calculate Titanium Floor (PVP + EMA + Wavelet confluence).
        """
        self.state = EngineState.PERCEIVING
        
        # 1. Get Real Microstructure State (via Scanner)
        micro_state_obj = micro_scanner.get_dict()
        
        # Fallback mock for demo visualization if scanner has no tick history yet
        if self.demo_mode and micro_state_obj['entropy_score'] == 0.5:
             micro_state_obj = {
                "entropy_score": 0.35 if fractal_data['coherence_score'] > 0.6 else 0.75,
                "ofi_z_score": 1.5 if fractal_data['dominant_bias'] == 'BULLISH' else -0.5,
                "hurst": 0.65
             }

        coherence = fractal_data['coherence_score']
        
        # Titanium Floor Detection (Simulated for now, would use ConvergenceEngine)
        titanium = coherence > 0.70
        
        # Mock Neural Probabilities (Mapping from fractal diagnosis)
        if fractal_data['dominant_bias'] == 'BULLISH':
            neural_probs = {'scalp': 0.85, 'day': 0.60, 'swing': 0.40}
        elif fractal_data['dominant_bias'] == 'BEARISH':
            neural_probs = {'scalp': 0.80, 'day': 0.55, 'swing': 0.30}
        else:
            neural_probs = {'scalp': 0.50, 'day': 0.40, 'swing': 0.20}
            
        snapshot = MarketSnapshot(
            price=2050.0,  # Simulation Price
            fractal_coherence=coherence,
            dominant_bias=fractal_data['dominant_bias'],
            titanium_floor=titanium,
            neural_probs=neural_probs,
            recommendation=fractal_data['recommendation'],
            microstructure=micro_state_obj
        )
        
        if titanium:
            logger.info(f"🔥 TITANIUM FLOOR DETECTED | Direction: {snapshot.dominant_bias}")
        
        return snapshot
    
    def decide(self, snapshot: MarketSnapshot) -> bool:
        """
        PHASE 3: DECIDE
        Should we trade? Gate based on Coherence, Titanium and Noise.
        """
        self.state = EngineState.DECIDING
        
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
            macro_context={'direction': direction},
            titanium_level=snapshot.titanium_floor,
            microstructure_state=snapshot.microstructure
        )
        
        for i, leg in enumerate(legs):
            logger.info(f"   Leg {i+1}: {leg.direction} | {leg.strategy_type.value} | Vol: {leg.volume} | {leg.intent}")
            
        return legs
    
    def execute(self, legs: List[TradeLeg]):
        """
        PHASE 5: EXECUTE
        Dispatch orders via MT5 Bridge (or Mock in Demo).
        """
        self.state = EngineState.EXECUTING
        
        for leg in legs:
            if self.demo_mode:
                logger.info(f"📨 [DEMO] ORDER SENT: {leg.direction} {leg.volume} lots @ Market")
            else:
                # TODO: Integrate with MT5 bridge
                pass
                
            self.active_trades.append(leg)
            
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
        snapshot = self.perceive(fractal_data)
        
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
        
    def run_loop(self, interval_seconds: int = 60):
        """
        Main execution loop (for live trading).
        """
        logger.info("🏁 NEXUS ENGINE STARTING MAIN LOOP...")
        
        try:
            while True:
                self.run_cycle()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("🛑 NEXUS ENGINE STOPPED BY USER")

# Singleton
nexus = NexusEngine(demo_mode=True)

if __name__ == "__main__":
    # Run a single cycle for testing
    nexus.run_cycle()
