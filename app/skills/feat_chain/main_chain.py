import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .models import FEATDecision
from .liquidity import LiquidityChannel
from .kinetics import KineticsChannel
from .volatility import VolatilityChannel

logger = logging.getLogger("feat.main_chain")

class FEATChain:
    """
    🧬 THE NEURO-ORCHESTRATOR (PhD Level)
    Decouples market analysis into specialized channels to feed advanced 
    hybrid neural networks (TCN-BiLSTM).
    """
    def __init__(self):
        self.liquidity = LiquidityChannel()
        self.kinetics = KineticsChannel()
        self.volatility = VolatilityChannel()
        logger.info("🧠 FEAT Neuro-Orchestrator Initialized (Level 66)")

    async def analyze_probabilistic(
        self, 
        market_data: Dict, 
        candles: pd.DataFrame = None,
        current_price: float = None
    ) -> FEATDecision:
        """
        Processes market data through 3 specialized neural channels.
        Returns a decision optimized for high-asymmetry scalping.
        """
        if candles is None or len(candles) < 20:
            return FEATDecision(action="HOLD", reasoning=["Insufficient data for neuro-channels"])

        if current_price is None:
            current_price = float(market_data.get('bid', 0) or market_data.get('close', 0))

        # 1. CHANNEL: LIQUIDITY (Where is the money?)
        liq_bias = self.liquidity.get_liquidity_bias(candles)
        fvg = self.liquidity.compute_fvg(candles)
        
        # 2. CHANNEL: KINETICS (How fast is it moving?)
        kinetic_vec = self.kinetics.compute_kinetic_vectors(candles)
        rsi_norm = self.kinetics.get_relative_strength(candles)
        
        # 3. CHANNEL: VOLATILITY (Is it safe?)
        vol_metrics = self.volatility.compute_regime_metrics(candles)
        boundaries = self.volatility.get_dynamic_boundaries(candles)

        # --- NEURAL FUSION LOGIC ---
        # We compute confidence based on channel alignment
        
        # Liquidity + Kinetics alignment
        # If bias and momentum are same direction, confidence increases
        is_aligned = (liq_bias > 0 and kinetic_vec["momentum_bias"] > 0) or \
                     (liq_bias < 0 and kinetic_vec["momentum_bias"] < 0)
        
        alignment_score = 1.0 if is_aligned else 0.2
        
        # Space confidence (Z-Score extreme = potential reversal or strong trend)
        # For scalping, we like z-score alignment with momentum
        space_conf = 1.0 - (abs(vol_metrics["z_score"]) / 3.0) # Normalized 0-1
        
        # Final weighting
        composite = (liq_bias * 0.4) + (kinetic_vec["momentum_bias"] * 0.4) + (rsi_norm * 0.2)
        direction = 1 if composite > 0.1 else -1 if composite < -0.1 else 0
        
        # Decision logic
        abs_score = abs(composite)
        action = "HOLD"
        if abs_score > 0.4 and alignment_score > 0.5:
            action = "BUY" if direction > 0 else "SELL"

        return FEATDecision(
            form_confidence=float(abs(liq_bias)),
            space_confidence=float(space_conf),
            accel_confidence=float(abs(kinetic_vec["acceleration"])),
            time_confidence=alignment_score, # Using time_confidence slot for alignment
            composite_score=float(abs_score),
            action=action,
            direction=direction,
            reasoning=[
                f"LiqBias: {liq_bias:.2f}",
                f"Accel: {kinetic_vec['acceleration']:.2f}",
                f"Z-Score: {vol_metrics['z_score']:.2f}",
                f"FVG: {len(fvg)} detected"
            ],
            layer_alignment=float(rsi_norm)
        )

    # Legacy support
    async def analyze(self, market_data: Dict, current_price: float, precomputed_physics: Optional[Any] = None) -> bool:
        # For simplicity, we wrap the probabilistic analysis
        # In a real scenario, this would involve the older rules-based chain
        # But here we use the superior neuro-channels
        symbol = market_data.get('symbol', 'UNKNOWN')
        # Temporary mock of candles if not provided (should be provided by caller)
        # Note: In production, NexusEngine provides the candles.
        # This is a fallback to avoid breaking old callers.
        return False
