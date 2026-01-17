"""
Market Physics Engine v2.0 - Microstructure Hydrodynamics (Doctoral Grade)
==========================================================================
The Sensory Cortex: Institutional-grade market physics calculator.

Architecture based on "Market Microstructure Theory" (Kyle/Amihud):
- Mass (Inertia) != Volume. Mass = LIQUIDITY DEPTH (Resistance).
- Force != Price Change. Force = ORDER FLOW IMBALANCE (Aggression).
- Dynamics: Price moves when Aggression consumes Liquidity.

Key Metrics:
1. Amihud Illiquidity (ILLIQ): Cost of moving price (Proxy for Lack of Depth).
2. Order Flow Imbalance (OFI): Net aggressive buy/sell pressure.
3. Impact Pressure: True "Power" of the move (OFI * ILLIQ).
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Deque
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from app.core.config import settings

# Setup Logger
logger = logging.getLogger("feat.market_physics")


@dataclass
class MarketRegime:
    """[LEVEL 64] Institutional Microstructure State."""
    is_accelerating: bool       # True if Impact > Resistance
    is_liquidity_vacuum: bool   # True if Low Resistance (Thin Book)
    is_absorption: bool         # True if High Force + Low Move (Iceberg)
    
    impact_pressure: float      # The effective force displacing price
    liquidity_resistance: float # The estimated mass (Inverse ILLIQ)
    ofi_score: float           # Order Flow Imbalance Z-Score
    
    trend: str
    timestamp: str


class MarketPhysics:
    """
    The Sensory Cortex: Motor de Física de Microestructura.
    
    Reemplaza la física Newtoniana ingenua con Dinámica de Fluidos de Mercado.
    
    Equation of Motion:
        ΔP ~ Force / Mass
        ΔP ~ OFI / Liquidity
        
    We approximate 'Liquidity Mass' using the Amihud Illiquidity Ratio inverse.
    We approximate 'Force' (OFI) using the Tick Rule on Volume.
    """
    
    # Physics Constants
    MIN_DELTA_T: float = settings.PHYSICS_MIN_DELTA_T
    WARMUP_PERIODS: int = 50 # Need statistical significance for Z-scores
    DECAY_FACTOR: float = 0.95 # Generic decay for EWMA calculations

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Circular buffers for raw data
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self.prices: Deque[float] = deque(maxlen=window_size)
        self.volumes: Deque[float] = deque(maxlen=window_size)
        
        # Derived Microstructure buffers
        self.returns_abs: Deque[float] = deque(maxlen=window_size) # |Rt|
        self.ofi_history: Deque[float] = deque(maxlen=window_size)
        
        logger.info("[SATELLITE] Market Physics v2.0 Online (Microstructure Hydrodynamics)")

    def hydrate(self, ticks: List[Dict]) -> None:
        """
        Populates buffers with historical data for cold-start readiness.
        """
        if not ticks:
            return
            
        logger.info(f"[PHYSICS] Hydrating with {len(ticks)} samples...")
        
        for tick in ticks:
            self.ingest_tick(tick, is_hydration=True)
            
        logger.info("[PHYSICS] Hydration detailed analysis complete.")

    def ingest_tick(self, data: Dict, force_timestamp: float = None, is_hydration: bool = False) -> Optional[MarketRegime]:
        """
        Ingesta de Ticks con cálculo de microestructura.
        """
        try:
            # 1. Extraction
            vol = float(data.get('tick_volume', 0.0) or data.get('real_volume', 0.0))
            price = float(data.get('bid', 0.0) or data.get('close', 0.0))
            ts = force_timestamp if force_timestamp else datetime.now().timestamp()
            
            # Guard against zero volume/price artifacts
            if price <= 0:
                return None

            # [P0] Temporal Monotonicity
            if self.timestamps and ts <= self.timestamps[-1]:
                ts = self.timestamps[-1] + self.MIN_DELTA_T

            # 2. Delta Calculations (Tick Rule for OFI)
            delta_p = 0.0
            ofi_current = 0.0
            
            if len(self.prices) > 0:
                prev_price = self.prices[-1]
                delta_p = price - prev_price
                
                # Tick Rule:
                # If price went up, assume buy volume.
                # If price went down, assume sell volume.
                # If unchanged, assume continuation of previous direction (simplified to 0 here for robustness)
                direction = 1.0 if delta_p > 0 else (-1.0 if delta_p < 0 else 0.0)
                ofi_current = direction * vol
            
            # 3. Update Buffers
            self.timestamps.append(ts)
            self.prices.append(price)
            self.volumes.append(vol)
            self.returns_abs.append(abs(delta_p))
            self.ofi_history.append(ofi_current)

            # Warmup Check
            if len(self.prices) < self.WARMUP_PERIODS:
                if not is_hydration:
                    # logger.debug(f"Warming up... {len(self.prices)}/{self.WARMUP_PERIODS}")
                    _warming_up = True
                return None

            # 4. Compute Metrics
            return self._compute_regime(price, ts)

        except Exception as e:
            logger.error(f"Sensory Cortex Failure: {e}")
            return None

    def _compute_regime(self, current_price: float, current_ts: float) -> MarketRegime:
        """
        Calculates Institutional Metrics: ILLIQ, OFI, Impact.
        """
        # Convert to numpy for vector math
        vols = np.array(self.volumes)
        rets = np.array(self.returns_abs)
        ofis = np.array(self.ofi_history)
        
        # 1. Amihud Illiquidity (ILLIQ)
        # Ratio of |Return| to Volume.
        # High ILLIQ = Price moves easily with little volume (Low Liquidity).
        # Low ILLIQ = Price resists movement despite volume (High Liquidity).
        # We add epsilon to volume to avoid div by zero.
        epsilon = 1e-6
        daily_illiq_series = rets / (vols + epsilon)
        
        # We use an EWMA or rolling mean to estimate current friction
        current_illiq = np.mean(daily_illiq_series[-10:]) if len(daily_illiq_series) >= 10 else daily_illiq_series[-1]
        
        # "Resistance" is the inverse of Illiquidity.
        # High Resistance = Hard to move.
        liquidity_resistance = 1.0 / (current_illiq + epsilon)
        
        # 2. Order Flow Imbalance (OFI) Z-Score
        # Measures the statistical significance of the buying/selling pressure
        ofi_mean = np.mean(ofis)
        ofi_std = np.std(ofis) + epsilon
        ofi_z = (ofis[-1] - ofi_mean) / ofi_std
        
        # 3. Impact Pressure
        # Aggression * Sensitivity
        # A large Buy Order (Positive OFI) in a thin market (High ILLIQ) = Massive Impact
        impact_pressure = ofis[-1] * current_illiq * 1000 # Scaled for readability
        
        # 4. Regime Classification
        
        # Vacuum: Price moving fast on low volume (High ILLIQ)
        is_vacuum = current_illiq > (np.mean(daily_illiq_series) + 1.5 * np.std(daily_illiq_series))
        
        # Absorption: High OFI (Force) but Low Price Move (High Resistance)
        # i.e., Price didn't move as much as the force predicted
        expected_move = ofis[-1] * np.mean(daily_illiq_series)
        actual_move = rets[-1]
        
        # If we pushed strict volume but price didn't budge -> Absorption
        is_absorption = (abs(ofis[-1]) > np.percentile(np.abs(ofis), 80)) and \
                        (actual_move < np.mean(rets))
                        
        # Acceleration: High Impact Pressure
        # True displacement happening
        is_accelerating = abs(impact_pressure) > (np.mean(np.abs(ofis * daily_illiq_series * 1000)) * 1.5)
        
        trend = "BULLISH" if ofis[-1] > 0 else "BEARISH" if ofis[-1] < 0 else "NEUTRAL"
        
        return MarketRegime(
            is_accelerating=is_accelerating,
            is_liquidity_vacuum=bool(is_vacuum),
            is_absorption=bool(is_absorption),
            impact_pressure=float(impact_pressure),
            liquidity_resistance=float(liquidity_resistance),
            ofi_score=float(ofi_z),
            trend=trend,
            timestamp=datetime.fromtimestamp(current_ts).isoformat()
        )

    def get_status(self) -> Dict:
        """Get current engine status for monitoring."""
        return {
            "window_size": self.window_size,
            "samples": len(self.prices),
            "mode": "MICROSTRUCTURE_V2"
        }


# Global Singleton
market_physics = MarketPhysics()
