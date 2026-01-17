"""
Market Physics Engine - ATR-Normalized Acceleration (P0 REPAIR)
================================================================
The Sensory Cortex: Institutional-grade market physics calculator.

[P0 REPAIR] Fixes:
- ATR Normalization: Velocity divided by ATR for asset-agnostic metrics
- Acceleration is now dimensionless (works identically for Gold and Bitcoin)
- Added ATR calculation from price window
- Improved documentation of physical formulas
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
    """[LEVEL 61] Stochastic Market State Representation."""
    is_accelerating: bool
    is_initiative_candle: bool
    acceleration_score: float
    vol_z_score: float
    acceleration_prob: float  # P(Accelerating | Price, Volume, Time)
    energy_score: float       # Kinetic Energy (Mass * Velocity^2)
    trend: str
    atr: float
    timestamp: str


class MarketPhysics:
    """
    The Sensory Cortex: Motor de Física de Mercado Institucional.
    
    [P0 REPAIR] ATR-Normalized Acceleration Formula:
    
        Raw Velocity = ΔP / Δt                    [USD/second]
        Normalized Velocity = Raw Velocity / ATR  [dimensionless]
        Acceleration = Vol_Intensity × |Norm_Velocity|  [dimensionless]
    
    This makes acceleration ASSET-AGNOSTIC:
    - A 0.5% move in Gold (from $2000) = acceleration X
    - A 0.5% move in BTC (from $60000) = acceleration X (same!)
    
    [P0-1 FIX] Invariantes de Física Garantizados:
    - MIN_DELTA_T: Floor de 1ms para evitar división por cero
    - Monotonía Temporal: Timestamps siempre crecientes (causalidad)
    - Aceleración Finita: Matemáticamente acotada en todos los escenarios
    """
    
    # Physical Invariants [LEVEL 57]
    MIN_DELTA_T: float = settings.PHYSICS_MIN_DELTA_T
    MAX_VELOCITY: float = settings.PHYSICS_MAX_VELOCITY
    MIN_ATR: float = settings.PHYSICS_MIN_ATR
    WARMUP_PERIODS: int = settings.PHYSICS_WARMUP_PERIODS

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Circular buffers for raw data
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self.volume_window: Deque[float] = deque(maxlen=window_size)
        self.price_window: Deque[float] = deque(maxlen=window_size)
        
        # Buffer for acceleration history (for σ calculation)
        self.acceleration_history: Deque[float] = deque(maxlen=window_size)
        
        # [P0 FIX] Cache ATR for efficiency
        self._cached_atr: float = 0.0
        
        logger.info("[SATELLITE] Market Physics Engine Online (ATR-Normalized, NumPy Accelerated)")

    def hydrate(self, ticks: List[Dict]) -> None:
        """
        [P0 FIX] Profound Hydration Protocol.
        Populates buffers with historical data to avoid warmup lag.
        
        Args:
            ticks: List of historical ticks/candles
        """
        if not ticks:
            return
            
        logger.info(f"[PHYSICS] Hydrating with {len(ticks)} samples...")
        
        for tick in ticks:
            price = float(tick.get('bid', 0.0) or tick.get('close', 0.0))
            vol = float(tick.get('tick_volume', 0.0) or tick.get('real_volume', 0.0))
            ts = tick.get('time')
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                except ValueError:
                    ts = None
            
            self.price_window.append(price)
            self.volume_window.append(vol)
            if ts:
                self.timestamps.append(ts)
            else:
                # Synthetic sequential timestamp if missing
                last_ts = self.timestamps[-1] if self.timestamps else datetime.now().timestamp()
                self.timestamps.append(last_ts + 0.01)
                
        # Calculate initial ATR
        if len(self.price_window) >= 2:
            self._cached_atr = self._calculate_atr(np.array(self.price_window))
            
        logger.info(f"[PHYSICS] Hydration complete. ATR={self._cached_atr:.4f}")

    def _calculate_atr(self, prices: np.ndarray) -> float:
        """
        Calculate Average True Range from price series.
        
        For tick data without full OHLC, we approximate ATR as the 
        average of absolute price changes (simplified True Range).
        
        [P0 FIX] This enables asset-agnostic velocity normalization.
        
        Args:
            prices: Array of closing prices
            
        Returns:
            float: ATR value (same units as price)
        """
        if len(prices) < 2:
            return self.MIN_ATR
        
        # Simplified ATR: mean of absolute price changes
        price_changes = np.abs(np.diff(prices))
        atr = np.mean(price_changes)
        
        # Floor to prevent division by zero
        return max(atr, self.MIN_ATR)

    def ingest_tick(self, data: Dict, force_timestamp: float = None) -> Optional[MarketRegime]:
        """
        Ingesta de Ticks con cálculo de física JIT.
        
        Args:
            data: Dict with 'bid'/'close' and 'tick_volume'/'real_volume'
            force_timestamp: Optional forced timestamp for backtesting
            
        Returns:
            MarketRegime if enough data, None during warmup
        """
        try:
            # 1. Extraction and Normalization
            vol = float(data.get('tick_volume', 0.0) or data.get('real_volume', 0.0))
            price = float(data.get('bid', 0.0) or data.get('close', 0.0))
            ts = force_timestamp if force_timestamp else datetime.now().timestamp()

            # [P0-1 FIX] Temporal Monotonicity Guard
            if self.timestamps and ts <= self.timestamps[-1]:
                ts = self.timestamps[-1] + self.MIN_DELTA_T
                # logger.debug(f"[PHYSICS] Sub-second tick adjusted: ts={ts:.6f}")

            # 2. Update Buffers
            self.volume_window.append(vol)
            self.price_window.append(price)
            self.timestamps.append(ts)

            # Warmup Check - need enough periods for reliable statistics
            if len(self.price_window) < self.WARMUP_PERIODS:
                return None

            # 3. Vectorized Calculation
            return self._calculate_acceleration_vectorized(vol, price, ts)

        except Exception as e:
            logger.error(f"Sensory Cortex Failure: {e}")
            return None

    def _calculate_acceleration_vectorized(self, current_vol: float, current_price: float, current_ts: float) -> MarketRegime:
        """
        [PH.D. REFACTOR] Stochastic FEAT Physics:
        
        Converts deterministic triggers into Probabilistic Log-Likelihoods.
        Computes Kinetic Energy Tensor E = 0.5 * m * v^2 where m is Volume Intensity.
        """
        from scipy.stats import norm
        
        # Convert to numpy for vectorized operations
        vols = np.array(self.volume_window)
        prices = np.array(self.price_window)
        times = np.array(self.timestamps)

        # 1. Volume Intensity (Mass)
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        vol_intensity = current_vol / max(mean_vol, 1e-6)
        vol_z = (current_vol - mean_vol) / max(std_vol, 1e-6)

        # 2. ATR-Normalized Velocity
        if len(prices) >= 2:
            delta_p = prices[-1] - prices[-2]
            delta_t = max(times[-1] - times[-2], self.MIN_DELTA_T)
            raw_velocity = delta_p / delta_t
            raw_velocity = np.clip(raw_velocity, -self.MAX_VELOCITY, self.MAX_VELOCITY)
        else:
            raw_velocity = 0.0

        atr = self._calculate_atr(prices)
        self._cached_atr = atr
        norm_velocity = raw_velocity / atr

        # 3. FEAT Acceleration (Dimensionless Momentum)
        raw_acceleration = vol_intensity * abs(norm_velocity)
        self.acceleration_history.append(raw_acceleration)

        # 4. Kinetic Energy Calculation (Mass * V^2)
        energy_score = 0.5 * vol_intensity * (norm_velocity ** 2)

        # 5. Stochastic Regime Detection (P-Value of Acceleration)
        acc_prob = 0.0
        is_accelerating = False
        if len(self.acceleration_history) > 10:
            acc_array = np.array(self.acceleration_history)
            acc_mean = np.mean(acc_array)
            acc_std = max(np.std(acc_array), 1e-6)
            
            # Z-Score of current acceleration
            acc_z = (raw_acceleration - acc_mean) / acc_std
            # Map Z-Score to CDF (Probability)
            acc_prob = float(norm.cdf(acc_z))
            
            # Deterministic Trigger derived from Probabilistic Threshold
            threshold_z = settings.PHYSICS_REGIME_SIGMA 
            is_accelerating = acc_z > threshold_z
        
        trend = "BULLISH" if raw_velocity > 0 else "BEARISH" if raw_velocity < 0 else "NEUTRAL"

        # 6. Initiative Candle (Probabilistic Requirement)
        is_initiative = (vol_intensity > settings.PHYSICS_INITIATIVE_VOL_THRESHOLD and 
                         abs(norm_velocity) > settings.PHYSICS_INITIATIVE_VEL_THRESHOLD)

        return MarketRegime(
            is_accelerating=is_accelerating,
            is_initiative_candle=is_initiative,
            acceleration_score=raw_acceleration,
            vol_z_score=vol_z,
            acceleration_prob=acc_prob,
            energy_score=energy_score,
            trend=trend,
            atr=atr,
            timestamp=datetime.fromtimestamp(current_ts).isoformat()
        )

    def get_status(self) -> Dict:
        """Get current engine status for monitoring."""
        return {
            "buffer_size": len(self.price_window),
            "window_size": self.window_size,
            "cached_atr": round(self._cached_atr, 4),
            "acceleration_history_size": len(self.acceleration_history),
            "is_warmed_up": len(self.price_window) >= self.WARMUP_PERIODS
        }


# Global Singleton
market_physics = MarketPhysics()
