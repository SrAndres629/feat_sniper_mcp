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

# Setup Logger
logger = logging.getLogger("feat.market_physics")


@dataclass
class MarketRegime:
    """
    Output of physics engine - describes current market state.
    
    Attributes:
        is_accelerating: True if momentum exceeds μ + 2σ threshold
        is_initiative_candle: True if Volume > 2.5x and Displacement > 1.5x ATR (FEAT CORE)
        acceleration_score: Dimensionless acceleration coefficient [0, ∞)
        vol_z_score: Volume Z-score (standard deviations from mean)
        trend: BULLISH, BEARISH, or NEUTRAL
        atr: Current ATR value (for downstream use)
        timestamp: ISO timestamp of calculation
    """
    is_accelerating: bool
    is_initiative_candle: bool
    acceleration_score: float
    vol_z_score: float
    trend: str
    atr: float  # [P0 FIX] Added for downstream guards
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
    
    # Physical Invariants
    MIN_DELTA_T: float = 0.001  # 1ms floor - prevents infinite acceleration
    MAX_VELOCITY: float = 1e6   # Velocity cap to prevent overflow
    MIN_ATR: float = 1e-8       # [P0 FIX] ATR floor to prevent division by zero

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

            # Warmup Check - need 50 periods for reliable statistics
            if len(self.price_window) < 50:
                return None

            # 3. Vectorized Calculation
            return self._calculate_acceleration_vectorized(vol, price, ts)

        except Exception as e:
            logger.error(f"Sensory Cortex Failure: {e}")
            return None

    def _calculate_acceleration_vectorized(self, current_vol: float, current_price: float, current_ts: float) -> MarketRegime:
        """
        [P0 REPAIR] ATR-Normalized FEAT Acceleration Formula:
        
        Step 1: Vol_Intensity = Current_Vol / Mean_Vol  [dimensionless ratio]
        Step 2: Raw_Velocity = ΔP / Δt                  [price/second]
        Step 3: ATR = mean(|ΔP|)                        [price units]
        Step 4: Norm_Velocity = Raw_Velocity / ATR      [dimensionless]
        Step 5: Acceleration = Vol_Intensity × |Norm_Velocity|  [dimensionless]
        
        Trigger: Acceleration > μ(Acceleration) + 2σ(Acceleration)
        
        This formula is ASSET-AGNOSTIC:
        - Gold at $2000 with $10 move = same acceleration as
        - BTC at $60000 with $300 move (both 0.5% moves)
        """
        # Convert to numpy for vectorized operations
        vols = np.array(self.volume_window)
        prices = np.array(self.price_window)
        times = np.array(self.timestamps)

        # =====================================================
        # STEP 1: Volume Intensity (dimensionless)
        # =====================================================
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        
        if mean_vol == 0:
            mean_vol = 1
        vol_intensity = current_vol / mean_vol
        
        # Volume Z-Score (for diagnostics)
        vol_z = (current_vol - mean_vol) / max(std_vol, 0.0001)

        # =====================================================
        # STEP 2: Raw Velocity (price/second)
        # =====================================================
        if len(prices) >= 2:
            delta_p = prices[-1] - prices[-2]
            raw_delta_t = times[-1] - times[-2]
            
            # INVARIANT: δt never less than MIN_DELTA_T
            delta_t = max(raw_delta_t, self.MIN_DELTA_T)
            
            # Raw velocity with safety cap
            raw_velocity = delta_p / delta_t
            raw_velocity = np.clip(raw_velocity, -self.MAX_VELOCITY, self.MAX_VELOCITY)
        else:
            raw_velocity = 0.0

        # =====================================================
        # STEP 3: ATR Calculation (price units)
        # =====================================================
        atr = self._calculate_atr(prices)
        self._cached_atr = atr

        # =====================================================
        # STEP 4: Normalized Velocity (dimensionless)
        # =====================================================
        # [P0 FIX] This is the key normalization step
        normalized_velocity = raw_velocity / atr

        # =====================================================
        # STEP 5: FEAT Acceleration (dimensionless)
        # =====================================================
        # Combines volume conviction with price momentum
        raw_acceleration = vol_intensity * abs(normalized_velocity)
        
        # Store in history for σ calculation
        self.acceleration_history.append(raw_acceleration)

        # =====================================================
        # TRIGGER: μ + 2σ Dynamic Threshold
        # =====================================================
        if len(self.acceleration_history) > 10:
            acc_array = np.array(self.acceleration_history)
            acc_mean = np.mean(acc_array)
            acc_std = np.std(acc_array)
            
            # Dynamic threshold based on recent acceleration distribution
            threshold = acc_mean + (2.0 * acc_std)
            is_accelerating = raw_acceleration > threshold and raw_acceleration > 0
        else:
            is_accelerating = False
            threshold = 0

        # Trend direction (from raw velocity, not normalized)
        trend = "BULLISH" if raw_velocity > 0 else "BEARISH" if raw_velocity < 0 else "NEUTRAL"
        
        if is_accelerating:
            logger.info(
                f"🚀 FEAT PHYSICS TRIGGER: Accel={raw_acceleration:.4f} > {threshold:.4f} "
                f"(Vol:{vol_intensity:.2f}x, NormVel:{normalized_velocity:.4f}, ATR:{atr:.2f}) "
                f"Trend: {trend}"
            )

        # =====================================================
        # STEP 6: Initiative Candle (FEAT CORE)
        # =====================================================
        # Rule: Volume Intensity > 2.5 AND Normalized Velocity > 1.5
        is_initiative = vol_intensity > 2.5 and abs(normalized_velocity) > 1.5

        if is_initiative:
            logger.info(f"🧬 [FEAT] INITIATIVE CANDLE DETECTED: Vol:{vol_intensity:.2f}x | Vel:{normalized_velocity:.2f}x")

        return MarketRegime(
            is_accelerating=is_accelerating,
            is_initiative_candle=is_initiative,
            acceleration_score=raw_acceleration,
            vol_z_score=vol_z,
            trend=trend,
            atr=atr,  # [P0 FIX] Include ATR for downstream use
            timestamp=datetime.fromtimestamp(current_ts).isoformat()
        )

    def get_status(self) -> Dict:
        """Get current engine status for monitoring."""
        return {
            "buffer_size": len(self.price_window),
            "window_size": self.window_size,
            "cached_atr": round(self._cached_atr, 4),
            "acceleration_history_size": len(self.acceleration_history),
            "is_warmed_up": len(self.price_window) >= 50
        }


# Global Singleton
market_physics = MarketPhysics()
