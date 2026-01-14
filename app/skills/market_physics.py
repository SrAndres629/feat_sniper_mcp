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
    is_accelerating: bool
    acceleration_score: float
    vol_z_score: float
    trend: str
    timestamp: str

class MarketPhysics:
    """
    The Sensory Cortex: Motor de Física de Mercado Institucional.
    Calcula Velocidad, Aceleración y Anomalías de Volumen en tiempo real.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        # Buffers circulares para datos crudos
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self.volume_window: Deque[float] = deque(maxlen=window_size)
        self.price_window: Deque[float] = deque(maxlen=window_size)
        
        # Buffer de aceleracion histórica para cálculo de sigma
        self.acceleration_history: Deque[float] = deque(maxlen=window_size)
        
        logger.info("[SATELLITE] Market Physics Engine Online (Numpy Accelerated)")

    def ingest_tick(self, data: Dict, force_timestamp: float = None) -> Optional[MarketRegime]:
        """
        Ingesta de Ticks con cálculo de física JIT.
        """
        try:
            # 1. Extracción y Normalización
            vol = float(data.get('tick_volume', 0.0) or data.get('real_volume', 0.0))
            price = float(data.get('bid', 0.0) or data.get('close', 0.0))
            ts = force_timestamp if force_timestamp else datetime.now().timestamp()

            # 2. Update Buffers
            self.volume_window.append(vol)
            self.price_window.append(price)
            self.timestamps.append(ts)

            # Warmup Check
            if len(self.price_window) < 50: # Minimo 50 periodos para StdDev confiable
                return None

            # 3. Cálculo Vectorizado
            return self.calculate_acceleration_vectorized(vol, price, ts)

        except Exception as e:
            logger.error(f"Sensory Cortex Failure: {e}")
            return None

    def calculate_acceleration_vectorized(self, current_vol: float, current_price: float, current_ts: float) -> MarketRegime:
        """
        Fórmula FEAT: 
        A = (Vol / AvgVol) * (DeltaPrecio / DeltaTiempo)
        Trigger: A > 2.0 * Sigma(A)
        """
        # Convertir a numpy arrays para velocidad
        vols = np.array(self.volume_window)
        prices = np.array(self.price_window)
        times = np.array(self.timestamps)

        # Estadísticas de Volumen
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        
        # Vol Ratio (Intensidad)
        if mean_vol == 0: mean_vol = 1
        vol_intensity = current_vol / mean_vol
        
        # Z-Score de Volumen (Para diagnósticos)
        vol_z = (current_vol - mean_vol) / (std_vol if std_vol > 0 else 1)

        # Velocidad del Precio (Delta P / Delta T)
        # Usamos los últimos 2 puntos para velocidad instantánea
        if len(prices) >= 2 and (times[-1] - times[-2]) > 0:
            delta_p = prices[-1] - prices[-2]
            delta_t = times[-1] - times[-2]
            velocity = delta_p / delta_t
        else:
            velocity = 0.0

        # Aceleración FEAT (Magnitud Absoluta)
        raw_acceleration = vol_intensity * abs(velocity)
        
        # Guardar en histórico de aceleración
        self.acceleration_history.append(raw_acceleration)

        # Regla de Activación: 2 Sigma
        if len(self.acceleration_history) > 10:
            acc_array = np.array(self.acceleration_history)
            acc_mean = np.mean(acc_array)
            acc_std = np.std(acc_array)
            
            # Threshold Dinámico
            threshold = acc_mean + (2.0 * acc_std)
            is_accelerating = raw_acceleration > threshold and raw_acceleration > 0
        else:
            is_accelerating = False

        trend = "BULLISH" if velocity > 0 else "BEARISH" if velocity < 0 else "NEUTRAL"
        
        if is_accelerating:
            logger.info(f"🚀 FEAT PHYSICS TRIGGER: Accel={raw_acceleration:.4f} > Threshold (Trend: {trend})")

        return MarketRegime(
            is_accelerating=is_accelerating,
            acceleration_score=raw_acceleration,
            vol_z_score=vol_z,
            trend=trend,
            timestamp=datetime.fromtimestamp(current_ts).isoformat()
        )

# Instancia Global
market_physics = MarketPhysics()
