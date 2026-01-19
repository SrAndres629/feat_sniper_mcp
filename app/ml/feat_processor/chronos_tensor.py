import numpy as np
import datetime
import pytz
from typing import Dict

from nexus_core.chronos_engine.probability import ChronosProbabilityEngine

class ChronosTensorFactory:
    """
    [CHRONOS TENSOR]
    The 'Embedding' of Time.
    1. Cyclic (Sin/Cos) - Continuous representation.
    2. Probabilistic (Bayesian Priors) - Regime estimation.
    """
    
    def __init__(self):
        self.prob_engine = ChronosProbabilityEngine()
        self.bolivia_tz = pytz.timezone('America/La_Paz')

    def process(self, utc_time: datetime.datetime) -> Dict[str, np.ndarray]:
        """
        Generates the Full Temporal Embedding.
        """
        local_time = utc_time.astimezone(self.bolivia_tz)
        
        # 1. Cyclic Embeddings (The "Clock")
        # Hour (0-23)
        h = local_time.hour
        h_rad = 2 * np.pi * (h / 24.0)
        hour_sin = np.sin(h_rad)
        hour_cos = np.cos(h_rad)
        
        # Minute (0-59)
        m = local_time.minute
        m_rad = 2 * np.pi * (m / 60.0)
        min_sin = np.sin(m_rad)
        min_cos = np.cos(m_rad)
        
        # Day of Week (0-6)
        dow = local_time.weekday()
        # Embed Day of week as Sin/Cos too? Or One-Hot?
        # User said "day_of_week_onehot". Let's do One-Hot for DOW as it's discrete categorical often.
        # But Cyclic is better for "Friday -> Monday" continuity? No, weekend gap.
        # Let's stick to One-Hot for DOW (7 dims).
        dow_onehot = np.eye(7)[dow]
        
        # 2. Bayesian Priors (The "Map")
        probs = self.prob_engine.get_probabilities(utc_time)
        
        # 3. Volatility Regime Embedding (The "State")
        # We need the Phaser state to know the Regime
        from nexus_core.chronos_engine.phaser import GoldCyclePhaser, VolatilityRegime, Intent
        
        phaser = GoldCyclePhaser() 
        state = phaser.get_current_state(utc_time)
        
        # 3. Volatility Regime Embedding (Old Requirement - Keep for legacy or replace?)
        # User asked for "Liquidity State" (One Hot Intent) and "Expected Volatility".
        # Let's keep VolRegime as it provides useful context, but add the new ones.
        
        # Encoding: [LOW, HIGH, EXTREME, BLACKOUT]
        regime_map = {
            VolatilityRegime.LOW: [1, 0, 0, 0],
            VolatilityRegime.HIGH: [0, 1, 0, 0],
            VolatilityRegime.EXTREME: [0, 0, 1, 0],
            VolatilityRegime.BLACKOUT: [0, 0, 0, 1]
        }
        regime_tensor = np.array(regime_map.get(state.vol_regime, [1, 0, 0, 0]), dtype=np.float32)

        # 4. Liquidity State (Intent) - One Hot
        # [ACCUMULATION, MANIPULATION, EXPANSION, DISTRIBUTION]
        intent_map = {
            Intent.ACCUMULATION: [1, 0, 0, 0],
            Intent.MANIPULATION: [0, 1, 0, 0],
            Intent.EXPANSION:    [0, 0, 1, 0],
            Intent.DISTRIBUTION: [0, 0, 0, 1]
        }
        intent_tensor = np.array(intent_map.get(state.intent, [1, 0, 0, 0]), dtype=np.float32)
        
        # 5. Expected Volatility (Scalar from Profile)
        expected_vol = state.profile.expected_volatility

        return {
            # Continuous Time Coordinates
            "time_cyclic": np.array([hour_sin, hour_cos, min_sin, min_cos], dtype=np.float32),
            
            # Context Embeddings
            "day_of_week": dow_onehot.astype(np.float32),
            "volatility_regime": regime_tensor,
            "liquidity_state": intent_tensor,
            "expected_volatility": np.array([expected_vol], dtype=np.float32),
            
            # Bayesian Priors
            "prob_manipulation": np.array([probs.p_manipulation], dtype=np.float32),
            "prob_expansion": np.array([probs.p_expansion], dtype=np.float32),
            "prob_liquidity": np.array([probs.p_liquidity], dtype=np.float32),
            
            # Debug
            "debug_time_str": local_time.strftime("%H:%M"),
            "debug_regime": state.vol_regime.name,
            "debug_intent": state.intent.name
        }
