import numpy as np
import datetime
from nexus_core.fundamental_engine.engine import FundamentalEngine

class MacroTensorFactory:
    """
    [MACRO SENTINEL - NEURAL BRIDGE]
    Converts Fundamental Engine state into consumption-ready Tensors.
    """
    
    def __init__(self):
        # We use the FundamentalEngine (singleton-ish or instantiated here)
        # Note: In production, this should share the same instance as the Risk Engine to avoid double-fetching.
        # For now, we instantiate safely.
        self.engine = FundamentalEngine(calendar_provider="mock") 
        # TODO: Switch to "forexfactory" when live or inject provider config

    def process(self, timestamp: datetime.datetime = None) -> dict:
        """
        Generates the Macro Regime Tensor.
        Input: Timestamp (for potential backtesting simulation, though FundamentalEngine currently uses 'now').
        """
        # Note: FundamentalEngine.check_event_proximity() acts on 'now' by default.
        # Future improvement: Allow passing timestamp to check_event_proximity for backtesting.
        
        # For Neural Training (offline), we would likely mock this or use a database of historical events.
        # For Live Trading, check_event_proximity() is correct.
        
        tensor_dict = self.engine.get_macro_regime_tensor(currencies=["USD", "EUR", "GBP", "JPY"])
        
        # Convert dict to numpy arrays
        return {
            "macro_safe": np.array([tensor_dict["macro_safe"]], dtype=np.float32),
            "macro_caution": np.array([tensor_dict["macro_caution"]], dtype=np.float32),
            "macro_danger": np.array([tensor_dict["macro_danger"]], dtype=np.float32),
            "macro_time_to_event": np.array([tensor_dict["minutes_to_event"]], dtype=np.float32),
            "macro_position_multiplier": np.array([tensor_dict["position_multiplier"]], dtype=np.float32)
        }
