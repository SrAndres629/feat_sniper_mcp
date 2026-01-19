import numpy as np
import pandas as pd
from typing import Dict
import datetime

# Import sub-processors
from app.ml.feat_processor.time_flow import TemporalTensorEngine
from app.ml.feat_processor.chronos import ChronosTensorFactory
from app.ml.feat_processor.macro import MacroTensorFactory

from app.ml.feat_processor.alpha_tensor import AlphaTensorOrchestrator

class TimeFeatureProcessor:
    """
    [ALPHA CORE: MASTER INTERFACE]
    The unified interface for the Doctoral Alpha Tensors.
    Orchestrates Structure, Chronos, and Macro layers via vectorization.
    """
    
    def __init__(self):
        self.orchestrator = AlphaTensorOrchestrator()

    def process(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Processes entire DataFrame block.
        """
        return self.orchestrator.process_dataframe(df)

    def get_last_state(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        For live trading: gets the final probabilistic state.
        """
        return self.orchestrator.get_live_alpha(df)
