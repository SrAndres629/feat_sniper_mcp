import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .utils import process_ticks_to_ohlcv
from .engineering import apply_feat_engineering
from .kinetics import calculate_multifractal_layers
from .vision import generate_energy_map
from .tensor import tensorize_snapshot
from .io import export_parquet, export_jsonl_gz
from nexus_core.math_engine import calculate_kalman_filter
from nexus_core.features import feat_features

logger = logging.getLogger("FeatProcessor.Engine")

from nexus_core.kinetic_engine import kinetic_engine
from app.ml.feat_processor.alpha_tensor import AlphaTensorOrchestrator
# We use AlphaTensorOrchestrator logic to ensure training matches live logic exactly.

class FeatProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "ohlcv_parquet"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ticks_jsonl"), exist_ok=True)
        self.alpha = AlphaTensorOrchestrator() # Use the Master Logic

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline: Engineering -> Kinetics -> Physics Fusion."""
        return self.process_dataframe(df)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [PHASE 14 - UNIFIED PHYSICS PIPELINE]
        Now delegates to AlphaTensorOrchestrator to ensure Training == Live Execution.
        """
        # We calculate the full Alpha Tensor payload which includes Structure, Chronos, and Physics.
        # But for 'df' enrichment we need to flatten it back to columns.
        
        # 1. Run Standard Feature Engineering (Legacy Support + Indicators)
        df = apply_feat_engineering(df)
        
        # 2. Run The Alpha Core (Physics, Structure, Time)
        # Note: AlphaTensorOrchestrator returns a dict of arrays. We need to assign them to DF.
        # Ideally, we should use AlphaTensorOrchestrator's internal components directly or map payload back.
        
        # Mapping payload back is safest to guarantee 1:1 match.
        payload = self.alpha.process_dataframe(df)
        
        # Assign Physics Tensors
        df["physics_force"] = payload.get("physics_force", np.zeros(len(df)))
        df["physics_energy"] = payload.get("physics_energy", np.zeros(len(df)))
        df["physics_entropy"] = payload.get("physics_entropy", np.zeros(len(df)))
        df["physics_viscosity"] = payload.get("physics_viscosity", np.zeros(len(df)))
        
        # Assign Temporal
        df["temporal_sin"] = payload.get("temporal_sin", np.zeros(len(df)))
        df["temporal_cos"] = payload.get("temporal_cos", np.zeros(len(df)))
        df["killzone_intensity"] = payload.get("killzone_intensity", np.zeros(len(df)))
        df["session_weight"] = payload.get("session_weight", np.zeros(len(df)))
        
        # Assign Structural
        df["feat_index"] = payload.get("structural_feat_index", np.zeros(len(df))) * 100.0
        df["confluence_score"] = payload.get("confluence_tensor", np.zeros(len(df))) * 5.0
        
        # Assign Meta
        df["trap_score"] = payload.get("trap_score", np.zeros(len(df)))
        
        # [Residual Legacy for Visualization if needed]
        df["feat_form"] = (df["high"] - df["low"]) / ((df["high"] - df["low"]).rolling(14).mean() + 1e-9)
        
        return df

    def compute_latent_vector(self, row: pd.Series) -> Dict[str, float]:
        """
        Extracts exactly the 12 Physics-Supremacy dimensions specified in neural_config.py.
        """
        def safe_get(key, default=0.0):
            val = row.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        # Mapping dict according to settings.NEURAL_FEATURE_NAMES (New Constitution)
        return {
            "temporal_sin": safe_get("temporal_sin"),
            "temporal_cos": safe_get("temporal_cos"),
            "killzone_intensity": safe_get("killzone_intensity"),
            "session_weight": safe_get("session_weight"),
            "structural_feat_index": safe_get("feat_index") / 100.0,
            "confluence_tensor": safe_get("confluence_score") / 5.0,
            "physics_force": safe_get("physics_force"),
            "physics_energy": safe_get("physics_energy"),
            "physics_entropy": safe_get("physics_entropy"),
            "physics_viscosity": safe_get("physics_viscosity"),
            "volatility_context": safe_get("volatility_context", 1.0), # Computed in process if mapped, or recomputed here
            "trap_score": safe_get("trap_score")
        }

    def tensorize_snapshot(self, snap: Dict, feature_names: List[str]) -> np.ndarray:
        return tensorize_snapshot(snap, feature_names)
