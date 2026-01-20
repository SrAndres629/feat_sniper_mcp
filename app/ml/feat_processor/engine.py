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
        if df.empty: return pd.DataFrame() # Consistent Empty DF return

        # --- [FIX] SANITIZE DATA INTEGRITY ---
        # 1. Reset index to remove duplicates (Fixes 'cannot reindex' error)
        df = df.reset_index(drop=True)
        
        # 2. Ensure we have a clean numerical index for vector operations
        if not isinstance(df.index, pd.RangeIndex):
            df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
        # -------------------------------------

        # We calculate the full Alpha Tensor payload which includes Structure, Chronos, and Physics.
        # But for 'df' enrichment we need to flatten it back to columns.
        
        # 1. Run Standard Feature Engineering (Legacy Support + Indicators)
        df = apply_feat_engineering(df)
        
        # 2. Run The Alpha Core (Physics, Structure, Time)
        # Note: AlphaTensorOrchestrator returns a dict of arrays. We need to assign them to DF.
        # Ideally, we should use AlphaTensorOrchestrator's internal components directly or map payload back.
        
        # Mapping payload back is safest to guarantee 1:1 match.
        payload = self.alpha.process_dataframe(df)
        
        # [CRITICAL] Safe Broadcast Helper - Handles scalar/array length mismatch
        def safe_assign(col_name, default_val=0.0):
            val = payload.get(col_name, default_val)
            
            # If scalar or single element, broadcast to full column
            if np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1):
                return np.full(len(df), float(val) if np.isscalar(val) else float(val.item()))
            
            # If array, ensure length matches
            if isinstance(val, (list, np.ndarray, pd.Series)):
                arr = np.array(val).flatten()
                if len(arr) != len(df):
                    # Resize to match (emergency padding/truncation)
                    arr = np.resize(arr, len(df))
                return arr
            
            # Fallback
            return np.full(len(df), float(default_val))
        
        # Assign Physics Tensors (Broadcast-Safe)
        df["physics_force"] = safe_assign("physics_force", 0.0)
        df["physics_energy"] = safe_assign("physics_energy", 0.0)
        df["physics_entropy"] = safe_assign("physics_entropy", 0.0)
        df["physics_viscosity"] = safe_assign("physics_viscosity", 0.0)
        
        # Assign Temporal
        df["temporal_sin"] = safe_assign("temporal_sin", 0.0)
        df["temporal_cos"] = safe_assign("temporal_cos", 0.0)
        df["killzone_intensity"] = safe_assign("killzone_intensity", 0.0)
        df["session_weight"] = safe_assign("session_weight", 0.0)
        
        # Assign Structural
        df["structural_feat_index"] = safe_assign("structural_feat_index", 0.0)
        df["confluence_tensor"] = safe_assign("confluence_tensor", 0.0)
        
        # Assign Meta
        df["volatility_context"] = safe_assign("volatility_context", 1.0)
        df["trap_score"] = safe_assign("trap_score", 0.0)
        
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

        # Mapping dict according to standardized column names in process_dataframe
        return {
            "temporal_sin": safe_get("temporal_sin"),
            "temporal_cos": safe_get("temporal_cos"),
            "killzone_intensity": safe_get("killzone_intensity"),
            "session_weight": safe_get("session_weight"),
            "structural_feat_index": safe_get("structural_feat_index"),
            "confluence_tensor": safe_get("confluence_tensor"),
            "physics_force": safe_get("physics_force"),
            "physics_energy": safe_get("physics_energy"),
            "physics_entropy": safe_get("physics_entropy"),
            "physics_viscosity": safe_get("physics_viscosity"),
            "volatility_context": safe_get("volatility_context", 1.0),
            "trap_score": safe_get("trap_score")
        }

    def tensorize_snapshot(self, snap: Dict, feature_names: List[str]) -> np.ndarray:
        return tensorize_snapshot(snap, feature_names)
