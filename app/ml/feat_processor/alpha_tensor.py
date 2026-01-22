import numpy as np
import pandas as pd
from typing import Dict, Any
from nexus_core.structure_engine.engine import StructureEngine
from app.ml.feat_processor.vectorized_tensor import VectorizedChronosProcessor
from app.ml.feat_processor.macro import MacroTensorFactory

from nexus_core.physics_engine.engine import physics_engine
from tqdm import tqdm

class AlphaTensorOrchestrator:
    """
    [DOCTORAL ALPHA CORE]
    The ultimate feature factory for Probabilistic Neural Networks.
    Orchestrates the 'Division of Structure', 'Chronos Core', and 'Physics Core'.
    """
    
    def __init__(self):
        self.chronos = VectorizedChronosProcessor()
        self.structure = StructureEngine()
        self.macro = MacroTensorFactory()
        
    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Processes a raw market history block into a multi-dimensional Alpha Tensor.
        Optimized for Scalping (M1/M5), Day Trading (H1), and Swing (H4+).
        """
        if df.empty: return {}

        # [CRITICAL DEFENSE] Ensure Unique Index
        # Duplicate timestamps cause critical failures in pd.concat/reindex
        if not df.index.is_unique:
             df = df[~df.index.duplicated(keep='last')]
        
        # Ensure Monotonic
        if not df.index.is_monotonic_increasing:
             df = df.sort_index()

        # 1. TEMPORAL & LIQUIDITY VECTORIZATION
        print("🧪 [Hydration] Phase 1: Temporal & Liquidity...")
        df = self.chronos.process(df)
        
        # [CRITICAL] Force clean index after chronos
        df = df.reset_index(drop=True)
        
        # 2. STRUCTURAL TOPOLOGY (Vectorized SMC)
        print("🧪 [Hydration] Phase 2: Structural Topology...")
        struct_res = self.structure.compute_feat_index(df)
        
        # [CRITICAL] Ensure struct_res has same length as df before concat
        struct_res = struct_res.reset_index(drop=True)
        if len(struct_res) != len(df):
            # Truncate to shorter length to force alignment
            min_len = min(len(struct_res), len(df))
            struct_res = struct_res.iloc[:min_len]
            df = df.iloc[:min_len]
        
        # Merge columns directly instead of concat (safer)
        for col in struct_res.columns:
            if col not in df.columns:
                df[col] = struct_res[col].values

        # 3. PHYSICS CORE (Vectorized Newton/Thermodynamics)
        print("🧪 [Hydration] Phase 3: Physics Core...")
        physics_res = physics_engine.compute_vectorized_physics(df)
        physics_res = physics_res.reset_index(drop=True)
        
        # Merge physics columns directly
        for col in physics_res.columns:
            if col not in df.columns:
                df[col] = physics_res[col].values

        # 4. PROBABILISTIC REGIME GATING (Vectorized Acceleration)
        print("🧪 [Hydration] Phase 4: Regime Gating...")
        
        atr = (df["high"] - df["low"]).rolling(14).mean().ffill()
        
        # [CRITICAL ALIGNMENT] Ensure variables share same index before math
        force = df["feat_force"].values
        killzone = df["killzone_intensity"].values
        entropy = df["physics_entropy"].values
        energy = df["physics_energy"].values
        confluence = df["confluence_score"].values
        viscosity = df["physics_viscosity"].values
        
        # Scale & Clip using Numpy (Bypass df.iloc overhead)
        raw_scalp = np.clip(force * killzone * entropy, 0, 1)
        raw_day = np.clip(energy * confluence * (1.0 - viscosity), 0, 1)
        
        if "struct_displacement_z" in df.columns:
            struct_disp = df["struct_displacement_z"].values
            raw_swing = np.clip(np.abs(struct_disp) * (1.0 - entropy), 0, 1)
        else:
            raw_swing = np.full(len(df), 0.1)
            
        df["p_scalp"] = raw_scalp
        df["p_daytrade"] = raw_day
        df["p_swing"] = raw_swing
            
        # Normalize Probabilities (Softmax-ish) - Use .values for safety
        p_scalp = np.array(df["p_scalp"].values).flatten()
        p_daytrade = np.array(df["p_daytrade"].values).flatten()
        p_swing = np.array(df["p_swing"].values).flatten()
        
        total_p = p_scalp + p_daytrade + p_swing + 1e-9
        
        df["p_scalp"] = p_scalp / total_p
        df["p_daytrade"] = p_daytrade / total_p
        df["p_swing"] = p_swing / total_p
            
        # 5. ASSEMBLE TENSOR PAYLOAD (Neural Ready)
        # Flattened for direct ingestion by MLEngine based on NEURAL_FEATURE_NAMES
        payload = {
            "temporal_sin": df["time_sin"].values.astype(np.float32),
            "temporal_cos": df["time_cos"].values.astype(np.float32),
            "killzone_intensity": df["killzone_intensity"].values.astype(np.float32),
            "session_weight": df["session_weight"].values.astype(np.float32),
            
            "structural_feat_index": df["feat_index"].values.astype(np.float32) / 100.0,
            "confluence_tensor": df["confluence_score"].values.astype(np.float32) / 5.0,
            
            "physics_force": df["feat_force"].values.astype(np.float32),
            "physics_energy": df["physics_energy"].values.astype(np.float32),
            "physics_entropy": df["physics_entropy"].values.astype(np.float32),
            "physics_viscosity": df["physics_viscosity"].values.astype(np.float32),
            
            "volatility_context": (df["close"] / (atr + 1e-9)).values.astype(np.float32),
            "trap_score": df.get("trap_score", pd.Series(0, index=df.index)).values.astype(np.float32),
            
            # --- META (Not Input) ---
            "regime_probability": df[["p_scalp", "p_daytrade", "p_swing"]].values.astype(np.float32)
        }
        
        return payload

    def get_live_alpha(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extraction for the current tick/candle for Live Execution.
        """
        payload = self.process_dataframe(df)
        # Extract the tail (last state)
        last_state = {k: v[-1] for k, v in payload.items()}
        
        # Add human-readable intent for the Risk Engine
        probs = last_state["regime_probability"]
        modes = ["SCALP", "DAY_TRADE", "SWING"]
        dominant_mode = modes[np.argmax(probs)]
        
        last_state["dominant_mode"] = dominant_mode
        last_state["execution_confidence"] = float(np.max(probs))
        
        return last_state
