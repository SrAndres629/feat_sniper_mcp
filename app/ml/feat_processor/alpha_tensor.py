import numpy as np
import pandas as pd
from typing import Dict, Any
from nexus_core.structure_engine.engine import StructureEngine
from app.ml.feat_processor.vectorized_tensor import VectorizedChronosProcessor
from app.ml.feat_processor.macro import MacroTensorFactory

from nexus_core.kinetic_engine import kinetic_engine

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
        df = self.chronos.process(df)
        
        # 2. STRUCTURAL TOPOLOGY (Vectorized SMC)
        struct_res = self.structure.compute_feat_index(df)
        df = pd.concat([df, struct_res], axis=1)

        # 3. PHYSICS CORE (Vectorized Newton/Thermodynamics)
        physics_res = kinetic_engine.compute_vectorized_physics(df)
        df = pd.concat([df, physics_res], axis=1)

        # 4. PROBABILISTIC REGIME GATING (Physics-Driven Heuristic Head)
        # This simulates the "Neural Output" for the Risk Engine using fundamental laws.
        # In the future, this is replaced by the actual Model Inference.
        
        atr = (df["high"] - df["low"]).rolling(14).mean().ffill()
        
        # P(Scalp): High Force (F=ma) + High Entropy (Volatility) + Killzone
        # Logic: Scalping needs violence (Force) and Opportunity (Entropy/Killzone)
        
        # [CRITICAL ALIGNMENT] Ensure variables share same index before math
        # [NUCLEAR OPTION] Force Flattened Numpy Arrays to bypass Pandas Checks
        force = np.array(df["feat_force"].values).flatten()
        killzone = np.array(df["killzone_intensity"].values).flatten()
        entropy = np.array(df["physics_entropy"].values).flatten()
        
        # Ensure shapes match (min length)
        min_len = min(len(force), len(killzone), len(entropy))
        raw_scalp = force[:min_len] * killzone[:min_len] * entropy[:min_len]
        
        df["p_scalp"] = 0.0
        df.iloc[:min_len, df.columns.get_loc("p_scalp")] = np.clip(raw_scalp, 0, 1)
        
        # P(DayTrade): Sustained Energy + Structural Confluence - Viscosity
        # Logic: Day trading needs clean moves (Low Viscosity) and Structure
        energy = np.array(df["physics_energy"].values).flatten()
        confluence = np.array(df["confluence_score"].values).flatten()
        viscosity = np.array(df["physics_viscosity"].values).flatten()
        
        min_len_day = min(len(energy), len(confluence), len(viscosity))
        raw_day = energy[:min_len_day] * confluence[:min_len_day] * (1.0 - viscosity[:min_len_day])
        
        df["p_daytrade"] = 0.0
        df.iloc[:min_len_day, df.columns.get_loc("p_daytrade")] = np.clip(raw_day, 0, 1)
        
        # P(Swing): Massive Structural Displacement + Low Entropy (Order) + Macro Alignment
        # Logic: Swing needs Order (Low Entropy) and major Structural/Macro shifts
        if "struct_displacement_z" in df.columns:
            # Low Entropy favored for Swing Entry (Accumulation)
            entropy_factor = 1.0 - df["physics_entropy"]
            raw_swing = (df["struct_displacement_z"].abs() * entropy_factor)
            df["p_swing"] = raw_swing.clip(0, 1)
        else:
            df["p_swing"] = 0.1
            
        # Normalize Probabilities (Softmax-ish)
        total_p = df["p_scalp"] + df["p_daytrade"] + df["p_swing"] + 1e-9
        df["p_scalp"] /= total_p
        df["p_daytrade"] /= total_p
        df["p_swing"] /= total_p
            
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
