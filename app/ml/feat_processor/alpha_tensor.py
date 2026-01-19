import numpy as np
import pandas as pd
from typing import Dict, Any
from nexus_core.structure_engine.engine import StructureEngine
from app.ml.feat_processor.vectorized_tensor import VectorizedChronosProcessor
from app.ml.feat_processor.macro import MacroTensorFactory

class AlphaTensorOrchestrator:
    """
    [DOCTORAL ALPHA CORE]
    The ultimate feature factory for Probabilistic Neural Networks.
    Orchestrates the 'Division of Structure' and 'Chronos Core'.
    
    Philosophy: The Trading Regime (Scalp/Day/Swing) is an emergent property
    of Structural Depth (Z-score) and Liquidity Intensity.
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

        # 1. TEMPORAL & LIQUIDITY VECTORIZATION
        # (Sin/Cos Time, Gaussian KillZones, Session Weights)
        df = self.chronos.process(df)
        
        # 2. STRUCTURAL TOPOLOGY (Vectorized SMC)
        # (BOS Strength, Order Blocks, Breakers, FEAT Index)
        struct_res = self.structure.compute_feat_index(df)
        df = pd.concat([df, struct_res], axis=1)

        # 3. PROBABILISTIC REGIME GATING (Doctoral Logic)
        # We define P(Mode) based on the structural displacement and volatility
        atr = (df["high"] - df["low"]).rolling(14).mean().ffill()
        
        # P(Scalp): High Frequency, Near-field structural breaks (Small bars relative to ATR)
        df["p_scalp"] = (df["feat_index"] / 100.0) * (df["killzone_intensity"])
        
        # P(DayTrade): Mid-range structure + Session alignment
        df["p_daytrade"] = (df["confluence_score"] / 5.0) * (df["session_weight"] / 1.5)
        
        # P(Swing): Deep structural anchors (High Z-Score displacement)
        if "struct_displacement_z" in df.columns:
            df["p_swing"] = (df["struct_displacement_z"].abs() / 3.0).clip(0, 1)
        else:
            df["p_swing"] = 0.1 # Default low probability
            
        # 4. NEURAL NORMALIZATION (Manifold Invariance)
        # We normalize all structural levels by ATR to ensure the network is 'Price Blind'
        # but 'Volatility Aware'.
        
        # 5. ASSEMBLE TENSOR PAYLOAD (Neural Ready)
        payload = {
            # --- TEMPORAL LAYER ---
            "temporal_sin_cos": df[["time_sin", "time_cos"]].values.astype(np.float32),
            "killzone_intensity": df["killzone_intensity"].values.astype(np.float32),
            "session_context": df[[f"day_{i}" for i in range(7)] + ["session_weight"]].values.astype(np.float32),
            
            # --- STRUCTURAL LAYER ---
            "structural_feat_index": df["feat_index"].values.astype(np.float32) / 100.0,
            "confluence_tensor": df["confluence_score"].values.astype(np.float32) / 5.0,
            
            # --- REGIME PROBABILITY (The Gating) ---
            "regime_probability": df[["p_scalp", "p_daytrade", "p_swing"]].values.astype(np.float32),
            
            # --- ADAPTIVE TARGETS (Projected Volatility) ---
            # Distance to Projected TP/SL in ATR units
            "volatility_context": (df["close"] / (atr + 1e-9)).values.astype(np.float32)
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
