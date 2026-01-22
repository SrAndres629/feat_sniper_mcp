import pandas as pd
import numpy as np
import logging
from .sessions import identify_trading_session

logger = logging.getLogger("feat.trap_guard")

def calculate_trap_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v6.0 - DOCTORAL REFINEMENT] Institutional vs Retail Trap Analysis.
    Philosophy: Penalize 'Retail Obviousness', Protect 'Institutional Footprints'.
    """
    if df.empty: return df
    df = df.copy()
    
    # Initialize
    df["trap_score"] = 0.0
    df["trap_reason"] = ""
    df["hunter_mode"] = False
    
    # --- 1. RETAIL OBVIOUSNESS (The Bait) ---
    retail_bait = pd.Series(0.0, index=df.index)
    if "rsi" in df.columns:
        is_retail_extrema = (df["rsi"] > 70) | (df["rsi"] < 30)
        retail_bait += is_retail_extrema.astype(float) * 1.0
    
    # --- 2. INSTITUTIONAL FOOTPRINTS (The Sincerity) ---
    inst_sincerity = pd.Series(0.0, index=df.index)
    if "ob_bull" in df.columns and "ob_bear" in df.columns:
        # A fresh OB mitigation is sincere institutional flow (State 2 = Mitigated)
        is_mitigating = ((df["ob_bull"] == 2.0) | (df["ob_bear"] == 2.0))
        inst_sincerity += is_mitigating.astype(float) * 2.0 # High priority
    
    if "fvg_gravity" in df.columns:
        has_fvg = (df["fvg_gravity"].abs() > 0)
        inst_sincerity += has_fvg.astype(float) * 1.0

    # --- 3. LIQUIDITY HUNTER MODE (EQH/EQL) ---
    if "is_eqh" in df.columns or "is_eql" in df.columns:
        is_eq = (df.get("is_eqh", False) | df.get("is_eql", False))
        df["hunter_mode"] = is_eq
        df.loc[is_eq, "trap_score"] += 0.5
        df.loc[is_eq, "trap_reason"] = "Equal Levels Detected (Liquidity Trap); "

    # --- FINAL SCORE CALCULATION ---
    # Simplified vectorized calc
    base_trap = (retail_bait * 0.3) - (inst_sincerity * 0.2)
    df["trap_score"] = (df["trap_score"] + base_trap).clip(0, 1)

    return df

def get_trap_report(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    trap_score = last.get("trap_score", 0.0)
    
    if trap_score > 0.8:
        verdict = "🚨 DEFINITE TRAP"
    elif trap_score > 0.6:
        verdict = "⚠️ HIGH RISK"
    else:
        verdict = "🟢 LOW RISK"
    
    return {
        "trap_score": round(float(trap_score), 2),
        "verdict": verdict,
        "session": last.get("session_type", "UNKNOWN")
    }
