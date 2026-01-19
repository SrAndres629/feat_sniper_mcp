import pandas as pd
import numpy as np
import logging
from .sessions import identify_trading_session

logger = logging.getLogger("feat.trap_guard")

def calculate_trap_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.1 - PREDATORY REFINEMENT] Institutional vs Retail Trap Analysis.
    Philosophy: Penalize 'Retail Obviousness', Protect 'Institutional Footprints'.
    """
    if df.empty: return df
    df = df.copy()
    
    # Initialize
    df["trap_score"] = 0.0
    df["trap_reason"] = ""
    df["hunter_mode"] = False # Flag for "Wait for Sweep"
    
    # --- 1. RETAIL OBVIOUSNESS (The Bait) ---
    # Penalize indicators that retail masses use
    retail_bait = 0
    # Logic: If RSI is overbought/oversold while at an obvious level
    if "rsi" in df.columns:
        is_retail_extrema = (df["rsi"] > 70) | (df["rsi"] < 30)
        retail_bait += is_retail_extrema.astype(int)
    
    # Obvious Support/Resistance (Tested 3+ times)
    if "test_count" in df.columns:
        is_obvious_level = df["test_count"] >= 3
        retail_bait += is_obvious_level.astype(int)
        
    # --- 2. INSTITUTIONAL FOOTPRINTS (The Sincerity) ---
    # We DO NOT penalize these; they reduce the trap score
    inst_sincerity = 0
    if "ob_bull" in df.columns and "ob_bear" in df.columns:
        # A fresh OB mitigation is sincere institutional flow
        is_mitigating = (df["ob_bull"] | df["ob_bear"]) & (~df.get("is_mitigated", False))
        inst_sincerity += is_mitigating.astype(int)
    
    if "fvg_bull" in df.columns or "fvg_bear" in df.columns:
        has_fvg = (df.get("fvg_bull", False) | df.get("fvg_bear", False))
        inst_sincerity += has_fvg.astype(int)

    # --- 3. LIQUIDITY HUNTER MODE (EQH/EQL) ---
    # If Equal Highs exist, VETO selling and wait for sweep
    if "is_eqh" in df.columns or "is_eql" in df.columns:
        is_eq = (df.get("is_eqh", False) | df.get("is_eql", False))
        df["hunter_mode"] = is_eq
        df.loc[is_eq, "trap_score"] += 0.5
        df.loc[is_eq, "trap_reason"] += "Equal Levels Detected (Liquidity Trap); "

    # --- 4. TEMPORAL BAIT ---
    df["session_type"] = df.index.map(lambda x: identify_trading_session(x))
    is_dead_zone = df["session_type"].isin(["ASIA", "NY_LATE"])
    
    # --- FINAL SCORE CALCULATION ---
    # Base Trap Score from Retail Bait - Institutional Sincerity
    base_trap = (retail_bait * 0.3) - (inst_sincerity * 0.2)
    df["trap_score"] = (df["trap_score"] + base_trap).clip(0, 1)

    # Log trigger for the user
    last = df.iloc[-1]
    if last["trap_score"] > 0.6:
        reason = last["trap_reason"] or "Excessive Retail Confirmation"
        logger.warning(f"[TRAP GUARD] Setup rejected. Trap Score: {last['trap_score']:.2f} (Motivo: {reason})")

    return df

def get_trap_report(df: pd.DataFrame) -> dict:
    """
    Returns a human-readable report of the trap analysis.
    """
    last = df.iloc[-1]
    trap_score = last.get("trap_score", 0.0)
    
    if trap_score > 0.8:
        verdict = "🚨 DEFINITE TRAP - DO NOT ENTER"
    elif trap_score > 0.6:
        verdict = "⚠️ HIGH TRAP PROBABILITY - EXTREME CAUTION"
    elif trap_score > 0.4:
        verdict = "🟡 MODERATE TRAP RISK - WAIT FOR SWEEP"
    else:
        verdict = "🟢 LOW TRAP RISK - PROCEED WITH LOGIC"
    
    return {
        "trap_score": round(float(trap_score), 2),
        "verdict": verdict,
        "is_obvious_setup": bool(last.get("is_obvious_setup", False)),
        "session": last.get("session_type", "UNKNOWN")
    }
