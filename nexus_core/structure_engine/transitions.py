from .fractals import identify_fractals
from .sessions import identify_trading_session, get_session_weight
from app.skills.volume_profile import volume_profile

def detect_structural_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.0 - SMC DOCTORAL] Precise BOS/CHOCH Detection.
    Factors: Temporal Session, Volume POC Shift, and Hierarchical Fractals.
    """
    df = identify_fractals(df)

    # 1. Track Hierarchical Levels
    df["swing_h"] = df["high"].where(df["major_h"]).ffill()
    df["swing_l"] = df["low"].where(df["major_l"]).ffill()
    df["internal_h"] = df["high"].where(df["minor_h"]).ffill()
    df["internal_l"] = df["low"].where(df["minor_l"]).ffill()

    # 2. TEMPORAL WEIGHTING (Rule 1: Time is Law)
    df["session_type"] = df.index.map(lambda x: identify_trading_session(x))
    df["session_weight"] = df["session_type"].map(get_session_weight)

    # 3. VOLUME TRUTH (Rule 2: POC Shift)
    # Get current POC from Volume Profile
    profile = volume_profile.get_profile(df, bins=50)
    poc_price = profile.get("poc_price", df["close"].iloc[-1])
    df["poc_price"] = poc_price

    # 4. SWING BOS (Structural Confirmation)
    # Rule: Candle Close > Previous Swing High + Validated by Session & POC
    raw_bos_bull = (df["close"] > df["swing_h"].shift(1)) & (df["close"].shift(1) <= df["swing_h"].shift(1))
    raw_bos_bear = (df["close"] < df["swing_l"].shift(1)) & (df["close"].shift(1) >= df["swing_l"].shift(1))
    
    # Validation: POC must be trending in BOS direction to avoid "Low Volume Fakeouts"
    # For Bullish BOS: Current Price must be > POC
    df["bos_bull"] = raw_bos_bull & (df["close"] > df["poc_price"])
    df["bos_bear"] = raw_bos_bear & (df["close"] < df["poc_price"])

    # BOS Strength calculation for AI Ingest
    df["bos_strength"] = df["session_weight"] * (1.2 if (df["bos_bull"] | df["bos_bear"]) else 0.0)

    # 5. INTERNAL CHOCH (Change of Character)
    df["choch_bull"] = (df["close"] > df["internal_h"].shift(1)) & (df["swing_h"] > df["swing_h"].shift(5))
    df["choch_bear"] = (df["close"] < df["internal_l"].shift(1)) & (df["swing_l"] < df["swing_l"].shift(5))
    
    # 7. RANGE EQUILIBRIUM (Premium vs Discount)
    # 0.0 = Bottom, 1.0 = Top, 0.5 = Equilibrium
    df["range_h"] = df["high"].where(df["major_h"]).ffill()
    df["range_l"] = df["low"].where(df["major_l"]).ffill()
    
    # Avoid division by zero
    range_dist = (df["range_h"] - df["range_l"]) + 1e-9
    df["range_pos"] = (df["close"] - df["range_l"]) / range_dist
    
    # Premium (> 0.5) vs Discount (< 0.5)
    df["in_discount"] = df["range_pos"] < 0.5
    df["in_premium"] = df["range_pos"] > 0.5

    # 8. REFINED BOS STRENGTH
    # An institutional BOS is most 'sincere' when it originates from the opposite side of the range
    # Long BOS starting from Discount is Strong (1.5x)
    # Long BOS at Premium Top is Weak/Exhaustive (0.5x)
    df["bos_strength"] = df["session_weight"] * (1.2 if (df["bos_bull"] | df["bos_bear"]) else 0.0)
    
    # Conditional boosting based on PD Arrays
    df.loc[df["bos_bull"] & df["in_discount"], "bos_strength"] *= 1.5
    df.loc[df["bos_bear"] & df["in_premium"], "bos_strength"] *= 1.5

    return df
