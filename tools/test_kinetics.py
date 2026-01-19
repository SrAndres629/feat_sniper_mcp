import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nexus_core.kinetic_engine import kinetic_engine

def verify_kinetics():
    print("==================================================")
    print("🧪 [KINETICS] STATE MACHINE VALIDATION")
    print("==================================================")
    
    # Base params
    base_price = 100.0
    atr = 1.0
    
    # Create a history of 20 candles (Neutral)
    history = []
    for i in range(20):
        history.append({
            "open": base_price, "close": base_price + 0.5, 
            "high": base_price + 1.0, "low": base_price - 0.5,
            "volume": 1000.0
        })
        base_price = history[-1]["close"]
        
    # --- SIMULATION LOOP ---
    
    # 1. IMPULSE EVENT (T=20)
    # Body = 3.0 (3x Average). High Vol.
    impulse_candle = {
        "open": base_price, "close": base_price + 3.0, # Bull Impulse
        "high": base_price + 3.2, "low": base_price - 0.2,
        "volume": 3000.0 # High Vol -> Force > 2.0
    }
    history.append(impulse_candle)
    base_price = impulse_candle["close"]
    
    # Analyze State
    df_imp = pd.DataFrame(history)
    metrics_imp = kinetic_engine.compute_kinetic_state(df_imp)
    
    print(f"\n[T=0] Impulse Candle Detectada.")
    print(f"   FEAT Force: {metrics_imp.get('feat_force', 0):.2f}")
    print(f"   Estado Raw: {metrics_imp.get('absorption_state', 0)}")
    # Note: State might be Neutral if loop logic in validator is strict about "looking back".
    # Validator checks T-1 to T-5. Our impulse is at T (last).
    # If the validator only looks at *past* impulses, it won't trigger yet?
    # Let's check validator logic: "Check T (current) back to T-3". 
    # Yes, it checks current 'candle' in fallback loop.
    
    # 2. MONITORING 1 (T=21)
    # Small candle, holding > 50% of impulse body (Body=3.0, 50%=1.5)
    # Impulse Open=110, Close=113. Limit = 110 + 1.5 = 111.5.
    c1 = {
        "open": base_price, "close": base_price - 0.5, # Retrace but above 111.5
        "high": base_price + 0.2, "low": base_price - 0.6,
        "volume": 1000.0
    }
    history.append(c1)
    
    df_m1 = pd.DataFrame(history)
    metrics_m1 = kinetic_engine.compute_kinetic_state(df_m1)
    st_val = metrics_m1.get('absorption_state', 0)
    st_str = "MONITORING" if st_val == 2.0 else str(st_val)
    print(f"[T+1] Candle close {c1['close']:.2f}. Limit > 111.5. State: {st_str}")
    
    # 3. MONITORING 2 (T=22)
    c2 = {
        "open": c1["close"], "close": c1["close"] + 0.2,
        "high": c1["close"] + 0.5, "low": c1["close"] - 0.2,
        "volume": 1000.0
    }
    history.append(c2)
    df_m2 = pd.DataFrame(history)
    metrics_m2 = kinetic_engine.compute_kinetic_state(df_m2)
    print(f"[T+2] Candle close {c2['close']:.2f}. State: {metrics_m2.get('absorption_state', 0)}")

    # 4. MONITORING 3 (T=23) -> CONFIRMED
    c3 = {
        "open": c2["close"], "close": c2["close"] + 0.2,
        "high": c2["close"] + 0.5, "low": c2["close"] - 0.2,
        "volume": 1000.0
    }
    history.append(c3)
    df_m3 = pd.DataFrame(history)
    metrics_m3 = kinetic_engine.compute_kinetic_state(df_m3)
    
    final_state = metrics_m3.get('absorption_state', 0)
    state_label = "UNKNOWN"
    if final_state == 3.0: state_label = "CONFIRMED"
    
    print(f"[T+3] Candle close {c3['close']:.2f}. State: {state_label} ({final_state})")
    
    if final_state == 3.0:
        print("\n✅ SUCCESS: Absorción Exitosa Detectada.")
    else:
        print("\n❌ FAILURE: No se confirmó la absorción.")

if __name__ == "__main__":
    verify_kinetics()
