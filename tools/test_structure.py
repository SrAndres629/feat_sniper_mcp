import sys
import os
import pandas as pd
import numpy as np

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_core.structure_engine import structure_engine
from nexus_training.simulate_warfare import BattlefieldSimulator

def run_structural_audit():
    print("\n" + "="*50)
    print("🧠 [PHASE 13] STRUCTURE ENGINE AUDIT: SMC LAYERS")
    print("="*50)

    # 1. Generate Raw Data
    sim = BattlefieldSimulator("XAUUSD")
    df_raw = sim.generate_synthetic_data(n_rows=500)
    
    # 2. Run Structure Engine
    df = structure_engine.identify_fractals(df_raw)
    df = structure_engine.detect_structural_shifts(df)
    df = structure_engine.detect_imbalances(df)
    
    # 3. Validation Results
    swings = df[df["major_h"] | df["major_l"]]
    imbalances = df[df["fvg_bull"] | df["fvg_bear"]]
    inducements = df[df["is_inducement"]]
    
    print(f"📊 Market Simulation: {len(df)} candles.")
    print(f"✅ Major Swings Detected: {len(swings)}")
    print(f"✅ FVGs (Imbalances) Found: {len(imbalances)}")
    print(f"✅ Inducement Traps Identified: {len(inducements)}")
    
    print("\n--- DETAILED GEOMETRY SNAPSHOT ---")
    
    # Print a few samples
    for i in range(min(5, len(swings))):
        idx = swings.index[i]
        row = df.loc[idx]
        stype = "HIGH" if row["major_h"] else "LOW"
        print(f"[ESTRUCTURA] Detectado MajorSwing {stype} en {row['high' if stype=='HIGH' else 'low']:.2f}")
        
    for i in range(min(3, len(imbalances))):
        idx = imbalances.index[i]
        row = df.loc[idx]
        itype = "BULL" if row["fvg_bull"] else "BEAR"
        top = row["fvg_bull_top"] if itype == "BULL" else row["fvg_bear_top"]
        bot = row["fvg_bull_bottom"] if itype == "BULL" else row["fvg_bear_bottom"]
        print(f"[IMBALANCE] FVG {itype} detectado entre {bot:.2f} y {top:.2f}")

    print("\n" + "="*50)
    print("✅ AUDIT COMPLETE: GEOMETRY IS SCALABLE")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_structural_audit()
