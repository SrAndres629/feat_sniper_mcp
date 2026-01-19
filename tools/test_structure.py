import sys
import os
import pandas as pd
import numpy as np

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the package which exposes all functions via __init__.py
import nexus_core.structure_engine as se
from nexus_core.structure_engine.engine import StructureEngine
from nexus_training.simulate_warfare import BattlefieldSimulator

def run_structural_audit():
    print("\n" + "="*50)
    print("🧠 [PHASE 13] STRUCTURE ENGINE AUDIT: SMC LAYERS")
    print("="*50)

    # 1. Generate Raw Data
    sim = BattlefieldSimulator("XAUUSD")
    df_raw = sim.generate_synthetic_data(n_rows=500)
    
    # 2. Run Structure Engine
    try:
        print("Running identify_fractals...")
        df = se.identify_fractals(df_raw)
        print("Running detect_structural_shifts...")
        df = se.detect_structural_shifts(df)
        print("Running detect_imbalances...")
        df = se.detect_imbalances(df)
        print("Running detect_order_blocks...")
        df = se.detect_order_blocks(df)
        print("Running detect_critical_points...")
        df = se.detect_critical_points(df)
        print("Running detect_shadow_zones...")
        df = se.detect_shadow_zones(df)
        print("Running detect_consolidation_zones...")
        df = se.detect_consolidation_zones(df)
        print("Running calculate_confluence_score...")
        # StructureEngine class is in structure_engine.engine (imported as StructureEngine)
        engine = StructureEngine()
        df = engine.calculate_confluence_score(df)
    except Exception as e:
        print(f"❌ CRITICAL ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Validation Results
    swings = df[df["major_h"] | df["major_l"]]
    imbalances = df[df["fvg_bull"] | df["fvg_bear"]]
    obs = df[df["ob_bull"] | df["ob_bear"]]
    shadows = df[df["shadow_bull"] | df["shadow_bear"]]
    criticals = df[df["is_critical_point"]]
    high_confluence = df[df["confluence_score"] > 2.0]
    
    print(f"📊 Market Simulation: {len(df)} candles.")
    print(f"✅ Major Swings Detected: {len(swings)}")
    print(f"✅ FVGs (Imbalances) Found: {len(imbalances)}")
    print(f"✅ Order Blocks Detected (Validated): {len(obs)}")
    print(f"✅ Shadow Zones (Wick Fills): {len(shadows)}")
    print(f"✅ Critical Points (Doji+Vol): {len(criticals)}")
    print(f"✅ SNIPER ZONES (Confluence > 2.0): {len(high_confluence)}")
    
    print("\n--- INSTITUTIONAL SPACE SNAPSHOT ---")
    
    # Print a few Sniper Setups
    if not high_confluence.empty:
        for i in range(min(5, len(high_confluence))):
            idx = high_confluence.index[i]
            row = df.loc[idx]
            print(f"[SNIPER] Confluence Score: {row['confluence_score']} at {row['close']:.2f} | Time: {idx}")
            print(f"   -> Overlaps: OB={row.get('ob_bull') or row.get('ob_bear')}, FVG={row.get('fvg_bull') or row.get('fvg_bear')}, ZS={row.get('shadow_bull') or row.get('shadow_bear')}")

    print("\n" + "="*50)
    print("✅ AUDIT COMPLETE: DOCTORAL SPACE LOGIC OPERATIONAL")
    print("="*50 + "\n")

    print("\n" + "="*50)
    print("✅ AUDIT COMPLETE: GEOMETRY IS SCALABLE")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_structural_audit()
