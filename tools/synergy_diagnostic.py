import numpy as np
import pandas as pd
import time
from app.ml.feat_processor.spectral import SpectralTensorBuilder
from app.skills.volume_profile import volume_profile

def run_synergy_audit():
    print("=== FEAT SNIPER: CYBER-AUDIT (SYNERGY & AUCTION) ===")
    
    # 1. GENERATE DYNAMIC MARKET SCENARIO (Trending but Overextended)
    # Price is moving up (Physics) but Volume is staying at the bottom (Auction Trap)
    n = 100
    prices = np.linspace(2000, 2050, n) + np.random.normal(0, 1, n)
    # Volume is concentrated at the START (2000-2010), representing an "Auction Vacuum"
    volumes = np.random.uniform(50, 100, n)
    volumes[50:] = np.random.uniform(5, 15, 50) # Thin volume above
    
    df = pd.DataFrame({'close': prices, 'high': prices + 0.5, 'low': prices - 0.5, 'tick_volume': volumes})
    
    builder = SpectralTensorBuilder()
    
    print("\n[STEP 1: INDIVIDUAL SIGNAL AUDIT]")
    # Get Volume Signal
    vol_prof = volume_profile.get_profile(df)
    print(f"📊 Volume POC: {vol_prof['poc']:.2f} | Shape: {vol_prof['shape']}")
    
    # Get Spectral Signal
    tensors = builder.build_tensors(df)
    print(f"📐 Form (Domino): {tensors['domino_alignment']:.2f}")
    print(f"🚀 Acceleration (Energy): {tensors['energy_burst']:.2f}")
    print(f"🌌 Space (Elastic Gap): {tensors['elastic_gap']:.2f}")
    
    print("\n[STEP 2: SYNERGY VERIFICATION]")
    
    # SYNERGY 1: Value Area vs Elastic Gap
    price_now = prices[-1]
    is_outside_va = price_now > vol_prof['vah'] or price_now < vol_prof['val']
    is_extended = abs(tensors['elastic_gap']) > 0.8 # Empirical threshold
    
    if is_outside_va and is_extended:
        print("🔍 CHECK: [SPACE SYNERGY] - CORRECT. Both Physics and Volume detect Overextension.")
    else:
        print("⚠️ CHECK: [SPACE SYNERGY] - CALIBRATION NEEDED. Signal discrepancy detected.")
        
    # SYNERGY 2: Energy Burst vs Volume Shape
    is_p_shape = "P-Shape" in vol_prof['shape']
    is_high_energy = tensors['energy_burst'] > 1.0
    
    if is_p_shape and is_high_energy:
        print("🔍 CHECK: [ACCELERATION SYNERGY] - CORRECT. Bullish Momentum confirmed by Auction Shape.")
    else:
        print("🔍 CHECK: [ACCELERATION SYNERGY] - NEUTRAL. No momentum explosion detected.")

    print("\n[STEP 3: AUCTION-PHYSICS DIVERGENCE (APD)]")
    # APD = abs(Price - POC) normalized by Vol_Scalar
    apd = abs(price_now - vol_prof['poc']) / (vol_prof['vah'] - vol_prof['val'] + 1e-9)
    print(f"🛡️ Synergy Score (APD): {apd:.2f}")
    if apd > 2.0:
        print("🔴 ALERT: High Divergence! Price is floating without auction support (Vacuum).")
    else:
        print("🟢 STATUS: Price is supported by current auction liquidity.")

if __name__ == "__main__":
    run_synergy_audit()
