import numpy as np
import pandas as pd
from app.ml.feat_processor.spectral import SpectralTensorBuilder
from nexus_core.convergence_engine import convergence_engine

def verify_gravity_binding():
    print("=== OPERATION GRAVITY BINDING: HOLLOW RALLY AUDIT ===")
    
    # 1. GENERATE HOLLOW RALLY (Vacuum Scenario)
    # Price jumps from 2000 to 2050 in the last 10 bars.
    # But 90% of Volume is left behind at 2000.
    n = 100
    prices = np.concatenate([
        np.linspace(2000, 2005, 90), # Base accumulation
        np.linspace(2005, 2050, 10)  # Sudden "Hollow" breakout
    ])
    
    # Volume is massive at the base, tiny at the top
    volumes = np.concatenate([
        np.random.uniform(500, 1000, 90), 
        np.random.uniform(5, 10, 10)
    ])
    
    df = pd.DataFrame({
        'close': prices, 
        'high': prices + 1, 
        'low': prices - 1, 
        'tick_volume': volumes
    })
    
    builder = SpectralTensorBuilder()
    tensors = builder.build_tensors(df)
    
    print("\n[SCENARIO: SUDDEN BREAKOUT WITHOUT VOLUME]")
    print(f"💰 Price Now: {prices[-1]:.2f}")
    print(f"📐 Elastic Gap: {tensors['elastic_gap']:.2f} (Price vs Sniper)")
    print(f"🧲 SGI Gravity: {tensors['sgi_gravity']:.2f} (PVP vs Ribbon)")
    print(f"🛡️ APD Divergence: {tensors['auction_physics_divergence']:.2f}")
    
    # EXPECTED: SGI should be negative (PVP is far below) while Elastic Gap is positive
    is_hollow = convergence_engine.detect_hollow_rally(
        tensors['sgi_gravity'], 
        tensors['elastic_gap']
    )
    
    print(f"\n[DIAGNOSTIC]")
    if is_hollow:
        print("✅ SUCCESS: 'Hollow Rally' (Vacuum) DETECTED. The Grave Index flagged the lack of mass support.")
    else:
        print("❌ FAILURE: System failed to detect the Vacuum Divergence.")
        
    print(f"🔥 VAM Purity: {tensors['vam_purity']:.2f} (Speed adjusted by Density)")
    # VAM should be low relative to the price move because density at the top is low

if __name__ == "__main__":
    verify_gravity_binding()
