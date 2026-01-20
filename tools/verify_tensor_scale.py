import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from app.ml.strategic_cortex.state_encoder import state_encoder, StateVector
from nexus_core.features import feat_features

async def main():
    print("🔬 FEAT SNIPER: TENSOR SCALE AUDIT")
    print("====================================")
    
    # Mock data to simulate complex inputs
    dummy_account = {"balance": 20.0, "phase_name": "SURVIVAL"}
    dummy_micro = {"ofi_z_score": 1.5, "entropy_score": 0.8, "hurst": 0.6, "spread_normalized": 1.2}
    dummy_probs = {"scalp": 0.7, "day": 0.4, "swing": 0.2}
    dummy_physics = {"feat_composite": 75.0, "titanium": "TITANIUM_SUPPORT", "acceleration": 0.3}
    
    # Create Temporal Physics Map (Simulating unnormalized vs normalized)
    tpd = {}
    target_tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
    
    for tf in target_tfs:
        tpd[f"{tf}_direction"] = 1.0 # Buy bias
        # We test if the encoder properly clamps/scales these
        label = "energy"
        if tf == "W1":
            tpd[f"{tf}_{label}"] = 500.0 # Extreme unnormalized energy
        else:
            tpd[f"{tf}_{label}"] = 1.2
            
        tpd[f"{tf}_accel"] = 0.05
        
    state = state_encoder.encode(
        account_state=dummy_account,
        microstructure=dummy_micro,
        neural_probs=dummy_probs,
        physics_state=dummy_physics,
        fractal_coherence=0.8,
        temporal_physics_dict=tpd
    )
    
    tensor = state.to_tensor()
    
    print(f"\n✅ Tensor Dimension: {len(tensor)}")
    print(f"✅ Max Value: {np.max(tensor)}")
    print(f"✅ Min Value: {np.min(tensor)}")
    
    # Check groups
    groups = {
        "Account (0-3)": tensor[0:4],
        "Micro (4-7)": tensor[4:8],
        "Neural (8-11)": tensor[8:12],
        "Physics (12-15)": tensor[12:16],
        "Sentiment (16-19)": tensor[16:20],
        "Fractal (20)": tensor[20],
        "Temporal Physics (21-44)": tensor[21:45]
    }
    
    for name, vals in groups.items():
        if isinstance(vals, np.ndarray):
            print(f"- {name:25} | Range: [{np.min(vals):>7.2f}, {np.max(vals):>7.2f}] | Mean: {np.mean(vals):.4f}")
        else:
            print(f"- {name:25} | Value: {vals:>7.2f}")

    # CRITICAL CHECK: Contrast M1 vs W1 in Temporal Physics
    tp_tensor = tensor[21:45] 
    m1_phys = tp_tensor[0:3]
    w1_phys = tp_tensor[21:24]
    
    print("\n🔍 CROSS-TIMEFRAME CONTRAST")
    print(f"M1 Physics: {m1_phys}")
    print(f"W1 Physics (Simulated Hubris): {w1_phys}")
    
    # We expect W1 to be clamped or normalized
    if np.max(w1_phys) > 3.0:
        print(f"\n❌ FAIL: Normalization Hubris! Value {np.max(w1_phys)} exceeds safe neural range.")
    else:
        print("\n✅ PASS: Normalization within tolerance (Weights safe for gradient descent).")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
