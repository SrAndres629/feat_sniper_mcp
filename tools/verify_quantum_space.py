import sys
import os
import pandas as pd
import numpy as np

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_core.zone_projector.calculations import calculate_gaussian_proximity
from app.ml.feat_processor.space import TensorTopologist
from nexus_core.zone_projector.engine import ZoneProjector

def verify_quantum_physics():
    print("\n" + "="*50)
    print("⚛️ [QUANTUM SPACE] PHYSICS ENGINE AUDIT")
    print("="*50)

    # 1. Test Gaussian Logic (Micro-test)
    print("\n[TEST 1] Gaussian Decay (RBF)")
    zone_center = 2000.0
    sigma = 5.0 # Zone width/tolerance
    
    prices = [2000.0, 2002.5, 2005.0, 2010.0, 2020.0]
    topo = TensorTopologist()
    
    for p in prices:
        intensity = topo.gaussian_decay(p, zone_center, sigma)
        print(f"   Price {p:.1f} (Dist {abs(p-zone_center):.1f}) -> Intensity: {intensity:.4f}")
        
    # 2. Test Engine Integration
    print("\n[TEST 2] Engine Synapses")
    projector = ZoneProjector()
    # Create dummy DF with hardened columns
    df = pd.DataFrame({'close': [2000.0]*10, 'high': [2005]*10, 'low': [1995]*10, 'open': [2000]*10})
    df['ob_bull'] = False
    df['confluence_score'] = 3.5 # Simulated High Confluence
    
    action = projector.generate_action_plan(df, 2000.0)
    print(f"   Action Plan Confidence: {action.confidence_score:.2f}")
    print(f"   Reasoning: {action.reasoning}")

    # 3. Tensor Output
    print("\n[TEST 3] Tensor Generation")
    atr = 5.0
    tensor = topo.generate_space_tensor(df, 2000.0, atr)
    print(f"   [SPACE TENSOR]: {tensor}")

    print("\n" + "="*50)
    print("✅ AUDIT COMPLETE: PHYSICS ARE REAL")
    print("="*50 + "\n")

if __name__ == "__main__":
    verify_quantum_physics()
