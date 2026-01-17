import numpy as np
import os
import sys

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.config import settings

def generate_synthetic_convergence_data(n_samples=1000, seq_len=60, input_dim=16):
    """
    [PHD LEVEL SYNTHETIC GENERATOR]
    Creates a mathematically structured dataset to stress-test the 
    HybridProbabilistic Neural Network.
    
    Simulates:
    1. Temporal Manifold (TCN Input): Geometric Brownian Motion with Regime Switching.
    2. Kinetic Manifold (Static Input): Correlated Pattern IDs (Expansion/Compression).
    3. Physics Manifold (Loss Label): Acceleration derived from volatility.
    """
    print(f"🔬 GENERATING SYNTHETIC CONVERGENCE DATA (N={n_samples})...")
    
    # 1. TEMPORAL MANIFOLD (X)
    # ------------------------
    # Shape: (N, Seq, Channels)
    # We simulate 3 Regimes: 0=Bear, 1=Range, 2=Bull
    regimes = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    
    X = np.zeros((n_samples, seq_len, input_dim))
    
    for i in range(n_samples):
        # Base Drift based on Regime
        drift = 0.0
        if regimes[i] == 0: drift = -0.001 # Bear
        if regimes[i] == 2: drift = 0.001  # Bull
        
        # Volatility
        vol = 0.01 if regimes[i] == 1 else 0.02
        
        # Random Walk
        walk = np.cumsum(np.random.normal(drift, vol, size=(seq_len, input_dim)), axis=0)
        X[i] = walk

    # 2. KINETIC MANIFOLD (Static Features)
    # -------------------------------------
    # We must correlate Kinetic Features with Regimes to verify Learning capability
    
    # Kinetic: [PatternID, Coherence, Alignment, BiasDist]
    kinetic = np.zeros((n_samples, 4))
    
    for i in range(n_samples):
        r = regimes[i]
        
        if r == 2: # Bull Trend
            p_id = 1.0 # Expansion
            coh = np.random.uniform(0.7, 1.0)
            align = 1.0 # Bull
            dist = np.random.uniform(0.5, 2.0)
        elif r == 0: # Bear Trend
            p_id = 1.0 # Expansion check (Expansion Bear)
            coh = np.random.uniform(0.7, 1.0)
            align = -1.0 # Bear
            dist = np.random.uniform(-2.0, -0.5)
        else: # Range
            p_id = 2.0 # Compression
            coh = np.random.uniform(0.0, 0.4)
            align = 0.0 # Neutral
            dist = np.random.uniform(-0.2, 0.2)
            
        kinetic[i] = [p_id, coh, align, dist]

    # Other Static Features (Noise/Standard for now)
    form = np.random.normal(0, 1, (n_samples, 4)) # Skew, Kurtosis, Entropy...
    space = np.random.normal(0, 1, (n_samples, 3)) # DistPOC...
    accel = np.random.normal(0, 1, (n_samples, 3)) # EnergyZ...
    time = np.zeros((n_samples, 4))
    time[:, 2] = 1.0 # Session encoded (e.g. NY)

    static_features = {
        "form": form,
        "space": space,
        "accel": accel,
        "time": time,
        "kinetic": kinetic
    }

    # 3. LABELS (y) & PHYSICS (accel)
    # -------------------------------
    # y matches Regime perfectly (Ideal case for unit test)
    y = regimes.astype(int) 
    
    # Physics (Acceleration) matches Coherence
    physics = kinetic[:, 1] # Use coherence as proxy for acceleration magnitude

    # 4. SAVE
    # -------
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "synthetic_convergence.npz")
    
    np.savez(
        save_path, 
        X=X, 
        y=y, 
        physics=physics, 
        static_features=static_features
    )
    
    print(f"✅ Data Saved to {save_path}")
    print(f"   X Shape: {X.shape}")
    print(f"   Kinetic Shape: {kinetic.shape}")
    print(f"   Classes: {np.unique(y, return_counts=True)}")

if __name__ == "__main__":
    generate_synthetic_convergence_data()
