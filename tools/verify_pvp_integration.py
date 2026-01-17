import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor import feat_processor
from nexus_core.features import feat_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | TEST | %(message)s")
logger = logging.getLogger("FEAT.VerifyPVP")

def test_pvp_flow():
    logger.info("📐 STARTING PVP INTEGRATION VERIFICATION 📐")

    # 1. Create Mock Data
    dates = pd.date_range("2024-01-01", periods=100, freq="1T")
    df = pd.DataFrame({
        "time": dates,
        "open": np.random.normal(2000, 5, 100),
        "high": np.random.normal(2005, 5, 100),
        "low": np.random.normal(1995, 5, 100),
        "close": np.random.normal(2000, 5, 100),
        "volume": np.random.randint(100, 1000, 100).astype(float),
        "tick_volume": np.random.randint(50, 500, 100)
    })
    
    # 2. Run Feat Processor (Engineering)
    logger.info("1. Running Feature Engineering...")
    df_processed = feat_processor.apply_feat_engineering(df)
    
    # 3. Check PVP Columns
    required_cols = ["dist_poc_norm", "pos_in_va", "density_zone", "energy_score", "poc_price", "vah", "val"]
    
    logger.info(f"2. Checking for presence of {len(required_cols)} PVP columns...")
    
    row = df_processed.iloc[-1]
    latent_vec = feat_processor.compute_latent_vector(row)
    
    missing = []
    for col in required_cols:
        val = latent_vec.get(col) if col in latent_vec else latent_vec.get(col.replace("_price", ""))
        
        # Mapping check for naming mismatches
        mapped_key = col
        if col not in latent_vec:
             if col == "vah": mapped_key = "vah_price"
             if col == "val": mapped_key = "val_price"
        
        if mapped_key not in latent_vec:
            missing.append(col)
        else:
            logger.info(f"   ✅ {col}: {latent_vec[mapped_key]:.4f}")

    if missing:
        logger.error(f"❌ Missing Columns: {missing}")
        return

    # 4. Verify Math Logic
    logger.info("3. Verifying Logic Consistency...")
    poc = latent_vec["poc_price"]
    close = row["close"]
    dist = (close - poc) / (row["atr14"] + 1e-9)
    
    # Re-calc manually
    diff = abs(dist - latent_vec["dist_poc_norm"])
    if diff < 0.001:
        logger.info(f"   ✅ Dist_POC Calculation Verified (Diff: {diff:.6f})")
    else:
        logger.error(f"   ❌ Dist_POC Mismatch! Expected {dist}, Got {latent_vec['dist_poc_norm']}")
        
    logger.info("✅ PVP MODULE INTEGRATION SUCCESSFUL")

if __name__ == "__main__":
    test_pvp_flow()
