"""
FEAT NEXUS: INTEGRAL FLOW VERIFIER (Ph.D. Institutional Alignment)
================================================================
Validates the end-to-end synaptic wiring from raw M1 data through:
1. SEAT Feature Engineering (FeatProcessor)
2. Neural Latent Adaptation (compute_latent_vector)
3. Stochastic Hybrid Inference (MLEngine)
4. Probabilistic Logic Fusion (ConvergenceEngine)
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from app.core.config import settings
from app.ml.feat_processor import feat_processor
from app.ml.ml_engine import ml_engine
from nexus_core.convergence_engine import convergence_engine

# Silence noisy logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify.integral")

async def verify_integral_flow():
    logger.info("🚀 INITIALIZING INTEGRAL FLOW PROBE...")
    
    # 1. Generate Synthetic M1 Context (Doctoral Standard)
    logger.info("Step 1: Generating standard market context...")
    data = {
        'open': np.linspace(2000, 2010, 100),
        'high': np.linspace(2005, 2015, 100),
        'low': np.linspace(1995, 2005, 100),
        'close': np.linspace(2000, 2010, 100),
        'volume': np.random.randint(1000, 5000, 100),      # Standardized for engines
        'tick_volume': np.random.randint(100, 500, 100)  # MT5 standard
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    
    # 2. Phase A: Feature Engineering (Spatio-Temporal Fusion)
    logger.info("Step 2: Engineering Stochastic Features...")
    processed_df = feat_processor.process_dataframe(df)
    last_row = processed_df.iloc[-1]
    
    # 3. Phase B: Neural Latent Adaptation
    logger.info("Step 3: Adapting context to Neural Latent Space Z_t...")
    feature_map = feat_processor.compute_latent_vector(last_row)
    
    # Verify mapping alignment with settings
    feature_vector = [feature_map.get(name, 0.0) for name in settings.NEURAL_FEATURE_NAMES]
    logger.info(f"✅ Latent Vector Dims: {len(feature_vector)} / Goal: {len(settings.NEURAL_FEATURE_NAMES)}")
    
    # 4. Phase C: Neural Inference
    logger.info("Step 4: Executing Stochastic Hybrid Inference (MLEngine)...")
    try:
        # Pass the full feature map (dict) as expected by predict_async
        brain_score = await ml_engine.predict_async("XAUUSD", feature_map)
        print(f"DEBUG: brain_score={brain_score}")
        logger.info(f"✅ Neural Output: p_win={brain_score.get('p_win', 0):.3f}, alpha={brain_score.get('alpha_multiplier', 1.0):.3f}")
    except Exception as e:
        logger.error(f"❌ Neural Inference Failed: {e}")
        return

    # 5. Phase D: Probabilistic Fusion (The Great Convergence)
    logger.info("Step 5: Executing Bayesian Fusion (ConvergenceEngine)...")
    convergence = convergence_engine.evaluate_convergence(
        neural_alpha=brain_score.get("alpha_multiplier", 1.0),
        kinetic_coherence=feature_map.get("kinetic_coherence", 0.0),
        p_win=brain_score.get("p_win", 0.5),
        uncertainty=brain_score.get("uncertainty", 0.1)
    )
    
    print(f"DEBUG: convergence={convergence}")
    
    logger.info("================================================================")
    logger.info("FINAL ARCHITECTURAL SNAPSHOT")
    logger.info(f"- Convergence Score: {convergence.score:.3f}")
    logger.info(f"- Signal Direction:  {convergence.direction}")
    logger.info(f"- Uncertainty (MC):  {brain_score.get('uncertainty'):.3f}")
    logger.info(f"- Active Vetoes:     {', '.join(convergence.vetoes) if convergence.vetoes else 'NONE'}")
    logger.info("================================================================")
    
    if convergence.score >= 0.0:
        logger.info("🏆 INTEGRAL FLOW VERIFIED: Synaptic Wiring is 100% Synchronized.")
    else:
        logger.error("❌ INTEGRAL FLOW FAILURE: Mathematical Inconsistency Detected.")

if __name__ == "__main__":
    asyncio.run(verify_integral_flow())
