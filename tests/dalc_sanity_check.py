import sys
import os
import numpy as np
import pandas as pd
import torch
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import components to test
from nexus_core.kinetic_engine import KineticEngine 
from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from nexus_training.loss import SovereignQuantLoss
from app.ml.feat_processor.engine import FeatProcessor
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [D.A.L.C.] | %(levelname)s | %(message)s")
logger = logging.getLogger("DALC_Auditor")

class DALC_Auditor:
    def __init__(self):
        self.errors = []
        self.warnings = []
        # [V6.1.4 STEEL-VAULT] Silence redundant hydration logs during audit
        # This prevents the Phase 1-4 logs from appearing twice in the terminal.
        logging.getLogger("FeatProcessor.Engine").setLevel(logging.ERROR)
        logging.getLogger("FeatProcessor.AlphaTensor").setLevel(logging.ERROR)
        logging.getLogger("FeatProcessor.Engineering").setLevel(logging.ERROR)

    def log_fail(self, test_name, message):
        logger.error(f"🛑 FAIL: {test_name} - {message}")
        self.errors.append((test_name, message))

    def log_pass(self, test_name):
        logger.info(f"✅ PASS: {test_name}")

    # ==========================================
    # 1. FINANCIAL LOGIC: The "Time Traveller" Test
    # ==========================================
    def test_look_ahead_bias(self):
        """
        Detects if changing FUTURE data affects CURRENT features.
        """
        test_name = "Time Traveller Check (Look-Ahead Bias)"
        try:
            # Create Dummy Data (100 candles for rolling indicators)
            df = pd.DataFrame({
                'open': np.random.randn(100) + 100,
                'high': np.random.randn(100) + 105,
                'low': np.random.randn(100) + 95,
                'close': np.random.randn(100) + 102,
                'volume': np.random.randint(100, 2000, 100),
                'tick_volume': np.random.randint(100, 2000, 100),
                'time': pd.date_range(start='2024-01-01', periods=100, freq='min')
            })
            df.set_index('time', inplace=True)

            processor = FeatProcessor()
            
            # 1. Calculate Features on Original Data
            res1 = processor.process_dataframe(df.copy())
            
            # 2. Modify Future (Change Candle #90)
            df_modified = df.copy()
            df_modified.iloc[90, df_modified.columns.get_loc('close')] *= 2.0 
            
            # 3. Calculate Features on Modified Data
            res2 = processor.process_dataframe(df_modified.copy())
            
            # 4. Compare Candle #50 (Should be IDENTICAL)
            # We check multiple columns that should be strictly causal
            check_cols = list(settings.NEURAL_FEATURE_NAMES)
            
            for col in check_cols:
                if col in res1.columns and col in res2.columns:
                    val1 = res1.iloc[50][col]
                    val2 = res2.iloc[50][col]
                    if not np.isclose(val1, val2, atol=1e-8):
                        self.log_fail(test_name, f"Feature '{col}' at t=50 changed when t=90 was modified! Leakage detected.")
                        return

            self.log_pass(test_name) 
            
        except Exception as e:
            import traceback
            self.log_fail(test_name, f"Crash during test: {str(e)}\n{traceback.format_exc()}")

    # ==========================================
    # 2. MATHEMATICAL LOGIC: The "Infinity" Test
    # ==========================================
    def test_mathematical_stability(self):
        """
        Injects Zeros and NaNs to check for crashes or propagation in Kinetic Engine.
        """
        test_name = "Infinity/NaN Stress Test"
        try:
            engine = KineticEngine()
            
            # Test case: Sudden volatility crush
            df = pd.DataFrame({
                'open': [100.0, 100.0],
                'high': [100.0, 100.0],
                'low': [100.0, 100.0],
                'close': [100.0, 100.0],
                'volume': [0.0, 0.0]
            })
            
            # Test vectorized output
            res = engine.compute_vectorized_physics(df)
            
            for col in res.columns:
                nan_mask = np.isnan(res[col]) | np.isinf(res[col])
                if nan_mask.any():
                    first_nan_idx = nan_mask.idxmax() if isinstance(nan_mask, pd.Series) else np.argmax(nan_mask)
                    val = res[col].loc[first_nan_idx] if isinstance(res[col], pd.Series) else res[col][first_nan_idx]
                    self.log_fail(test_name, f"Engine produced NaN/Inf in column '{col}' at index {first_nan_idx} (Value: {val}) with Zero input.")
                    return

            # Test Clipping/Scaling Integrity
            # Large Force should be compressed by log1p
            df_extreme = pd.DataFrame({
                'open': [100.0, 200.0],
                'high': [100.0, 300.0],
                'low': [100.0, 50.0],
                'close': [100.0, 250.0],
                'volume': [1.0, 1000000.0]
            })
            res_extreme = engine.compute_vectorized_physics(df_extreme)
            
            if res_extreme['feat_force'].iloc[-1] > 25.0: # Even with massive force, log1p should keep it reasonable
                self.warnings.append((test_name, f"Force value {res_extreme['feat_force'].iloc[-1]} seems unbounded despite log1p scaling."))

            self.log_pass(test_name)
            
        except Exception as e:
            self.log_fail(test_name, str(e))

    # ==========================================
    # 3. COMPUTATIONAL LOGIC: The "Tensor Shape" Test
    # ==========================================
    def test_tensor_dimensions(self):
        """
        Verifies Model Forward Pass dimensions match Trainer expectations.
        """
        test_name = "TCN Tensor Dimension Alignment"
        try:
            # Explicitly align with FeatProcessor output (24 features)
            input_dim = 24 
            logger.info(f"Testing Model with input_dim: {input_dim}")
            
            model = HybridProbabilistic(input_dim=input_dim, hidden_dim=64, num_classes=3)
            model.eval()
            
            # Trainer sends: (Batch, SeqLen, Chan)
            # The model internally permutes this to (Batch, Chan, SeqLen) for TCN.
            dummy_input = torch.randn(32, 60, input_dim) 
            
            try:
                # Mock auxiliary inputs (Static Bags)
                feat_input = {
                    "form": torch.randn(32, 4), 
                    "space": torch.randn(32, 3),
                    "accel": torch.randn(32, 4), # Synchronized with FeatEncoder
                    "time": torch.randn(32, 4),
                    "kinetic": torch.randn(32, 4)
                }
                # Physics Tensor (Batch, 6) -> [Energy, Force, Entropy, Viscosity, Volatility, Intensity]
                p_tensor = torch.randn(32, 6)
                
                # Forward Pass
                logger.info(f"Input shape: {dummy_input.shape}")
                outputs = model(dummy_input, feat_input=feat_input, physics_tensor=p_tensor)
                
                # Check for dictionary output
                if not isinstance(outputs, dict):
                    self.log_fail(test_name, f"Expected dict output, got {type(outputs)}")
                    return

                # Check output shape (Batch, Classes)
                logits = outputs.get("logits")
                if logits is None or logits.shape != (32, 3):
                    self.log_fail(test_name, f"Logits shape mismatch: Expected (32, 3), got {logits.shape if logits is not None else 'None'}")
                    return
                
                # Check for required heads
                required_heads = ["logits", "p_win", "uncertainty", "log_var", "alpha"]
                for head in required_heads:
                    if head not in outputs:
                        self.log_fail(test_name, f"Missing expected output head: '{head}'")
                        return

                self.log_pass(test_name)
                
            except RuntimeError as re:
                import traceback
                self.log_fail(test_name, f"Runtime Error during forward pass: {re}\n{traceback.format_exc()}")
                    
        except Exception as e:
            import traceback
            self.log_fail(test_name, f"Setup Error: {str(e)}\n{traceback.format_exc()}")

    def run_full_audit(self):
        logger.info("🛡️ STARTING D.A.L.C. SYSTEM AUDIT...")
        self.test_look_ahead_bias()
        self.test_mathematical_stability()
        self.test_tensor_dimensions()
        
        if self.errors:
            logger.error(f"❌ AUDIT FAILED. {len(self.errors)} CRITICAL ERRORS FOUND.")
            for name, msg in self.errors:
                logger.error(f"   - {name}: {msg}")
            sys.exit(1) # Abort Training
        else:
            logger.info("✅ D.A.L.C. AUDIT PASSED. SYSTEM INTEGRITY VERIFIED.")
            if self.warnings:
                logger.warning(f"⚠️ AUDIT PASSED WITH {len(self.warnings)} WARNINGS.")
                for name, msg in self.warnings:
                    logger.warning(f"   - {name}: {msg}")
            sys.exit(0) # Proceed

if __name__ == "__main__":
    auditor = DALC_Auditor()
    auditor.run_full_audit()
