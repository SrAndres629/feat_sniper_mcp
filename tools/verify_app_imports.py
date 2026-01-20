
import os
import sys
import logging
import traceback

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("IntegrityCheck")

# Project Root Setup
sys.path.insert(0, os.getcwd())

def verify_imports():
    """Systematically checks if all critical FEAT Sniper components are importsable."""
    logger.info("🛡️ INITIATING STRUCTURAL INTEGRITY CHECK...")
    
    components = [
        ("Core Engine", "app.core.nexus_engine", ["NexusEngine"]),
        ("Strategic Cortex", "app.ml.strategic_cortex", ["StrategicPolicyAgent", "StateEncoder", "policy_agent", "state_encoder"]),
        ("Neural Health", "nexus_core.neural_health", ["neural_health", "NeuralHealthTracker"]),
        ("Supabase Sync", "app.services.supabase_sync", ["supabase_sync"]),
        ("Feature Engine", "nexus_core.features", ["feat_features"]),
        ("Strategy Engine", "nexus_core.strategy_engine", ["StrategyEngine"]),
        ("Hybrid Model", "app.ml.models.hybrid_probabilistic", ["HybridProbabilistic"]),
        ("Training Logic", "nexus_training.train_hybrid", ["train_hybrid_model", "pre_flight_guard"]),
        ("Warfare Sim", "nexus_training.simulate_warfare", ["BattlefieldSimulator"])
    ]
    
    failed = 0
    for name, module_path, items in components:
        try:
            # Import module
            mod = __import__(module_path, fromlist=items)
            # Verify items
            for item in items:
                if not hasattr(mod, item):
                    logger.error(f"❌ {name}: Item '{item}' missing in {module_path}")
                    failed += 1
                else:
                    logger.info(f"✅ {name}: {item} verified.")
        except ImportError as e:
            logger.error(f"❌ {name}: Failed to import {module_path}. Error: {e}")
            failed += 1
        except Exception as e:
            logger.error(f"❌ {name}: Unexpected error in {module_path}: {e}")
            traceback.print_exc()
            failed += 1

    if failed == 0:
        print("\n" + "="*40)
        print("✅ ALL SYSTEMS NOMINAL - INTEGRITY VERIFIED")
        print("="*40)
        return True
    else:
        print("\n" + "!"*40)
        print(f"❌ INTEGRITY FAILURE: {failed} components compromised.")
        print("!"*40)
        return False

if __name__ == "__main__":
    if verify_imports():
        sys.exit(0)
    else:
        sys.exit(1)
