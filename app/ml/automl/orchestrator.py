"""
AutoML Orchestrator - FEAT NEXUS OPERATION SINGULARITY
=====================================================
The "Brain of the Brain". Manages the lifecycle of neural models.
Coordinates: Detection -> Tuning -> Deployment -> Monitoring.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from app.ml.automl.drift_detector import drift_detector
from app.ml.automl.tuner import automl_tuner
from app.core.config import settings

logger = logging.getLogger("FEAT.AutoML.Orchestrator")

class AutoMLOrchestrator:
    """
    Automated Intelligence Manager. 
    Continuously monitors performance and triggers retraining.
    """
    
    def __init__(self):
        self.is_tuning = False
        self.last_sync_time = None
        
    async def check_and_evolve(self):
        """
        Background task to monitor drift and trigger tuning.
        """
        if self.is_tuning:
            return
            
        if drift_detector.should_retrain():
            logger.info("🚀 DRIFT DETECTED. INITIALIZING SINGULARITY EVOLUTION (AutoML)...")
            self.is_tuning = True
            try:
                # Trigger retraining in a separate non-blocking loop
                # In a real production scenario, this would be a Celery task or similar
                # Here we simulate with an async local call to the training script
                await self._run_evolution_cycle()
            finally:
                self.is_tuning = False

    async def _run_evolution_cycle(self):
        """
        Orchestrates the full retraining and tuning loop via subprocess.
        Ensures the system remains non-blocking during heavy compute.
        Uses asyncio.to_thread for compatibility with Windows SelectorEventLoop.
        """
        import sys
        import subprocess
        
        script_path = settings.AUTOML_TRAIN_SCRIPT_PATH
        logger.info(f"Step 1: Spawning Evolution Subprocess: {script_path}")
        
        def run_sync():
            # Use subprocess.run for simplicity when running in a thread
            return subprocess.run(
                [sys.executable, script_path, "--auto"],
                capture_output=True,
                text=True,
                check=False
            )

        try:
            # We run the training script in a separate OS process via a thread
            # This avoids the NotImplementedError of create_subprocess_exec on Windows SelectorLoop
            result = await asyncio.to_thread(run_sync)
            
            if result.returncode == 0:
                logger.info("Step 3: Neural Hardening & Model Promotion Complete.")
                logger.info("Singularity Evolution Succeeded. New model weights initialized.")
                
                # Automatically trigger weight reload in the engine
                try:
                    from app.core.nexus_engine import engine
                    if engine and engine.ml_engine:
                        await engine.ml_engine.reload_weights()
                        logger.info("🚀 AI Brain synchronized with new Evolution weights.")
                except ImportError:
                    logger.debug("Engine not yet initialized for reload.")
                except Exception as e:
                    logger.warning(f"Failed to synchronize weights: {e}")
            else:
                logger.error(f"❌ Evolution Failed (Code {result.returncode})")
                if result.stderr:
                    logger.error(f"STDOUT: {result.stdout}")
                    logger.error(f"STDERR: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to orchestrate evolution: {str(e)}", exc_info=True)

automl_orchestrator = AutoMLOrchestrator()
