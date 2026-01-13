
# MLOPS GUARDIAN: Model Registry & Champion/Challenger
# ====================================================
# Manages the lifecycle of AI models.
# - Promotes "Champion" (Live Trading)
# - Tests "Challenger" (Shadow Mode)

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import json

logger = logging.getLogger("MLOps.Registry")

class ModelRegistry:
    """
    Central authority for Model Versioning and Promotion.
    """
    
    def __init__(self):
        self.champion_id = "v1.0.0_baseline"
        self.challenger_id = None # e.g., "v1.1.0_candidate"
        self.registry_db = {} # Mock DB for session (In prod: Supabase)
        
    def register_model(self, model_path: str, version: str, is_challenger: bool = False):
        """
        Registers a new model version.
        """
        model_hash = self._generate_hash(model_path)
        self.registry_db[version] = {
            "path": model_path,
            "hash": model_hash,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "CHALLENGER" if is_challenger else "CHAMPION"
        }
        
        if is_challenger:
            self.challenger_id = version
            logger.info(f"🏆 New Challenger Registered: {version} (Shadow Mode)")
        else:
            self.champion_id = version
            logger.info(f"👑 Champion Set: {version} (Live Mode)")

    def get_active_model_id(self) -> str:
        return self.champion_id

    def log_shadow_inference(self, input_data: Dict, prediction: float):
        """
        Logs a prediction from the Challenger model without executing it.
        """
        if not self.challenger_id: return
        
        # Simular log en DB
        log_entry = {
            "model_id": self.challenger_id,
            "input_hash": self._generate_hash(json.dumps(input_data, sort_keys=True)),
            "prediction": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
        # In real internal system, this would write to 'model_inferences' table
        logger.debug(f"👻 Shadow Inference logged for {self.challenger_id}: {prediction}")

    def promote_challenger(self):
        """
        Promotes the current Challenger to Champion.
        """
        if not self.challenger_id:
            logger.warning("No Challenger to promote.")
            return

        old_champion = self.champion_id
        self.champion_id = self.challenger_id
        self.challenger_id = None
        
        logger.info(f"🚀 PROMOTION EVENT: {self.champion_id} replaces {old_champion}")

    def _generate_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

# Singleton
model_registry = ModelRegistry()
