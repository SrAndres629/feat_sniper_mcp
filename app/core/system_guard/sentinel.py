import logging
import time
from typing import Dict, Any, Optional
from .resources import ResourcePredictor

logger = logging.getLogger("SystemGuard.Sentinel")

class SystemSentinel:
    def __init__(self, resource_predictor: ResourcePredictor):
        self.kill_switch_active = False
        self.kill_reason = None
        self.defcon = 5
        self.resource_predictor = resource_predictor
        
    def trigger_kill_switch(self, reason: str):
        if not self.kill_switch_active:
            self.kill_switch_active, self.kill_reason, self.defcon = True, reason, 1
            logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
            
    def reset_kill_switch(self):
        self.kill_switch_active, self.kill_reason, self.defcon = False, None, 5
        logger.info("🟢 Kill Switch RESET.")
        
    def is_safe(self) -> bool:
        return not self.kill_switch_active

    def check_health(self) -> Dict[str, Any]:
        rs = self.resource_predictor.predict_oom()
        if rs["status"] == "CRITICAL" and not self.kill_switch_active:
            self.trigger_kill_switch(f"RAM CRITICAL: {rs.get('message')}")
        return {"safe": self.is_safe(), "kill_switch": self.kill_switch_active, "reason": self.kill_reason, "ram": rs, "defcon": self.defcon}
