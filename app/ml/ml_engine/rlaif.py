import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from app.core.config import settings

logger = logging.getLogger("ML.RLAIF")

class ExperienceReplay:
    """Manages recording of trade outcomes for Reinforcement Learning from AI Feedback."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def record_trade_result(self, ticket: int, profit: float, symbol: str, context: Dict):
        """
        Saves the outcome of a trade to train the 'HybridProbabilistic' model later.
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            reward_score = profit # Raw PnL for now
            
            experience = {
                "ticket": ticket,
                "timestamp": timestamp,
                "symbol": symbol,
                "outcome": "WIN" if profit > 0 else "LOSS",
                "profit": profit,
                "reward_score": reward_score,
                "state_vector": context.get("feat_vector", {}),
                "raw_context": context 
            }
            
            buffer_path = os.path.join(self.data_dir, "experience_replay.jsonl")
            with open(buffer_path, "a") as f:
                f.write(json.dumps(experience) + "\n")
                
            logger.info(f"🧠 EXPERIENCE REPLAY: Recorded Trade #{ticket} (${profit:.2f})")
            
        except Exception as e:
            logger.error(f"RLAIF Record Error: {e}")
