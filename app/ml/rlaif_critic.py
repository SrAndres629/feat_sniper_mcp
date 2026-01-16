"""
RLAIF CRITIC - The Constitutional Judge
=======================================
Analyzes trade outcomes vs Strategy Constitution (Rules).
Assigns Rewards for Reinforcement Learning.
"""

import logging
import json
import os
from typing import Dict, Any

logger = logging.getLogger("FEAT.Critic")

class RLAIFCritic:
    """
    Evaluates trades based on Process Quality, not just PnL.
    Implements the 'Constitutional AI' layer.
    """
    

    def critique_trade(self, trade_result: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        [LEVEL 30] Enhanced Constitutional Critique.
        Evaluates Outcome aligned with Process (Physics, Regime, Score).
        """
        pnl = trade_result.get("profit", 0.0)
        feat_score = context.get("feat_score", 50.0)
        acceleration = context.get("acceleration", 0.0)
        regime = context.get("regime", "UNKNOWN")
        
        reward = 0.0
        reason = "Neutral"
        
        # 1. Outcome vs Process Matrix (The Compass)
        if pnl > 0:
            if feat_score > 60:
                reward = 1.0
                reason = "✅ EXCELLENT: Good Process + Good Outcome"
            else:
                reward = -0.2 # Reduced punishment if lucky, but still not encouraging
                reason = "⚠️ LUCK: Bad Process + Good Outcome"
        else:
             if feat_score > 60:
                 reward = 0.2
                 reason = "🛡️ STOIC: Good Process + Bad Outcome (Market Noise)"
             else:
                 reward = -1.0
                 reason = "❌ FAILURE: Bad Process + Bad Outcome"

        # 2. [LEVEL 30] Meta-Consistency Bonus/Penalty
        # Did we trade against Physics?
        if acceleration < 0.2 and abs(pnl) > 0:
             reward -= 0.3
             reason += " [PHYSICS VIOLATION]"
             
        # Did we trade against Regime? (Assume context has 'trend_aligned')
        if not context.get("trend_aligned", True):
             reward -= 0.5
             reason += " [TREND VIOLATION]"

        # Clamp Reward
        reward = max(-1.0, min(1.0, reward))

        # Log structure (Experience Memory)
        experience = {
            "type": "RLAIF_JUDGMENT",
            "pnl": pnl,
            "feat_score": feat_score,
            "acceleration": acceleration,
            "regime": regime,
            "reward": round(reward, 4),
            "critique": reason,
            "timestamp": context.get("timestamp", "N/A"),
            # For Future Training
            "state_snapshot": context.get("snapshot", {}), 
            "action": trade_result.get("type", "UNKNOWN")
        }
        
        logger.info(f"⚖️ {reason} | Reward: {reward}")
        
        # Persist Judgment to Experience Memory
        self._save_judgment(experience)
            
        return reward

    def _save_judgment(self, experience: Dict[str, Any]):
        try:
            # [LEVEL 30] Experience Memory for AutoML
            filename = "data/experience_memory.jsonl"
            os.makedirs("data", exist_ok=True)
            with open(filename, "a") as f:
                f.write(json.dumps(experience) + "\n")
        except Exception as e:
            logger.error(f"Failed to save judgment: {e}")

# Singleton
rlaif_critic = RLAIFCritic()
