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
        # Formula: Reward = (Profit * 1.0) - (Loss * 2.5) - (DD_Penalty) + (Entropy_Avoidance)
        
        # A. Base PnL Reward (Asymmetric)
        if pnl > 0:
            reward = 1.0 # Standard Win
            reason = "✅ WIN: Profit Secured"
        else:
            reward = -2.5 # Heavy Penalty for Loss (Survival Mode)
            reason = "❌ LOSS: Capital Erosion"

        # B. Drawdown Protection (The $20 Shield)
        # Penalize if this trade caused or deepened a drawdown
        drawdown_pct = context.get("drawdown_pct", 0.0) 
        if drawdown_pct > 0.05: # If Drawdown > 5%
            dd_penalty = drawdown_pct * 10
            reward -= dd_penalty
            reason += f" [DD PENALTY -{dd_penalty:.2f}]"

        # C. Entropy Avoidance Bonus (The Discipline)
        # If we stayed out (HOLD) during High Entropy, huge reward.
        # This requires knowing if action was HOLD. 
        # For now, if we Traded in High Entropy, we punish.
        entropy_score = context.get("entropy", 0.0)
        if entropy_score > 0.6:
            reward -= 1.0
            reason += " [HIGH ENTROPY VIOLATION]"
        elif entropy_score < 0.4 and pnl > 0:
             reward += 0.5
             reason += " [LOW ENTROPY SNIPE]"
             
        # D. Process Consistency
        if feat_score > 60 and pnl > 0:
             reward += 0.2
             reason += " + [GOOD PROCESS]"

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
