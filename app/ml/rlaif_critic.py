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
        # Formula: Reward = (Profit * (R:R ^ 2)) - (Loss * Asymmetric_Penalty)
        
        # A. Base Sniper Reward (Exponential R:R)
        initial_stop = context.get("initial_stop", 0.05) # 5 pips
        is_news = context.get("is_news_event", False)
        
        if pnl > 0:
            risk_reward = pnl / max(0.001, abs(initial_stop))
            # Exponential Reward: Premiamos mucho más un 1:10 que diez 1:1
            reward = 2.0 + (risk_reward ** 2) 
            
            # [MAESTRÍA EN NOTICIAS] Bonus for surviving and profiting from volatility
            if is_news:
                reward *= 2.0
                reason = f"🚀 LEGENDARY NEWS SURVIVAL: R:R {risk_reward:.1f}x | Reward {reward:.2f}"
            else:
                reason = f"🎯 SNIPER WIN: R:R {risk_reward:.1f}x | Reward {reward:.2f}"
        else:
            # Penalizamos no solo perder, sino perder por "tonto" (fuera de zona)
            if feat_score < 70:
                reward = -15.0 # Increased penalty for rogue trades
                reason = "❌ ROGUE LOSS: Trading against the FEAT Spectrum!"
            else:
                reward = -5.0 if is_news else -2.5
                reason = f"❌ LOSS: Market violence exceeded SL {'(News Event)' if is_news else ''}"

        # B. Drawdown Protection (The 'Drawdown Whip')
        # User Directive: "Castigo exponencial si el balance cae >1.5% en una sesión."
        # Here we approximate 'session' impact via the current trade's impact on balance.
        drawdown_pct = context.get("drawdown_pct", 0.0) 
        if drawdown_pct > 0.015: # > 1.5% Drawdown is Toxic
            # Exponential Penalty: e^(DD * 100)
            dd_penalty = (drawdown_pct * 1000.0) ** 1.5
            reward -= dd_penalty
            reason += f" [IRON WHIP DD -{dd_penalty:.1f}]"

        # C. Entropy (Indecision) Penalty [New Sovereign Spec]
        # "Si la probabilidad oscila violentamente (0.4-0.6)" -> Punish Indecision
        entropy_score = context.get("entropy", 0.0)
        # Using entropy score directly as proxy for indecision width
        if 0.4 < entropy_score < 0.6:
            reward -= 2.0 
            reason += " [INDECISION FRICTION]"
        elif entropy_score > 0.7:
             reward -= 5.0 # High penalty for pure chaos
             reason += " [CHAOS VIOLATION]"
        elif entropy_score < 0.35 and pnl > 0:
             reward += 2.0 # Bonus for clear skies
             reason += " [PRISTINE SNIPE]"

        # D. Inertia Friction (Time-Under-Risk)
        # "Castigo por cada barra que un trade está abierto sin movimiento"
        duration = context.get("duration_bars", 0)
        if duration > 5 and pnl < 0.001: # Stagnant trade
            friction = (duration - 5) * 0.5
            reward -= friction
            reason += f" [STAGNATION -{friction:.1f}]"
             
        # D. Process Consistency (FEAT Alignment)
        if feat_score > 85 and pnl > 0:
             reward *= 1.5 # Multiplier for high-confidence alignment
             reason += " x [GOD MODE ALIGNMENT]"

        # Asymmetric Clamping (No lower limit for massive failures, but capped at 100 for wins)
        reward = min(100.0, reward)

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
