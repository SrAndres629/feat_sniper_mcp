"""
TEMPORAL TREND INTENT (TTI-FEAT)
================================
Implements "Quantum Temporal Hierarchy" reasoning.

Logic:
- H4 Structure is the "Carrier Wave".
- Killzone are "Injection Windows".
- Signals outside windows are "Discounted".

Goal: Identify if the "Intent" of the market aligns with Institutional Time.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger("FEAT.Intent")

class TemporalTrendIntent:
    """
    Decides if a signal has 'Institutional Intent' behind it.
    """
    
    def __init__(self):
        pass

    def calculate_intent_score(
        self,
        killzone_block: Dict[str, Any],
        h4_direction: str,
        regime_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates the TTI Score (0.0 - 1.0).
        
        Args:
            killzone_block: Output from killzone_intelligence
            h4_direction: "BULLISH", "BEARISH", "NEUTRAL"
            regime_context: Output from feat_regime (is_trending, is_trap)
        
        Returns:
            Dict: {score, is_aligned, discount_factor}
        """
        
        # 1. Temporal Injection Window
        # Is this a high-probability block? (e.g. 09:30)
        expansion_prob = killzone_block.get("expansion_prob_prior", 0.5)
        # Or checking liquidity state
        liquidity_state = killzone_block.get("liquidity_state", "OUTSIDE")
        
        # 2. Carrier Wave Alignment
        # Does the regime align with H4?
        regime = regime_context.get("regime", "NEUTRAL")
        
        score = 0.5
        discount_factor = 1.0
        
        # --- LOGIC ---
        
        # A. Window Check
        if liquidity_state == "EXPANSION_PEAK":
            score += 0.2
            discount_factor = 1.0 # Full signal quality
        elif liquidity_state == "ACCUMULATION":
            score -= 0.1
            discount_factor = 0.8 # Reduced size
        elif liquidity_state == "FIX_EVENT":
            score -= 0.3
            discount_factor = 0.0 # No trade zone
        elif liquidity_state == "OUTSIDE":
             # Outside killzone -> Heavy discount
             discount_factor = 0.5
        
        # B. Structure Alignment
        # If Regime is Expansion Real and H4 is Aligned -> High Score
        if h4_direction != "NEUTRAL":
            if h4_direction == "BULLISH" and regime == "EXPANSION_REAL":
                score += 0.2
            elif h4_direction == "BEARISH" and regime == "EXPANSION_REAL":
                score += 0.2
            elif regime == "REVERSAL": 
                 # Reversal against H4? Dangerous unless H4 is turning
                 pass
        
        # C. Trap Logic
        if regime_context.get("is_trap", False):
            # If trap detected, Intent is "Malicious" -> Only trade Reversal
            score = 0.4 
        
        # Clamp
        score = max(0.0, min(1.0, score))
        
        return {
            "tti_score": round(score, 2),
            "discount_factor": discount_factor,
            "is_aligned": score > 0.6 and discount_factor > 0.7,
            "recommendation": self._get_recommendation(score, discount_factor)
        }

    def _get_recommendation(self, score, discount) -> str:
        if discount == 0.0: return "AVOID_FIX"
        if score > 0.8: return "INSTITUTIONAL_EXECUTE"
        if score > 0.6: return "VALID_ENTRY"
        if score < 0.4: return "WAIT_BETTER_CONTEXT"
        return "SCALP_ONLY"

# Singleton
tti_engine = TemporalTrendIntent()
