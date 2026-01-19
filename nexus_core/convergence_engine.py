import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("FEAT.Convergence")

@dataclass
class ConvergenceSignal:
    score: float          # 0.0 to 1.0 (Final Confidence)
    direction: str        # BUY, SELL, WAIT
    alpha: float          # Lot size multiplier (0.5 to 1.5)
    volatility: float     # Estimated regime intensity (0.0 to 1.0)
    vetoes: List[str]     # List of reasons if blocked
    meta: Dict[str, Any]  # Debug metadata

class ConvergenceEngine:
    """
    [LEVEL 50] THE GREAT CONVERGENCE ENGINE
    =======================================
    The "High Court" of the FEAT System.
    Fuses Probability (Neural), Intent (Kinetic), and Structure (PVP)
    into a single authoritative execution decision.
    
    Philosophy: "No Trade is better than a Bad Trade."
    Default State: WAIT (0.0)
    """
    
    def __init__(self):
        from app.core.config import settings
        self.settings = settings
        
    def evaluate_convergence(self, 
                             neural_alpha: float,
                             kinetic_coherence: float,
                             p_win: float,
                             uncertainty: float) -> ConvergenceSignal:
        """
        [LEVEL 62] Unified Probabilistic Fusion (Surgical Path).
        Streamlined wrapper for the multi-head cognition loop.
        """
        # [PHASE 1] Epistemic Veto
        vetoes = []
        if uncertainty > self.settings.CONVERGENCE_MAX_UNCERTAINTY:
            vetoes.append(f"UNSTABLE_EPISTEMIC ({uncertainty:.3f})")
            
        # [PHASE 2] Fusion Logic (Bayesian Weighted)
        # Simplified for high-speed SNIPER path
        # Normalize neural_alpha to [0.5, 1.5] for current lot sizing logic
        effective_alpha = max(0.5, min(1.5, neural_alpha))
        
        # Bayesian likelihood fusion
        # base_prob is boosted by kinetic coherence if in consensus
        base_prob = p_win * (1.0 + (kinetic_coherence * 0.2))
        
        # Convergence Score: Weighted balance of Probability and Expectancy (Alpha)
        # Score is centered around 0.5
        final_score = float(np.clip(base_prob * (effective_alpha / 1.0), 0.0, 1.0))
        
        # [PHASE 3] Directionality
        direction = "BUY" if p_win > 0.5 else "SELL"
        if final_score < self.settings.CONVERGENCE_MIN_SCORE:
            # Check if reversal conviction is high (Score very low near 0, but usually p_win < 0.5)
            # For the sniper path, we only trade high-conviction
            direction = "WAIT"
            
        return ConvergenceSignal(
            score=final_score,
            direction=direction,
            alpha=neural_alpha,
            volatility=uncertainty, # Proxy for now
            vetoes=vetoes,
            meta={"path": "LEVEL_62_FAST"}
        )
        
    def evaluate(self, 
                 neural_result: Dict[str, Any],
                 kinetic_patterns: Dict[str, Any],
                 market_state: Dict[str, Any]) -> ConvergenceSignal:
        """
        Evaluates total system state to render a verdict.
        
        Args:
            neural_result: {p_win, uncertainty, prediction}
            kinetic_patterns: {pattern_id, kinetic_coherence, control_layer, layer_alignment}
            market_state: {feat_structure_score, mtf_composite_score, vol_intensity, ...}
            
        Returns:
            ConvergenceSignal
        """
        vetoes = []
        score = 0.0
        direction = "WAIT"
        
        # 1. Unpack Inputs
        p_win = neural_result.get("p_win", 0.5)
        uncertainty = neural_result.get("uncertainty", 1.0)
        
        k_id = kinetic_patterns.get("pattern_id", 0)
        k_coh = kinetic_patterns.get("kinetic_coherence", 0.0)
        k_align = kinetic_patterns.get("layer_alignment", 0.0) # 1.0 (Bull), -1.0 (Bear)
        
        struct_score = market_state.get("feat_structure_score", 50.0) # 0-100
        news_risk = market_state.get("news_event", 0.0)
        
        # [LEVEL 35] Ghost Tools Metadata
        hurst = market_state.get("hurst_exponent", 0.5)
        ofi = market_state.get("ofi_signal", 0.0)

        # 2. VETO LAYER (Hard Stops)
        # ---------------------------
        
        # A1. Hurst Chaos Filter (Veto Random Walk) [LEVEL 35]
        if 0.45 <= hurst <= 0.55:
            vetoes.append(f"HURST_CHAOS ({hurst:.2f})")
            
        # A2. Epistemic Uncertainty (The "I don't know" Guard)
        if uncertainty > self.settings.CONVERGENCE_MAX_UNCERTAINTY:
            vetoes.append(f"HIGH_UNCERTAINTY ({uncertainty:.3f})")
            
        # B. Kinetic Chaos (The "Market is confused" Guard)
        # If Pattern is NOISE (0) or FALSE REVERSAL (3) -> Strict Veto
        if k_id == 0: 
            vetoes.append("KINETIC_NOISE")
        if k_id == 3: # False Reversal
            vetoes.append("FALSE_REVERSAL_TRAP")
            
        # C. News Risk (The "Don't gamble" Guard)
        if news_risk > 0.5:
            vetoes.append("NEWS_EVENT_ACTIVE")

        # D. Structural Conflict (The "Don't fight the map" Guard)
        # If Neural says BUY but Structure Score is super low (< 30)
        if p_win > 0.6 and struct_score < 30:
             vetoes.append("STRUCTURE_CONFLICT_BEAR")
        # If Neural says SELL but Structure Score is super high (> 70)
        if p_win < 0.4 and struct_score > 70:
             vetoes.append("STRUCTURE_CONFLICT_BULL")

        # 3. SCORING LAYER (Fusion Equation)
        # ----------------------------------
        # 3. PROBABILISTIC FUSION LAYER (Log-Likelihood)
        # --------------------------------------------
        if not vetoes:
            # P(Success) = Sigmoid( Sum( Logit(P_i) ) )
            # Neural Logit
            eps = 1e-6
            p_win = max(min(p_win, 1 - eps), eps)
            logit_neural = float(np.log(p_win / (1 - p_win)))
            
            # Kinetic Evidence (Normalized to Log-Space)
            # Alignment boosts likelihood of the neural direction
            k_evidence = 0.0
            if (p_win > 0.5 and k_align > 0) or (p_win < 0.5 and k_align < 0):
                k_evidence = k_coh * 0.5 # Confidence multiplier
            
            # Structural Evidence
            s_evidence = (struct_score - 50.0) / 100.0 # Range [-0.5, 0.5]
            if p_win < 0.5: s_evidence *= -1 # Flip for Sell conviction
            
            # Alpha/Volatility Evidence
            alpha_raw = neural_result.get("alpha_multiplier", 1.0)
            vol_raw = neural_result.get("volatility_regime", 0.5)
            
            # Conviction Factor
            conviction = (alpha_raw * (1.0 - vol_raw)) * 0.2
            
            # Fusion sum
            total_logit = logit_neural + k_evidence + s_evidence + conviction
            
            # Map back to [0, 1] probability
            score = 1.0 / (1.0 + np.exp(-total_logit))
            
            # Determine Final Direction based on Score
            if score > 0.5:
                direction = "BUY"
                final_conf = (score - 0.5) * 2
            else:
                direction = "SELL"
                final_conf = (0.5 - score) * 2
            
            # Final Execution Check (Ph.D. Strictness)
            if score < self.settings.CONVERGENCE_MIN_SCORE and score > (1 - self.settings.CONVERGENCE_MIN_SCORE):
                vetoes.append(f"LOW_PROBABILISTIC_CONVERGENCE ({score:.2f})")
                direction = "WAIT"

        return ConvergenceSignal(
            score=score,
            direction=direction,
            alpha=neural_result.get("alpha_multiplier", 1.0),
            volatility=neural_result.get("volatility_regime", 0.5),
            vetoes=vetoes,
            meta={
                "p_win": p_win,
                "k_id": k_id,
                "alpha": neural_result.get("alpha_multiplier", 1.0),
                "regime": neural_result.get("volatility_regime", 0.5)
            }
        )

# Singleton Export
convergence_engine = ConvergenceEngine()
