import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("FEAT.Convergence")

@dataclass
class ConvergenceSignal:
    score: float          # 0.0 to 1.0 (Final Confidence)
    direction: str        # BUY, SELL, WAIT
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
        # Thresholds (load from config in future)
        self.MIN_SCORE_EXECUTE = 0.70
        self.MIN_KINETIC_COHERENCE = 0.5
        self.MAX_NEURAL_UNCERTAINTY = 0.05
        
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
        if uncertainty > self.MAX_NEURAL_UNCERTAINTY:
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
        if not vetoes:
            # Base: Neural Probability (0.0 - 1.0)
            # We map 0.5 to 0.0 confidence, 1.0 to 1.0 confidence (Linear projection for Bull)
            # For Sell: 0.5 to 0.0, 0.0 to 1.0 confidence.
            
            raw_conf = 0.0
            if p_win > 0.5:
                direction = "BUY"
                raw_conf = (p_win - 0.5) * 2 # 0.6 -> 0.2, 0.9 -> 0.8
            else:
                direction = "SELL"
                raw_conf = (0.5 - p_win) * 2 # 0.4 -> 0.2, 0.1 -> 0.8
                
            # Boosters (Confirmation)
            kinetic_boost = 0.0
            if direction == "BUY" and k_align > 0: kinetic_boost += 0.1
            if direction == "SELL" and k_align < 0: kinetic_boost += 0.1
            if k_coh > 0.8: kinetic_boost += 0.05
            
            struct_boost = 0.0
            if direction == "BUY" and struct_score > 60: struct_boost += 0.1
            if direction == "SELL" and struct_score < 40: struct_boost += 0.1
            
            # [LEVEL 35] OFI Divergence Logic
            ofi_penalty = 0.0
            if direction == "BUY" and ofi < -0.2: ofi_penalty = 0.2
            if direction == "SELL" and ofi > 0.2: ofi_penalty = 0.2
            
            # Total Fusion
            score = raw_conf + kinetic_boost + struct_boost - ofi_penalty
            
            # Penalties
            if uncertainty > 0.03: score -= 0.1 # Mild uncertainty penalty
            
            # Cap
            score = min(max(score, 0.0), 1.0)
            
            # Final Execution Check
            if score < self.MIN_SCORE_EXECUTE:
                vetoes.append(f"LOW_CONVERGENCE_SCORE ({score:.2f})")
                direction = "WAIT"

        return ConvergenceSignal(
            score=score,
            direction=direction,
            vetoes=vetoes,
            meta={
                "p_win": p_win,
                "k_id": k_id,
                "k_align": k_align,
                "struct": struct_score
            }
        )

# Singleton Export
convergence_engine = ConvergenceEngine()
