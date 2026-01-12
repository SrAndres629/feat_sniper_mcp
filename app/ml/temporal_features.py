"""
TEMPORAL FEATURES GENERATOR for ML
===================================
Converts domain knowledge (killzone timing) into numerical tensors
for the neural network.

Features:
- session_heat_score (0.0-1.0)
- expansion_probability (bayesian prior)
- liquidity_state (one-hot encoded)
- alignment_factor (-1.0 to +1.0)
- volume_confirmation (binary)
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger("FEAT.TemporalFeatures")


# =============================================================================
# BAYESIAN PRIORS (initialized from domain knowledge, updated by learning)
# =============================================================================

class BayesianPriors:
    """
    Adaptive priors based on domain knowledge.
    
    These are initialized with your expert knowledge and adjusted
    nightly based on actual performance.
    """
    
    def __init__(self):
        # Block-level priors (can be updated by learning)
        self.block_priors = {
            "09:00": {"expansion_prob": 0.65, "confidence": 0.7, "accuracy_history": []},
            "09:15": {"expansion_prob": 0.75, "confidence": 0.8, "accuracy_history": []},
            "09:30": {"expansion_prob": 0.85, "confidence": 0.9, "accuracy_history": []},  # PEAK
            "09:45": {"expansion_prob": 0.75, "confidence": 0.8, "accuracy_history": []},
            "10:00": {"expansion_prob": 0.70, "confidence": 0.75, "accuracy_history": []},
            "10:15": {"expansion_prob": 0.65, "confidence": 0.7, "accuracy_history": []},
            "10:30": {"expansion_prob": 0.60, "confidence": 0.65, "accuracy_history": []},
            "10:45": {"expansion_prob": 0.55, "confidence": 0.6, "accuracy_history": []},
            "11:00": {"expansion_prob": 0.45, "confidence": 0.5, "accuracy_history": []},  # FIX
            "11:15": {"expansion_prob": 0.55, "confidence": 0.6, "accuracy_history": []},
            "11:30": {"expansion_prob": 0.50, "confidence": 0.55, "accuracy_history": []},
            "11:45": {"expansion_prob": 0.40, "confidence": 0.5, "accuracy_history": []},
            "12:00": {"expansion_prob": 0.35, "confidence": 0.45, "accuracy_history": []},
            "12:15": {"expansion_prob": 0.30, "confidence": 0.4, "accuracy_history": []},
            "12:30": {"expansion_prob": 0.20, "confidence": 0.35, "accuracy_history": []},
            "12:45": {"expansion_prob": 0.15, "confidence": 0.3, "accuracy_history": []},
        }
        
        # Confirmation level priors
        self.confirmation_boost = {
            "H1_retest_with_rejection": 0.20,
            "H1_volume_above_threshold": 0.15,
            "H4_in_killzone": 0.15,
            "D1_aligned": 0.10,
            "Full_alignment": 0.25,
        }
        
        # Penalty factors
        self.penalties = {
            "fix_event_proximity": -0.20,
            "conflict_alignment": -0.25,
            "low_volume": -0.15,
            "outside_killzone": -0.30,
        }
    
    def get_prior(self, block: str) -> Dict[str, float]:
        """Get prior for a specific block."""
        return self.block_priors.get(block, {"expansion_prob": 0.5, "confidence": 0.5})
    
    def update_prior(self, block: str, was_successful: bool) -> None:
        """
        Update prior based on actual outcome.
        Called nightly by the audit system.
        """
        if block not in self.block_priors:
            return
        
        prior = self.block_priors[block]
        prior["accuracy_history"].append(1 if was_successful else 0)
        
        # Keep last 20 samples
        if len(prior["accuracy_history"]) > 20:
            prior["accuracy_history"] = prior["accuracy_history"][-20:]
        
        # Calculate recent accuracy
        if len(prior["accuracy_history"]) >= 5:
            recent_accuracy = np.mean(prior["accuracy_history"][-5:])
            
            # Adjust prior based on performance
            if recent_accuracy < 0.4:  # Poor performance
                prior["expansion_prob"] = max(0.1, prior["expansion_prob"] - 0.10)
                prior["confidence"] = max(0.2, prior["confidence"] - 0.10)
                logger.warning(f"[PRIORS] Block {block} degraded: prob={prior['expansion_prob']:.2f}")
            
            elif recent_accuracy > 0.8:  # Excellent performance
                prior["expansion_prob"] = min(0.95, prior["expansion_prob"] + 0.05)
                prior["confidence"] = min(0.95, prior["confidence"] + 0.05)
                logger.info(f"[PRIORS] Block {block} boosted: prob={prior['expansion_prob']:.2f}")
    
    def calculate_posterior(
        self,
        block: str,
        volume_ratio: float,
        alignment_factor: float,
        h1_confirmed: bool,
        near_fix: bool = False
    ) -> float:
        """
        Calculate posterior probability using Bayes' theorem approximation.
        
        P(success | evidence)  P(success)  P(evidence | success)
        """
        prior = self.get_prior(block)
        base_prob = prior["expansion_prob"]
        
        # Apply evidence (likelihood ratio approximation)
        posterior = base_prob
        
        # Volume evidence
        if volume_ratio >= 1.5:
            posterior += self.confirmation_boost["H1_volume_above_threshold"]
        elif volume_ratio < 0.8:
            posterior += self.penalties["low_volume"]
        
        # Alignment evidence
        if alignment_factor > 0.8:  # Strong alignment
            posterior += self.confirmation_boost["Full_alignment"]
        elif alignment_factor < -0.3:  # Conflict
            posterior += self.penalties["conflict_alignment"]
        
        # H1 confirmation
        if h1_confirmed:
            posterior += self.confirmation_boost["H1_retest_with_rejection"]
        
        # Fix event penalty
        if near_fix:
            posterior += self.penalties["fix_event_proximity"]
        
        # Clamp to [0.05, 0.95]
        return max(0.05, min(0.95, posterior))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize priors for persistence."""
        return {
            "block_priors": self.block_priors,
            "confirmation_boost": self.confirmation_boost,
            "penalties": self.penalties
        }


# =============================================================================
# FEATURE VECTOR GENERATOR
# =============================================================================

class TemporalFeatureGenerator:
    """
    Generates ML-ready feature vectors from temporal and market data.
    """
    
    def __init__(self):
        self.priors = BayesianPriors()
        
        # One-hot encoding for liquidity states
        self.liquidity_states = [
            "ACCUMULATION", "SWEEP_HUNT", "EXPANSION_PEAK",
            "CONTINUATION", "CONSOLIDATION", "FIX_EVENT",
            "POST_FIX", "DECELERATION", "CLOSE_PHASE", "OUTSIDE"
        ]
    
    def one_hot_state(self, state: str) -> List[float]:
        """One-hot encode liquidity state."""
        encoding = [0.0] * len(self.liquidity_states)
        if state in self.liquidity_states:
            encoding[self.liquidity_states.index(state)] = 1.0
        else:
            encoding[-1] = 1.0  # OUTSIDE
        return encoding
    
    def generate_features(
        self,
        bolivia_hour: int,
        bolivia_minute: int,
        weekday: int,
        session_heat: float,
        expansion_prob: float,
        liquidity_state: str,
        volume_ratio: float,
        d1_direction: str,
        h4_direction: str,
        h1_direction: str,
        h1_confirmed: bool = False,
        near_fix: bool = False
    ) -> Dict[str, Any]:
        """
        Generate complete feature vector for ML model.
        
        Returns:
            Dict with 'vector' (numpy array) and 'labels' (feature names)
        """
        # Alignment factor
        def dir_to_val(d):
            d = d.upper() if d else "NEUTRAL"
            if d == "BULLISH": return 1.0
            if d == "BEARISH": return -1.0
            return 0.0
        
        d1_val = dir_to_val(d1_direction)
        h4_val = dir_to_val(h4_direction)
        h1_val = dir_to_val(h1_direction)
        alignment_factor = d1_val * 0.35 + h4_val * 0.40 + h1_val * 0.25
        
        # Block key for priors
        block_key = f"{bolivia_hour:02d}:{(bolivia_minute // 15) * 15:02d}"
        
        # Calculate posterior
        posterior = self.priors.calculate_posterior(
            block_key, volume_ratio, alignment_factor, h1_confirmed, near_fix
        )
        
        # Build feature vector
        features = {
            # Time features (4)
            "hour_sin": np.sin(2 * np.pi * bolivia_hour / 24),
            "hour_cos": np.cos(2 * np.pi * bolivia_hour / 24),
            "minute_normalized": bolivia_minute / 59.0,
            "weekday_normalized": weekday / 6.0,
            
            # Session features (3)
            "session_heat": session_heat,
            "expansion_prob_prior": expansion_prob,
            "posterior_prob": posterior,
            
            # Volume feature (2)
            "volume_ratio": min(3.0, volume_ratio),  # Cap at 3x
            "volume_above_threshold": 1.0 if volume_ratio >= 1.3 else 0.0,
            
            # Alignment features (4)
            "alignment_factor": alignment_factor,
            "d1_direction": d1_val,
            "h4_direction": h4_val,
            "h1_direction": h1_val,
            
            # Confirmation features (2)
            "h1_confirmed": 1.0 if h1_confirmed else 0.0,
            "near_fix": 1.0 if near_fix else 0.0,
            
            # Derived features (2)
            "alignment_quality": abs(alignment_factor),
            "trade_urgency": session_heat * (1.0 if volume_ratio >= 1.3 else 0.5),
        }
        
        # Add one-hot encoded state (10)
        state_encoding = self.one_hot_state(liquidity_state)
        for i, state_name in enumerate(self.liquidity_states):
            features[f"state_{state_name.lower()}"] = state_encoding[i]
        
        # Convert to vector
        vector = np.array(list(features.values()), dtype=np.float32)
        labels = list(features.keys())
        
        return {
            "vector": vector.tolist(),
            "labels": labels,
            "feature_count": len(vector),
            "posterior_probability": posterior,
            "alignment_factor": alignment_factor
        }
    
    def get_feature_importance_hints(self) -> Dict[str, str]:
        """
        Provide hints to the neural network about feature importance.
        Used for attention mechanisms.
        """
        return {
            "session_heat": "HIGH - Peak blocks (09:30-09:44) have highest expansion probability",
            "alignment_factor": "HIGH - Full D1/H4/H1 alignment indicates strong trade potential",
            "volume_above_threshold": "HIGH - Volume confirmation is essential for breakout validity",
            "posterior_prob": "HIGHEST - Combines all evidence into final probability",
            "h1_confirmed": "MEDIUM - Retest with rejection adds confidence",
            "near_fix": "NEGATIVE - LBMA fix events add noise/slippage risk"
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

temporal_feature_generator = TemporalFeatureGenerator()
bayesian_priors = temporal_feature_generator.priors


# =============================================================================
# ASYNC MCP WRAPPERS
# =============================================================================

async def feat_get_ml_feature_vector(
    bolivia_hour: int,
    bolivia_minute: int,
    weekday: int = 1,
    session_heat: float = 0.5,
    expansion_prob: float = 0.5,
    liquidity_state: str = "OUTSIDE",
    volume_ratio: float = 1.0,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    h1_confirmed: bool = False,
    near_fix: bool = False
) -> Dict[str, Any]:
    """MCP Tool: Generate ML feature vector."""
    return temporal_feature_generator.generate_features(
        bolivia_hour, bolivia_minute, weekday, session_heat, expansion_prob,
        liquidity_state, volume_ratio, d1_direction, h4_direction, h1_direction,
        h1_confirmed, near_fix
    )


async def feat_update_bayesian_prior(
    block: str,
    was_successful: bool
) -> Dict[str, Any]:
    """MCP Tool: Update Bayesian prior based on trade outcome."""
    bayesian_priors.update_prior(block, was_successful)
    return {
        "status": "updated",
        "block": block,
        "new_prior": bayesian_priors.get_prior(block)
    }


async def feat_get_all_priors() -> Dict[str, Any]:
    """MCP Tool: Get all current Bayesian priors."""
    return bayesian_priors.to_dict()
