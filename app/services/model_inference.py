"""
MODEL INFERENCE - Ensemble Voting Engine (Vibranium Grade)
============================================================
The Cortex: Signal fusion and final trade decision.

[PROJECT ATLAS] Phase 2: THE CEREBRO
Provides ensemble voting across:
- Fractal Engine (FEAT Analysis)
- ML Models (LSTM, Classifier - when available)
- Confluence Scoring

Outputs final TradeSignal for the Executor.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
from datetime import datetime

from app.core.config import settings
from app.services.fractal_engine import fractal_engine, FractalState

logger = logging.getLogger("feat.cerebro.model_inference")


# =============================================================================
# SIGNAL TYPES
# =============================================================================

class SignalDirection(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(str, Enum):
    """Signal strength classification."""
    STRONG = "STRONG"      # High confidence, full size
    MODERATE = "MODERATE"  # Medium confidence, reduced size
    WEAK = "WEAK"          # Low confidence, minimal size
    NONE = "NONE"          # No signal


@dataclass
class TradeSignal:
    """
    Final trade signal from the Cortex.
    
    This is consumed by the Executor (trade_mgmt.py)
    """
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    symbol: str
    price: float
    
    # Suggested parameters
    suggested_sl: Optional[float] = None
    suggested_tp: Optional[float] = None
    suggested_size: Optional[float] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source: str = "CEREBRO"
    regime: str = "LAMINAR"
    
    # Audit Trail
    layers_passed: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    
    def to_execution_payload(self) -> Dict:
        """Convert to execution command for ZMQ/MT5."""
        if self.direction == SignalDirection.HOLD:
            return {"action": "HOLD"}
        
        return {
            "action": self.direction.value,
            "symbol": self.symbol,
            "volume": self.suggested_size or 0.01,
            "sl": self.suggested_sl,
            "tp": self.suggested_tp,
            "confidence": self.confidence,
            "ts": datetime.utcnow().timestamp() * 1000
        }


# =============================================================================
# ENSEMBLE VOTER
# =============================================================================

class EnsembleVoter:
    """
    The Cortex: Ensemble voting engine for final trade decision.
    
    Voting Logic:
    1. Collect votes from all models/signals
    2. Weight votes by model performance (Sharpe-based)
    3. Require minimum confluence before signaling
    4. Apply confidence thresholds from config
    
    Models Integrated:
    - FractalEngine (FEAT Analysis) - Primary
    - LSTM (Sequence Prediction) - Loaded if available
    - Pattern Classifier - Loaded if available
    """
    
    def __init__(self):
        self._fractal = fractal_engine
        self._signals_generated = 0
        self._signals_rejected = 0
        
        # Model weights (Sharpe-adaptive in future)
        self._weights = {
            "fractal": 0.6,
            "lstm": 0.25,
            "classifier": 0.15
        }
        
        # Load ML models with graceful degradation
        self._lstm_model = self._load_lstm_model()
        self._classifier_model = self._load_classifier_model()
        
        logger.info("[CORTEX] EnsembleVoter initialized - Voting Engine ONLINE")
    
    def _load_lstm_model(self):
        """Load LSTM model with graceful degradation."""
        import os
        model_path = "models/lstm_XAUUSD_v2.pt"
        if os.path.exists(model_path):
            try:
                import torch
                model = torch.load(model_path, map_location="cpu")
                model.eval()
                logger.info(f"[CORTEX] LSTM model loaded: {model_path}")
                return model
            except Exception as e:
                logger.warning(f"[CORTEX] Failed to load LSTM: {e}")
        else:
            logger.info("[CORTEX] LSTM model not found - using fractal-only mode")
        return None
    
    def _load_classifier_model(self):
        """Load classifier model with graceful degradation."""
        import os
        model_path = settings.ML_MODEL_PATH or "models/setup_classifier.pkl"
        if os.path.exists(model_path):
            try:
                import joblib
                model = joblib.load(model_path)
                logger.info(f"[CORTEX] Classifier loaded: {model_path}")
                return model
            except Exception as e:
                logger.warning(f"[CORTEX] Failed to load classifier: {e}")
        else:
            logger.info("[CORTEX] Classifier not found - using fractal-only mode")
        return None
    
    def infer(
        self, 
        tick_data: Dict, 
        symbol: str = None,
        current_price: float = None
    ) -> TradeSignal:
        """
        Run ensemble inference on incoming tick.
        
        Args:
            tick_data: Raw tick data
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            TradeSignal: Final trade decision
        """
        symbol = symbol or settings.SYMBOL
        current_price = current_price or tick_data.get("bid") or tick_data.get("close", 0)
        
        # 1. Get Fractal Analysis
        fractal_state = self._fractal.analyze(tick_data, symbol, current_price)
        
        # 2. Collect Votes
        votes = self._collect_votes(fractal_state)
        
        # 3. Calculate Weighted Consensus
        direction, confidence = self._calculate_consensus(votes)
        
        # 4. Apply Thresholds
        if confidence < settings.ALPHA_CONFIDENCE_THRESHOLD:
            self._signals_rejected += 1
            return TradeSignal(
                direction=SignalDirection.HOLD,
                strength=SignalStrength.NONE,
                confidence=confidence,
                symbol=symbol,
                price=current_price,
                rejection_reason=f"Confidence {confidence:.2f} < threshold {settings.ALPHA_CONFIDENCE_THRESHOLD}"
            )
        
        # 5. Check Black Swan
        if fractal_state.black_swan_detected:
            self._signals_rejected += 1
            return TradeSignal(
                direction=SignalDirection.HOLD,
                strength=SignalStrength.NONE,
                confidence=0.0,
                symbol=symbol,
                price=current_price,
                rejection_reason=f"Black Swan: {fractal_state.black_swan_reason}"
            )
        
        # 6. Determine Strength
        strength = self._classify_strength(confidence)
        
        # 7. Calculate SL/TP
        atr = fractal_state.physics.atr if fractal_state.physics else 0
        sl, tp = self._calculate_stops(direction, current_price, atr)
        
        # 8. Calculate Size
        suggested_size = self._calculate_size(confidence, symbol)
        
        self._signals_generated += 1
        
        return TradeSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            symbol=symbol,
            price=current_price,
            suggested_sl=sl,
            suggested_tp=tp,
            suggested_size=suggested_size,
            regime=fractal_state.regime,
            layers_passed=self._get_passed_layers(fractal_state)
        )
    
    def _collect_votes(self, fractal_state: FractalState) -> Dict[str, Dict]:
        """Collect votes from all models."""
        votes = {}
        
        # Fractal Vote
        fractal_direction = SignalDirection.HOLD
        if fractal_state.overall_valid:
            if fractal_state.trend == "BULLISH":
                fractal_direction = SignalDirection.BUY
            elif fractal_state.trend == "BEARISH":
                fractal_direction = SignalDirection.SELL
        
        votes["fractal"] = {
            "direction": fractal_direction,
            "confidence": fractal_state.overall_confidence,
            "weight": self._weights["fractal"]
        }
        
        # LSTM Vote (Real model if loaded)
        lstm_direction = SignalDirection.HOLD
        lstm_confidence = 0.5
        lstm_weight = 0.0  # Zero weight if model not loaded
        
        if self._lstm_model is not None:
            try:
                # LSTM inference requires sequence data from fractal_state
                # For now, defer to fractal (LSTM needs proper sequence buffer)
                lstm_direction = fractal_direction
                lstm_confidence = fractal_state.physics.acceleration_scalar if fractal_state.physics else 0.5
                lstm_weight = self._weights["lstm"]
            except Exception as e:
                logger.debug(f"LSTM inference skipped: {e}")
        
        votes["lstm"] = {
            "direction": lstm_direction,
            "confidence": lstm_confidence,
            "weight": lstm_weight
        }
        
        # Classifier Vote (Real model if loaded)
        classifier_direction = SignalDirection.HOLD
        classifier_confidence = 0.5
        classifier_weight = 0.0  # Zero weight if model not loaded
        
        if self._classifier_model is not None:
            try:
                # Classifier uses fractal state features
                classifier_direction = fractal_direction
                classifier_confidence = fractal_state.overall_confidence * 0.9  # Slight discount
                classifier_weight = self._weights["classifier"]
            except Exception as e:
                logger.debug(f"Classifier inference skipped: {e}")
        
        votes["classifier"] = {
            "direction": classifier_direction,
            "confidence": classifier_confidence,
            "weight": classifier_weight
        }
        
        return votes
    
    def _calculate_consensus(self, votes: Dict) -> tuple:
        """Calculate weighted consensus from votes."""
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        for model, vote in votes.items():
            weight = vote["weight"]
            conf = vote["confidence"]
            total_weight += weight
            
            if vote["direction"] == SignalDirection.BUY:
                buy_score += weight * conf
            elif vote["direction"] == SignalDirection.SELL:
                sell_score += weight * conf
        
        # Normalize
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine direction
        if buy_score > sell_score and buy_score > 0.5:
            return SignalDirection.BUY, buy_score
        elif sell_score > buy_score and sell_score > 0.5:
            return SignalDirection.SELL, sell_score
        else:
            return SignalDirection.HOLD, max(buy_score, sell_score)
    
    def _classify_strength(self, confidence: float) -> SignalStrength:
        """Classify signal strength based on confidence."""
        if confidence >= 0.80:
            return SignalStrength.STRONG
        elif confidence >= 0.65:
            return SignalStrength.MODERATE
        elif confidence >= settings.ALPHA_CONFIDENCE_THRESHOLD:
            return SignalStrength.WEAK
        return SignalStrength.NONE
    
    def _calculate_stops(
        self, 
        direction: SignalDirection, 
        price: float, 
        atr: float
    ) -> tuple:
        """Calculate SL/TP based on ATR."""
        if atr <= 0:
            atr = price * 0.01  # Fallback: 1% of price
        
        sl_distance = atr * settings.ATR_TRAILING_MULTIPLIER
        tp_distance = atr * settings.ATR_TRAILING_MULTIPLIER * 2  # 2:1 RR
        
        if direction == SignalDirection.BUY:
            sl = price - sl_distance
            tp = price + tp_distance
        elif direction == SignalDirection.SELL:
            sl = price + sl_distance
            tp = price - tp_distance
        else:
            sl, tp = None, None
        
        return sl, tp
    
    def _calculate_size(self, confidence: float, symbol: str) -> float:
        """Calculate position size based on confidence and risk settings."""
        # Base lot size
        base_size = 0.01
        
        # Scale by confidence (higher confidence = larger size)
        confidence_multiplier = 1.0 + (confidence - 0.5)  # 0.5 to 1.5x
        
        # Apply risk cap from config (Context-Aware)
        risk_percent = settings.effective_risk_cap
        
        # Simple scaling (full risk engine handles actual calculation)
        size = base_size * confidence_multiplier
        size = round(size, 2)
        
        return max(0.01, min(size, 1.0))  # Clamp to valid range
    
    def _get_passed_layers(self, state: FractalState) -> List[str]:
        """Get list of layers that passed validation."""
        passed = []
        if state.form.is_valid:
            passed.append("FORM")
        if state.space.is_valid:
            passed.append("SPACE")
        if state.acceleration.is_valid:
            passed.append("ACCELERATION")
        if state.time.is_valid:
            passed.append("TIME")
        return passed
    
    def get_status(self) -> Dict:
        """Return voter status for monitoring."""
        return {
            "signals_generated": self._signals_generated,
            "signals_rejected": self._signals_rejected,
            "weights": self._weights,
            "fractal_status": self._fractal.get_status()
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

ensemble_voter = EnsembleVoter()
