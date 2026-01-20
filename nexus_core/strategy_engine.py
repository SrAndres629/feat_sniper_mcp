"""
FEAT SNIPER: STRATEGY ENGINE (THE TACTICIAN)
============================================
Orchestrates trade execution logic, shifting between Scalp, Day, and Swing modes
based on Neural Probabilities, Macro Context, and Capital Availability.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import numpy as np

class StrategyMode(Enum):
    SCALP = "SCALP"
    DAY = "DAY"
    SWING = "SWING"

@dataclass
class TradeLeg:
    direction: str # "BUY" or "SELL"
    volume: float
    stop_loss_price: float
    take_profit_price: float
    strategy_type: StrategyMode
    intent: str # "Cash Flow" or "Wealth Run"

from app.ml.strategic_cortex import policy_agent, state_encoder, StrategicAction

class StrategyEngine:
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        
        # User Directives
        self.SWING_BUY_ONLY = True
        self.ALLOW_TWIN_TRADING = True # Split positions on high probability
        
        # Adaptive Thresholds
        self.PROMOTION_THRESHOLD = 0.80 # Probability to promote Scalp -> Swing
        
    def analyze_strategic_intent(self, 
                                 market_price: float,
                                 neural_probs: dict, 
                                 macro_context: dict,
                                 titanium_level: bool,
                                 microstructure_state: Optional[dict] = None,
                                 physics_metrics: Optional[dict] = None,
                                 temporal_physics: Optional[dict] = None) -> List[TradeLeg]:
        """
        Decides HOW to trade. Returns a list of TradeLegs (1 or 2).
        """
        
        # --- STRATEGIC CORTEX (Shadow Mode) ---
        ai_decision = "N/A"
        if microstructure_state:
            # 1. Encode State
            # Use real physics metrics if available, otherwise fallback
            physics_input = {
                "feat_composite": physics_metrics.get("feat_force", 0.0) if physics_metrics else 0.0,
                "titanium": "TITANIUM_SUPPORT" if titanium_level else "NEUTRAL",
                "acceleration": physics_metrics.get("rvol", 0.0) if physics_metrics else 0.0,
                "hurst_gate": microstructure_state.get('hurst', 0.5) > 0.55
            }
            
            # Create State Vector
            state_vec = state_encoder.encode(
                account_state=self.risk_manager.get_fund_status(),
                microstructure=microstructure_state,
                neural_probs=neural_probs,
                physics_state=physics_input,
                fractal_coherence=microstructure_state.get('fractal_coherence', 0.5),
                temporal_physics_dict=temporal_physics
            )
            
            # 2. Consult Policy Network
            action, prob, val = policy_agent.select_action(state_vec)
            ai_decision = f"{action.name} ({prob:.1%})"
            
            # 3. Log Shadow Decision
            print(f"🧠 CORTEX THOUGHTS: {ai_decision} | Value: {val:.2f} | Entropy: {microstructure_state.get('entropy_score',0):.2f}")
            
            # In FUTURE, we will override 'legs' generation here based on 'action'
        
        
        # --- LEGACY LOGIC (Determininstic) ---
        
        # 1. Determine Bias
        # Example logic: If Titanium Floor (PVP+EMA), direction is determined by physics/trend
        # For this Engine, we assume the Signal Provider (ConvergenceEngine) gave us a direction.
        # Let's assume 'direction' comes from context for now.
        direction = macro_context.get('direction', 'BUY') 
        
        # 2. Enforce "Swing = Buy Only" Rule
        # If we are selling, we CANNOT do Swing. Max leverage is DayTrade.
        allowed_strategies = [StrategyMode.SCALP, StrategyMode.DAY]
        if direction == "BUY":
            allowed_strategies.append(StrategyMode.SWING)
            
        # 3. Probability Analysis
        p_scalp = neural_probs.get('scalp', 0.0)
        p_day = neural_probs.get('day', 0.0)
        p_swing = neural_probs.get('swing', 0.0)
        
        # 4. Twin Trading Decision (Titanium + High Conf)
        # If we have a Titanium Floor AND High Scalp Prob, we try a Twin Trade.
        legs = []
        
        if titanium_level and p_scalp > 0.85:
            # TWIN TRADING: 1 Scalp (Cash) + 1 Runner (Wealth)
            
            # Volume Calculation
            # Dynamic: Twin Trading Split (50/50)
            sl_pips_est = 20
            
            # Use the new specialized Twin Lot Calculator from MoneyManager
            lot_1, lot_2 = self.risk_manager.calculate_twin_lots(
                sl_pips_est, titanium_level * 1.0
            )
            
            # Only execute Twin Trade if we got 2 valid lots
            if lot_2 > 0:
                # Leg A: The Scalper (Guaranteed Cash)
                legs.append(TradeLeg(
                    direction=direction,
                    volume=lot_1, 
                    stop_loss_price=0.0, 
                    take_profit_price=0.0, 
                    strategy_type=StrategyMode.SCALP,
                    intent="CASH_FLOW (Twin Engine A)"
                ))
                
                # Leg B: The Runner (Wealth)
                # FRACTAL & BIAS CHECK:
                # 1. Swing Bias: MUST BE BUY
                # 2. Fractal: M5 must confirm H4
                
                alignment = macro_context.get('alignment_map', {})
                m5_bias = alignment.get('M5', 'NEUTRAL')
                h4_bias = alignment.get('H4', 'NEUTRAL')
                
                fractal_confirmed = (m5_bias == direction) and (h4_bias == direction)
                
                if direction == "BUY" and fractal_confirmed:
                    runner_mode = StrategyMode.SWING
                else:
                    # If Sell or no fractal alignment, we downgrade to Day Runner
                    runner_mode = StrategyMode.DAY 
                    
                legs.append(TradeLeg(
                    direction=direction,
                    volume=lot_2,
                    stop_loss_price=0.0,
                    take_profit_price=0.0, 
                    strategy_type=runner_mode,
                    intent=f"WEALTH_RUN (Twin Engine B) | Confirmed: {fractal_confirmed}"
                ))
            else:
                # Fallback to Single Shot if not enough equity for Twin
                legs.append(TradeLeg(
                    direction=direction,
                    volume=lot_1,
                    stop_loss_price=0.0,
                    take_profit_price=0.0,
                    strategy_type=StrategyMode.SCALP,
                    intent="SINGLE_SHOT (Low Equity)"
                ))
            
        else:
            # SINGLE ENTRY (Standard)
            best_strat = StrategyMode.SCALP
            # Promotion logic
            if p_swing > p_scalp and StrategyMode.SWING in allowed_strategies:
                best_strat = StrategyMode.SWING
            elif p_day > p_scalp:
                best_strat = StrategyMode.DAY
            
            # Calculate Volume (Standard Confidence)
            confidence = 0.95 if titanium_level else p_scalp 
            lot = self.risk_manager.calculate_lot_size(20, confidence)
                
            legs.append(TradeLeg(
                direction=direction,
                volume=lot,
                stop_loss_price=0.0,
                take_profit_price=0.0,
                strategy_type=best_strat,
                intent="SINGLE_SHOT"
            ))
            
        return legs

    def optimize_targets(self, active_trades: List[TradeLeg], fresh_probs: dict):
        """
        Called dynamically during trade life.
        Can promote a Scalp to a Runner if Probabilities shift.
        """
        for trade in active_trades:
            if trade.strategy_type == StrategyMode.SCALP:
                if fresh_probs['swing'] > self.PROMOTION_THRESHOLD and trade.direction == "BUY":
                    # PROMOTION!
                    trade.strategy_type = StrategyMode.SWING
                    trade.intent = "PROMOTED_RUNNER"
                    # Signal Executor to remove TP
                    return True # Signal change occurred
        return False
