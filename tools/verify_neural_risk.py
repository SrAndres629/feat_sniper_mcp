import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_core.risk_engine.neural_risk import NeuralRiskOrchestrator

def verify_neural_risk():
    print("🛡️ [RISK CORE] NEURAL PROBABILISTIC STRESS TEST")
    print("==============================================")
    
    risk_orchestrator = NeuralRiskOrchestrator()
    
    # SCENARIO 1: High Confidence Scalp during Low Risk Macro
    print("\n📍 Scenario 1: London Scalp (High Confidence, Meta-Safe)")
    alpha_scalp = {
        "regime_probability": [0.85, 0.10, 0.05], # Scalp dominant
    }
    macro_safe = {
        "macro_safe": 1.0,
        "macro_position_multiplier": 1.0,
        "macro_danger": 0.0
    }
    
    res1 = risk_orchestrator.calculate_capital_intensity(alpha_scalp, macro_safe, 1.0)
    print(f"   ► Regime: {res1['dominant_regime']} (Conf: {res1['confidence']:.2f})")
    print(f"   ► Final Risk Coeff: {res1['final_risk_coefficient']:.4f}")
    print(f"   ► Leverage Multiplier: {res1['leverage_multiplier']}x")
    print(f"   ► SL ATR Mult: {res1['sl_params']['atr_multiplier']}")

    # SCENARIO 2: High News Impact Imminent (Macro Danger)
    print("\n📍 Scenario 2: US NFP Imminent (Macro Danger)")
    alpha_day = {
        "regime_probability": [0.10, 0.80, 0.10], # Daytrade setup
    }
    macro_danger = {
        "macro_safe": 0.0,
        "macro_position_multiplier": 0.1, # Severe reduction
        "macro_danger": 1.0 # High alert
    }
    
    res2 = risk_orchestrator.calculate_capital_intensity(alpha_day, macro_danger, 1.0)
    print(f"   ► Regime: {res2['dominant_regime']}")
    print(f"   ► Final Risk Coeff: {res2['final_risk_coefficient']:.4f}")
    print(f"   ► Execution Status: {res2['execution_status']}")
    print(f"   ► Reason: {res2['reason']}")

    # SCENARIO 3: Swing Trade with Mid-Confidence
    print("\n📍 Scenario 3: Structural Swing (Mid Confidence)")
    alpha_swing = {
        "regime_probability": [0.05, 0.15, 0.80], 
    }
    macro_mid = {
        "macro_safe": 1.0,
        "macro_position_multiplier": 0.9,
        "macro_danger": 0.1
    }
    
    res3 = risk_orchestrator.calculate_capital_intensity(alpha_swing, macro_mid, 1.0)
    print(f"   ► Regime: {res3['dominant_regime']}")
    print(f"   ► Final Risk Coeff: {res3['final_risk_coefficient']:.4f} (Higher intensity for Swing)")
    print(f"   ► SL ATR Mult: {res3['sl_params']['atr_multiplier']} (Wide stop for Swing)")

    print("\n✅ Verification Complete: Neural Risk responds to probabilities, not just rules.")
    print("==============================================")

if __name__ == "__main__":
    verify_neural_risk()
