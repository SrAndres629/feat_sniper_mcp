import asyncio
import logging
import sys
import os

# Fix path
sys.path.append(os.getcwd())

# Mocking settings if needed, but RiskEngine imports them. 
# We might need to mock config if it fails import.
# Lets try direct import first.

from app.services.risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyRisk")

async def test_kelly_scenarios():
    engine = RiskEngine()
    
    scenarios = [
        # (WinProb, Unc, Description)
        (0.80, 0.01, "Sniper Shot (High Conf, Low Unc)"),
        (0.60, 0.01, "Weak Signal (Low Edge, Low Unc)"),
        (0.90, 0.07, "Dirty Signal (High Conf, High Unc)"),
        (0.99, 0.09, "Dangerous Signal (Extreme Conf, Extreme Unc)"),
        (0.55, 0.01, "Barely Profitable"),
    ]
    
    print("\n--- [LEVEL 41] Damped Kelly Math Verification ---\n")
    print(f"{'Description':<40} | {'Prob':<6} | {'Unc':<6} | {'Risk %':<8} | {'Status'}")
    print("-" * 90)
    
    for prob, unc, desc in scenarios:
        risk_pct = engine._calculate_damped_kelly(prob, unc)
        
        status = "✅ OK"
        if risk_pct == 0.0 and unc > 0.08: status = "🛡️ CLAMPED"
        if risk_pct == 0.02 and prob > 0.8 and unc < 0.02: status = "🔒 MAX CAP"
        
        print(f"{desc:<40} | {prob:<6.2f} | {unc:<6.3f} | {risk_pct*100:<7.2f}% | {status}")
        
    print("\n-------------------------------------------------\n")

if __name__ == "__main__":
    asyncio.run(test_kelly_scenarios())
