"""
MODULO 10: Live Trading Gate (Final Certification)
Phase 13: The Profit Pulse

Objetivo del Visionario:
- Gate final que controla el paso de Shadow a Live Trading
- Requiere multiples checks antes de permitir operaciones reales
- Sistema de permisos granular para escalado gradual

Requisitos para Live Trading:
1. Shadow Pilot completado (24h minimo)
2. EV Positivo y Profit Factor > 1.3
3. No drift detectado
4. Session apropiada
5. Circuit Breaker no activado
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any

logger = logging.getLogger("feat.live_gate")

class LiveTradingGate:
    """
    Gate de certificacion final para Live Trading.
    Controla el paso de Shadow a operaciones reales.
    """
    
    def __init__(self):
        self.shadow_start_time = None
        self.last_certification = None
        self.certifications_passed = {}
        
        # Certification requirements
        self.requirements = {
            "shadow_hours_min": 24,
            "ev_min": 0.0,
            "profit_factor_min": 1.3,
            "drift_allowed": False,
            "circuit_breaker_allowed": False,
            "valid_session_required": True
        }
        
    def start_shadow_pilot(self):
        """Inicia el piloto Shadow para certificacion."""
        self.shadow_start_time = datetime.now()
        logger.info(f"🌌 Shadow Pilot Started: {self.shadow_start_time.isoformat()}")
    
    def check_shadow_duration(self) -> Dict[str, Any]:
        """Verifica si el piloto Shadow ha cumplido el minimo requerido."""
        if not self.shadow_start_time:
            return {"passed": False, "reason": "Shadow pilot not started"}
        
        elapsed = datetime.now() - self.shadow_start_time
        required = timedelta(hours=self.requirements["shadow_hours_min"])
        
        passed = elapsed >= required
        return {
            "passed": passed,
            "elapsed_hours": round(elapsed.total_seconds() / 3600, 1),
            "required_hours": self.requirements["shadow_hours_min"],
            "reason": "Duration met" if passed else f"Need {(required - elapsed).total_seconds()/3600:.1f} more hours"
        }
    
    def run_certification(self, 
                          ev: float, 
                          profit_factor: float, 
                          drift_detected: bool,
                          circuit_breaker_active: bool,
                          session_valid: bool) -> Dict[str, Any]:
        """
        Ejecuta la certificacion completa para Live Trading.
        
        Returns:
            Certification result with pass/fail status and details.
        """
        checks = {}
        
        # Check 1: Shadow Duration
        shadow_check = self.check_shadow_duration()
        checks["shadow_duration"] = shadow_check
        
        # Check 2: EV
        ev_passed = ev >= self.requirements["ev_min"]
        checks["ev"] = {
            "passed": ev_passed,
            "value": round(ev, 4),
            "required": f">= {self.requirements['ev_min']}"
        }
        
        # Check 3: Profit Factor
        pf_passed = profit_factor >= self.requirements["profit_factor_min"]
        checks["profit_factor"] = {
            "passed": pf_passed,
            "value": round(profit_factor, 2),
            "required": f">= {self.requirements['profit_factor_min']}"
        }
        
        # Check 4: No Drift
        drift_passed = not drift_detected
        checks["no_drift"] = {
            "passed": drift_passed,
            "drift_detected": drift_detected
        }
        
        # Check 5: Circuit Breaker Not Active
        cb_passed = not circuit_breaker_active
        checks["circuit_breaker"] = {
            "passed": cb_passed,
            "active": circuit_breaker_active
        }
        
        # Check 6: Valid Session
        session_passed = session_valid or not self.requirements["valid_session_required"]
        checks["session"] = {
            "passed": session_passed,
            "valid": session_valid
        }
        
        # Overall certification
        all_passed = all(c.get("passed", False) for c in checks.values())
        failed_checks = [name for name, result in checks.items() if not result.get("passed", False)]
        
        certification = {
            "timestamp": datetime.now().isoformat(),
            "certified": all_passed,
            "checks": checks,
            "failed_checks": failed_checks,
            "recommendation": "PROCEED_TO_LIVE" if all_passed else f"BLOCKED: {', '.join(failed_checks)}"
        }
        
        self.last_certification = certification
        self.certifications_passed = {k: v.get("passed", False) for k, v in checks.items()}
        
        if all_passed:
            logger.info("✅ LIVE TRADING CERTIFIED - All checks passed!")
        else:
            logger.warning(f"❌ LIVE TRADING BLOCKED - Failed: {failed_checks}")
        
        return certification
    
    def is_live_allowed(self) -> bool:
        """Quick check if live trading is currently allowed."""
        if not self.last_certification:
            return False
        return self.last_certification.get("certified", False)
    
    def get_trading_mode(self) -> str:
        """Returns the recommended trading mode based on certification."""
        if self.is_live_allowed():
            return "LIVE"
        elif self.shadow_start_time:
            return "SHADOW"
        else:
            return "OFFLINE"

# Singleton global
live_gate = LiveTradingGate()

def test_live_gate():
    """Test the live trading gate."""
    print("=" * 60)
    print("🚀 FEAT SYSTEM - MODULE 10: LIVE TRADING GATE TEST")
    print("=" * 60)
    
    gate = LiveTradingGate()
    gate.start_shadow_pilot()
    
    # Simulate certification with good metrics
    result = gate.run_certification(
        ev=0.15,
        profit_factor=1.5,
        drift_detected=False,
        circuit_breaker_active=False,
        session_valid=True
    )
    
    print(f"\n📋 Certification Results:")
    print(f"   Certified: {'✅ YES' if result['certified'] else '❌ NO'}")
    print(f"   Recommendation: {result['recommendation']}")
    
    print(f"\n📊 Check Details:")
    for check_name, check_result in result['checks'].items():
        status = '✅' if check_result.get('passed') else '❌'
        print(f"   {status} {check_name}")
    
    print(f"\n🎯 Current Mode: {gate.get_trading_mode()}")

if __name__ == "__main__":
    test_live_gate()
