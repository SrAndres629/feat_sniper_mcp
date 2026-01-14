"""
MODULO 6: Session-Aware Trading Filter
Phase 13: The Profit Pulse

Objetico del Visionario:
- Filtrar trades segun la sesion de mercado (Asia, London, NY)
- Cada sesion tiene patrones de volatilidad distintos
- Escalar riesgo segun la sesion activa

Sesiones (UTC):
- ASIA: 00:00 - 08:00 (Baja volatilidad, rango estrecho)
- LONDON: 08:00 - 16:00 (Alta volatilidad, momentum)
- NEW YORK: 13:00 - 21:00 (Overlap con London = explosivo)
- OFF-HOURS: 21:00 - 00:00 (Evitar trading)
"""
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("feat.session_filter")

class SessionFilter:
    """
    Filtro de sesiones de trading basado en horarios de mercado.
    Ajusta el riesgo y bloquea trades en sesiones no optimas.
    """
    
    # Session definitions (hour ranges in UTC)
    SESSIONS = {
        "ASIA": {"start": 0, "end": 8, "risk_scale": 0.5, "volatility": "LOW"},
        "LONDON": {"start": 8, "end": 16, "risk_scale": 1.0, "volatility": "HIGH"},
        "NEWYORK": {"start": 13, "end": 21, "risk_scale": 1.0, "volatility": "HIGH"},
        "OVERLAP": {"start": 13, "end": 16, "risk_scale": 1.2, "volatility": "EXPLOSIVE"},
        "OFF_HOURS": {"start": 21, "end": 24, "risk_scale": 0.0, "volatility": "DEAD"}
    }
    
    def __init__(self, timezone_offset: int = 0):
        """
        Initialize with local timezone offset from UTC.
        Example: timezone_offset=-4 for EDT
        """
        self.timezone_offset = timezone_offset
        
    def get_current_session(self) -> Dict[str, Any]:
        """
        Determina la sesion activa basada en la hora actual.
        """
        now_utc = datetime.utcnow()
        hour = now_utc.hour
        
        # Check for overlap first (most valuable)
        if 13 <= hour < 16:
            session = "OVERLAP"
        elif 8 <= hour < 16:
            session = "LONDON"
        elif 13 <= hour < 21:
            session = "NEWYORK"
        elif 0 <= hour < 8:
            session = "ASIA"
        else:
            session = "OFF_HOURS"
        
        session_info = self.SESSIONS[session].copy()
        session_info["name"] = session
        session_info["current_hour_utc"] = hour
        
        return session_info
    
    def should_trade(self, min_volatility: str = "LOW") -> Dict[str, Any]:
        """
        Determina si debemos tradear en la sesion actual.
        """
        session = self.get_current_session()
        
        volatility_rank = {"DEAD": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "EXPLOSIVE": 4}
        current_vol = volatility_rank.get(session["volatility"], 0)
        required_vol = volatility_rank.get(min_volatility, 1)
        
        should_trade = current_vol >= required_vol and session["risk_scale"] > 0
        
        result = {
            "session": session["name"],
            "volatility": session["volatility"],
            "risk_scale": session["risk_scale"],
            "should_trade": should_trade,
            "reason": f"Session {session['name']} with {session['volatility']} volatility"
        }
        
        if not should_trade:
            logger.warning(f"🚫 TRADE BLOCKED: {result['reason']}")
        else:
            logger.info(f"✅ TRADE ALLOWED: {result['reason']} (Risk Scale: {session['risk_scale']}x)")
        
        return result
    
    def get_risk_multiplier(self) -> float:
        """
        Retorna el multiplicador de riesgo para la sesion actual.
        """
        session = self.get_current_session()
        return session["risk_scale"]

# Singleton global
session_filter = SessionFilter()

def test_session_filter():
    """Test the session filter."""
    print("=" * 60)
    print("🕐 FEAT SYSTEM - MODULE 6: SESSION FILTER TEST")
    print("=" * 60)
    
    sf = SessionFilter()
    
    session = sf.get_current_session()
    print(f"\n📊 Current Session:")
    print(f"   Name: {session['name']}")
    print(f"   Hour (UTC): {session['current_hour_utc']}")
    print(f"   Volatility: {session['volatility']}")
    print(f"   Risk Scale: {session['risk_scale']}x")
    
    trade_decision = sf.should_trade()
    print(f"\n🎯 Trade Decision:")
    print(f"   Should Trade: {'✅ YES' if trade_decision['should_trade'] else '❌ NO'}")
    print(f"   Reason: {trade_decision['reason']}")

if __name__ == "__main__":
    test_session_filter()
