"""
FEAT Full Chain Orchestrator
============================
Ejecuta la cadena completa FEAT en secuencia:
T (Tiempo) -> F (Forma) -> E (Espacio) -> A (Aceleración) -> EXECUTE

Si cualquier módulo retorna STOP, la cadena se detiene (Kill Switch).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from app.skills.feat_tiempo import analyze_tiempo
from app.skills.feat_forma import analyze_forma
from app.skills.feat_espacio import analyze_espacio
from app.skills.feat_aceleracion import analyze_aceleracion

logger = logging.getLogger("FEAT.Chain")


def execute_feat_chain(
    server_time_gmt: str = None,
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    m5_candles: List[Dict] = None,
    current_price: float = None,
    symbol: str = "XAUUSD",
    news_in_minutes: int = 999
) -> Dict[str, Any]:
    """
    🔗 FEAT FULL CHAIN: Ejecuta la lógica completa de trading institucional.
    
    Flujo:
    1. CHECK T (Tiempo): ¿Es hora de operar?
    2. CHECK F (Forma): ¿Cuál es la tendencia?
    3. CHECK E (Espacio): ¿El precio está en un POI?
    4. CHECK A (Aceleración): ¿Hay confirmación de entrada?
    5. EXECUTE: Generar señal de trading
    
    Returns:
        Dict con resultado completo de la cadena FEAT
    """
    chain_result = {
        "chain": "FEAT_COMPLETE",
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules_executed": [],
        "chain_status": "RUNNING"
    }
    
    # Determine H4 direction from candles
    h4_direction = "NEUTRAL"
    if h4_candles and len(h4_candles) > 0:
        last_h4 = h4_candles[-1]
        if last_h4["close"] > last_h4["open"]:
            h4_direction = "BULLISH"
        elif last_h4["close"] < last_h4["open"]:
            h4_direction = "BEARISH"
    
    # =========================================================================
    # MODULE T: TIEMPO
    # =========================================================================
    tiempo_result = analyze_tiempo(
        server_time_gmt=server_time_gmt,
        h4_candle=h4_direction,
        news_in_minutes=news_in_minutes
    )
    chain_result["modules_executed"].append({
        "module": "T_TIEMPO",
        "status": tiempo_result["status"],
        "session": tiempo_result.get("session"),
        "instruction": tiempo_result["instruction"]
    })
    
    # Kill Switch T
    if tiempo_result["status"] != "OPEN":
        chain_result["chain_status"] = "STOPPED_AT_TIEMPO"
        chain_result["final_decision"] = "NO_TRADE"
        chain_result["reason"] = tiempo_result["instruction"]
        logger.info(f"[FEAT-CHAIN] Stopped at T: {tiempo_result['instruction']}")
        return chain_result
    
    proposed_direction = "BUY" if h4_direction == "BULLISH" else "SELL" if h4_direction == "BEARISH" else None
    
    # =========================================================================
    # MODULE F: FORMA
    # =========================================================================
    forma_result = analyze_forma(
        h4_candles=h4_candles,
        h1_candles=h1_candles,
        m15_candles=m15_candles,
        current_price=current_price
    )
    chain_result["modules_executed"].append({
        "module": "F_FORMA",
        "bias": forma_result.get("bias_conclusion"),
        "h4_trend": forma_result["analysis"].get("H4", {}).get("trend"),
        "h1_trend": forma_result["analysis"].get("H1", {}).get("trend"),
        "instruction": forma_result["instruction"]
    })
    
    # Kill Switch F
    if forma_result["instruction"] not in ["PROCEED_TO_MODULE_E"]:
        chain_result["chain_status"] = "WAITING_AT_FORMA"
        chain_result["final_decision"] = "WAIT"
        chain_result["reason"] = forma_result["instruction"]
        logger.info(f"[FEAT-CHAIN] Waiting at F: {forma_result['instruction']}")
        return chain_result
    
    market_structure = forma_result.get("bias_conclusion", "NEUTRAL")
    
    # =========================================================================
    # MODULE E: ESPACIO
    # =========================================================================
    analysis_candles = h1_candles if h1_candles else m15_candles or []
    espacio_result = analyze_espacio(
        candles=analysis_candles,
        current_price=current_price,
        market_structure=market_structure,
        timeframe="H1"
    )
    chain_result["modules_executed"].append({
        "module": "E_ESPACIO",
        "fresh_zones": espacio_result["analysis"].get("total_fresh_zones", 0),
        "target_zone": espacio_result.get("target_zone"),
        "instruction": espacio_result["instruction"]
    })
    
    # Kill Switch E
    if "PROCEED_TO_MODULE_A" not in espacio_result["instruction"]:
        chain_result["chain_status"] = "WAITING_AT_ESPACIO"
        chain_result["final_decision"] = "WAIT"
        chain_result["reason"] = espacio_result["instruction"]
        logger.info(f"[FEAT-CHAIN] Waiting at E: {espacio_result['instruction']}")
        return chain_result
    
    # =========================================================================
    # MODULE A: ACELERACIÓN
    # =========================================================================
    recent_candles = m5_candles[-10:] if m5_candles else m15_candles[-10:] if m15_candles else []
    aceleracion_result = analyze_aceleracion(
        recent_candles=recent_candles,
        poi_status="PRICE_INSIDE_POI",
        proposed_direction=proposed_direction
    )
    chain_result["modules_executed"].append({
        "module": "A_ACELERACION",
        "momentum_score": aceleracion_result["analysis"].get("momentum_score"),
        "momentum_type": aceleracion_result["analysis"].get("momentum_type"),
        "instruction": aceleracion_result["instruction"]
    })
    
    # Kill Switch A
    if aceleracion_result["instruction"] not in ["EXECUTE_TRADE", "PREPARE_ENTRY_ON_RETRACEMENT"]:
        chain_result["chain_status"] = "STOPPED_AT_ACELERACION"
        chain_result["final_decision"] = "NO_TRADE"
        chain_result["reason"] = aceleracion_result["instruction"]
        logger.info(f"[FEAT-CHAIN] Stopped at A: {aceleracion_result['instruction']}")
        return chain_result
    
    # =========================================================================
    # EXECUTE: GENERATE TRADE SIGNAL
    # =========================================================================
    target_zone = espacio_result.get("target_zone", {})
    structural_low = forma_result["analysis"].get("H4", {}).get("structural_points", {}).get("last_valid_low")
    structural_high = forma_result["analysis"].get("H4", {}).get("structural_points", {}).get("last_valid_high")
    
    # Calculate SL based on structure
    if proposed_direction == "BUY":
        sl_price = structural_low if structural_low else (current_price * 0.998)
        tp1_price = target_zone.get("top", current_price * 1.002)
        tp2_price = structural_high if structural_high else (current_price * 1.005)
    else:  # SELL
        sl_price = structural_high if structural_high else (current_price * 1.002)
        tp1_price = target_zone.get("bottom", current_price * 0.998)
        tp2_price = structural_low if structural_low else (current_price * 0.995)
    
    confidence = aceleracion_result.get("confidence", "MEDIUM")
    momentum_score = aceleracion_result["analysis"].get("momentum_score", 50)
    
    chain_result["chain_status"] = "COMPLETED"
    chain_result["final_decision"] = "EXECUTE_TRADE"
    chain_result["trade_params"] = {
        "symbol": symbol,
        "order_type": f"{proposed_direction}_LIMIT" if confidence == "MEDIUM" else f"{proposed_direction}_MARKET",
        "direction": proposed_direction,
        "entry_price": target_zone.get("midpoint", current_price),
        "stop_loss": round(sl_price, 2),
        "take_profit_1": round(tp1_price, 2),
        "take_profit_2": round(tp2_price, 2),
        "lot_size_risk_percentage": 1.0 if confidence == "HIGH" else 0.5,
        "confidence_score": momentum_score,
        "reasoning": f"Tendencia H4 {h4_direction} (F), POI {target_zone.get('type', 'ZONE')} (E), Momentum {aceleracion_result['analysis'].get('momentum_type')} (A) en {tiempo_result.get('session')} (T)."
    }
    
    logger.info(f"[FEAT-CHAIN] ✅ EXECUTE: {proposed_direction} @ {current_price}, Confidence={momentum_score}")
    
    return chain_result


# =============================================================================
# Async wrapper for MCP
# =============================================================================

async def feat_full_chain(
    server_time_gmt: str = None,
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    m5_candles: List[Dict] = None,
    current_price: float = None,
    symbol: str = "XAUUSD",
    news_in_minutes: int = 999
) -> Dict[str, Any]:
    """
    MCP Tool: Execute complete FEAT chain analysis.
    """
    return execute_feat_chain(
        server_time_gmt, h4_candles, h1_candles, m15_candles, m5_candles,
        current_price, symbol, news_in_minutes
    )
