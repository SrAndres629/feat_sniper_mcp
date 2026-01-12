"""
FEAT CHAIN INSTITUCIONAL: Orquestador Completo
================================================
Integra todos los módulos MIP en una sola cadena de decisión.

Flujo:
1. TIEMPO → Session phase, fixes, alignment D1/H4/H1
2. FORMA → BOS/CHoCH, sweeps, structure
3. ESPACIO → FVG, OB, Premium/Discount
4. ACELERACIÓN → Momentum, fakeout, volume
5. FUSION → MultiTimeLearningManager weighted decision
6. LIQUIDITY → DoM preflight check
7. EXECUTION → Twin-Engine (Scalp + Swing)

Cada paso tiene probabilidad de éxito. 
El sistema NO tiene "kill switches" - solo multiplica probabilidades.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import asyncio

logger = logging.getLogger("FEAT.ChainInstitucional")


async def execute_feat_chain_institucional(
    symbol: str = "XAUUSD",
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    m5_candles: List[Dict] = None,
    current_price: float = None,
    has_sweep: bool = False,
    news_upcoming: bool = False
) -> Dict[str, Any]:
    """
    🏛️ FEAT CHAIN INSTITUCIONAL: Orquesta T→F→E→A→Fusion→Liquidity→Execution.
    
    NO hay kill switches - cada módulo aporta probabilidad.
    El resultado final es un score de 0.0 a 1.0.
    """
    from app.skills.feat_tiempo import analyze_tiempo_institucional
    from app.skills.feat_forma import analyze_forma
    from app.skills.feat_espacio import analyze_espacio
    from app.skills.feat_aceleracion import analyze_aceleracion
    from app.ml.multi_time_learning import mtf_manager
    
    result = {
        "module": "FEAT_CHAIN_INSTITUCIONAL",
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chain_stages": {},
        "ml_features": {},
        "final_decision": None
    }
    
    cumulative_probability = 1.0
    chain_log = []
    all_ml_features = {}
    
    # =========================================================================
    # STAGE 1: TIEMPO INSTITUCIONAL
    # =========================================================================
    try:
        tiempo = analyze_tiempo_institucional(
            server_time_utc, d1_direction, h4_direction, h1_direction, has_sweep, news_upcoming
        )
        
        tiempo_prob = tiempo["risk"]["combined_risk_multiplier"]
        cumulative_probability *= tiempo_prob
        
        result["chain_stages"]["1_TIEMPO"] = {
            "status": "OK",
            "session": tiempo["session"]["phase"],
            "alignment": tiempo["alignment"]["alignment_type"],
            "probability": round(tiempo_prob, 3),
            "templates": tiempo["templates"]["best_template"]["template"]
        }
        
        all_ml_features.update(tiempo.get("ml_features", {}))
        chain_log.append(f"T: {tiempo['session']['phase']} prob={tiempo_prob:.2f}")
        
        chrono_features = tiempo
    except Exception as e:
        logger.error(f"[CHAIN] Stage 1 (TIEMPO) failed: {e}")
        result["chain_stages"]["1_TIEMPO"] = {"status": "ERROR", "error": str(e)}
        chrono_features = None
        cumulative_probability *= 0.5
    
    # =========================================================================
    # STAGE 2: FORMA (Structure)
    # =========================================================================
    try:
        forma = analyze_forma(h4_candles, h1_candles, m15_candles, current_price, chrono_features)
        
        forma_prob = forma.get("alignment_score", 0.5)
        cumulative_probability *= forma_prob
        
        result["chain_stages"]["2_FORMA"] = {
            "status": "OK",
            "bias": forma["bias_conclusion"],
            "h4_trend": forma["analysis"].get("H4", {}).get("trend"),
            "h1_trend": forma["analysis"].get("H1", {}).get("trend"),
            "probability": round(forma_prob, 3)
        }
        
        all_ml_features.update(forma.get("ml_features", {}))
        chain_log.append(f"F: {forma['bias_conclusion']} prob={forma_prob:.2f}")
        
        market_structure = forma["analysis"].get("H4", {}).get("trend", "NEUTRAL")
    except Exception as e:
        logger.error(f"[CHAIN] Stage 2 (FORMA) failed: {e}")
        result["chain_stages"]["2_FORMA"] = {"status": "ERROR", "error": str(e)}
        market_structure = "NEUTRAL"
        cumulative_probability *= 0.5
    
    # =========================================================================
    # STAGE 3: ESPACIO (Zones)
    # =========================================================================
    try:
        espacio = analyze_espacio(
            h1_candles or m15_candles, current_price, market_structure, "H1", chrono_features
        )
        
        espacio_prob = espacio.get("guidance", {}).get("combined_probability", 0.5)
        cumulative_probability *= espacio_prob
        
        result["chain_stages"]["3_ESPACIO"] = {
            "status": "OK",
            "premium_discount": espacio["analysis"]["premium_discount"]["zone"],
            "zones_hq": espacio["analysis"]["zones_detected"]["high_quality"],
            "price_in_zone": espacio["price_in_zone"],
            "probability": round(espacio_prob, 3)
        }
        
        all_ml_features.update(espacio.get("ml_features", {}))
        chain_log.append(f"E: {espacio['analysis']['premium_discount']['zone']} prob={espacio_prob:.2f}")
        
        poi_status = "IN_ZONE" if espacio["price_in_zone"] else "WAITING"
    except Exception as e:
        logger.error(f"[CHAIN] Stage 3 (ESPACIO) failed: {e}")
        result["chain_stages"]["3_ESPACIO"] = {"status": "ERROR", "error": str(e)}
        poi_status = "NEUTRAL"
        cumulative_probability *= 0.5
    
    # =========================================================================
    # STAGE 4: ACELERACIÓN (Momentum)
    # =========================================================================
    try:
        proposed_direction = d1_direction if d1_direction != "NEUTRAL" else h4_direction
        
        aceleracion = analyze_aceleracion(
            m5_candles or m15_candles, poi_status, proposed_direction, 14, chrono_features
        )
        
        aceleracion_prob = aceleracion.get("execution_probability", 0.5)
        cumulative_probability *= aceleracion_prob
        
        result["chain_stages"]["4_ACELERACION"] = {
            "status": "OK",
            "momentum_type": aceleracion["analysis"]["momentum_type"],
            "momentum_score": aceleracion["analysis"]["momentum_score"],
            "is_fakeout": aceleracion["analysis"]["fakeout_check"].get("is_fakeout", False),
            "probability": round(aceleracion_prob, 3)
        }
        
        all_ml_features.update(aceleracion.get("ml_features", {}))
        chain_log.append(f"A: {aceleracion['analysis']['momentum_type']} prob={aceleracion_prob:.2f}")
    except Exception as e:
        logger.error(f"[CHAIN] Stage 4 (ACELERACION) failed: {e}")
        result["chain_stages"]["4_ACELERACION"] = {"status": "ERROR", "error": str(e)}
        cumulative_probability *= 0.5
    
    # =========================================================================
    # STAGE 5: FUSION LAYER (MTF Integration)
    # =========================================================================
    try:
        # Build signals dict for MTF manager
        signals = {
            "D1": 0.5 + (0.3 if d1_direction == "BULLISH" else (-0.3 if d1_direction == "BEARISH" else 0)),
            "H4": 0.5 + (0.3 if h4_direction == "BULLISH" else (-0.3 if h4_direction == "BEARISH" else 0)),
            "H1": 0.5 + (0.3 if h1_direction == "BULLISH" else (-0.3 if h1_direction == "BEARISH" else 0)),
            "M15": cumulative_probability,
            "M5": cumulative_probability
        }
        
        # Assume neutral Hurst for now (can be calculated from data)
        hurst_map = {"D1": 0.55, "H4": 0.55, "H1": 0.50, "M15": 0.50, "M5": 0.50}
        
        fusion_prob = mtf_manager.resolve_conflicts(signals, hurst_map)
        
        result["chain_stages"]["5_FUSION"] = {
            "status": "OK",
            "weights": mtf_manager.get_fractal_weights(hurst_map),
            "fused_probability": round(fusion_prob, 3)
        }
        
        chain_log.append(f"FUSION: prob={fusion_prob:.2f}")
    except Exception as e:
        logger.error(f"[CHAIN] Stage 5 (FUSION) failed: {e}")
        result["chain_stages"]["5_FUSION"] = {"status": "ERROR", "error": str(e)}
        fusion_prob = cumulative_probability
    
    # =========================================================================
    # STAGE 6: LIQUIDITY PREFLIGHT
    # =========================================================================
    try:
        from app.skills.liquidity import check_liquidity_preflight
        
        liquidity_ok = await check_liquidity_preflight(symbol)
        liquidity_penalty = 1.0 if liquidity_ok else 0.3
        
        result["chain_stages"]["6_LIQUIDITY"] = {
            "status": "OK" if liquidity_ok else "LOW",
            "institutional_grade": liquidity_ok,
            "penalty": liquidity_penalty
        }
        
        chain_log.append(f"LIQ: {'✅' if liquidity_ok else '⚠️'}")
    except Exception as e:
        logger.warning(f"[CHAIN] Stage 6 (LIQUIDITY) failed: {e}")
        result["chain_stages"]["6_LIQUIDITY"] = {"status": "SKIP", "note": "DoM not available"}
        liquidity_penalty = 1.0
    
    # =========================================================================
    # FINAL DECISION
    # =========================================================================
    final_probability = fusion_prob * liquidity_penalty
    final_probability = max(0.0, min(1.0, final_probability))
    
    # Determine action
    if final_probability >= 0.70:
        action = "EXECUTE_TWIN"
        size = "FULL"
    elif final_probability >= 0.55:
        action = "EXECUTE_SCALP"
        size = "HALF"
    elif final_probability >= 0.40:
        action = "PREPARE"
        size = "QUARTER"
    else:
        action = "WAIT"
        size = "NONE"
    
    # Direction
    final_direction = proposed_direction if proposed_direction != "NEUTRAL" else "NEUTRAL"
    
    result["final_decision"] = {
        "action": action,
        "direction": final_direction,
        "probability": round(final_probability, 3),
        "size": size,
        "chain_log": chain_log
    }
    
    result["ml_features"] = all_ml_features
    
    # Trade params for execution
    if action in ["EXECUTE_TWIN", "EXECUTE_SCALP"]:
        result["trade_params"] = {
            "symbol": symbol,
            "direction": final_direction,
            "entry_price": current_price,
            "size_multiplier": 1.0 if size == "FULL" else (0.5 if size == "HALF" else 0.25),
            "use_twin_engine": action == "EXECUTE_TWIN",
            "chrono_context": {
                "session": result["chain_stages"].get("1_TIEMPO", {}).get("session"),
                "template": result["chain_stages"].get("1_TIEMPO", {}).get("templates")
            }
        }
    
    logger.info(f"[FEAT-CHAIN] {action} {final_direction} prob={final_probability:.2f} | {' → '.join(chain_log)}")
    
    return result


# =============================================================================
# ASYNC MCP WRAPPER
# =============================================================================

async def feat_full_chain_institucional(
    symbol: str = "XAUUSD",
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    m5_candles: List[Dict] = None,
    current_price: float = None,
    has_sweep: bool = False,
    news_upcoming: bool = False
) -> Dict[str, Any]:
    """MCP Tool: Full institutional chain."""
    return await execute_feat_chain_institucional(
        symbol, server_time_utc, d1_direction, h4_direction, h1_direction,
        h4_candles, h1_candles, m15_candles, m5_candles, current_price,
        has_sweep, news_upcoming
    )
