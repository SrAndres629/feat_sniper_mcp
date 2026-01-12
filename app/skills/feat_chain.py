"""
FEAT CHAIN INSTITUCIONAL: Orquestador Completo - v7.0 (Physics Aware)
=====================================================================
Integra todos los mdulos MIP (Physics, Regime, Chrono) en una sola cadena de decisin.

Flujo:
1. TIEMPO  Session phase, fixes, alignment
2. PHYSICS  PVP, MCI, Liquidity Primitives
3. REGIME  FSM State (Manipulation vs Expansion)
4. FORMA  BOS/CHoCH, sweeps, structure
5. ESPACIO  FVG, OB, Premium/Discount
6. ACELERACIN  Momentum, fakeout, volume
7. FUSION  MultiTimeLearningManager weighted decision
8. INTENT  TTI Filter (Intent vs Noise)
9. LIQUIDITY  DoM preflight check
10. EXECUTION  Twin-Engine (Scalp + Swing)

Cada paso tiene probabilidad de xito. 
El sistema NO tiene "kill switches" - solo multiplica probabilidades.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import asyncio
import pandas as pd

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
     FEAT CHAIN INSTITUCIONAL: T-P-R-F-E-A-Fusion-I-L-Ex.
    
    Ahora con PHYSICS (PVP/MCI) y REGIME (FSM).
    """
    # Import modules (skill injection)
    from app.skills.feat_tiempo import analyze_tiempo_institucional
    from app.skills.market_physics import market_physics
    from app.skills.feat_regime import regime_fsm
    from app.skills.feat_forma import analyze_forma
    from app.skills.feat_espacio import analyze_espacio
    from app.skills.feat_aceleracion import analyze_aceleracion
    from app.skills.feat_intent import tti_engine
    from app.ml.multi_time_learning import mtf_manager
    from app.skills.killzone_intelligence import feat_get_current_killzone_block
    
    result = {
        "module": "FEAT_CHAIN_INSTITUCIONAL_v7",
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chain_stages": {},
        "ml_features": {},
        "final_decision": None
    }
    
    cumulative_probability = 1.0
    chain_log = []
    all_ml_features = {}
    ml_features_to_log = [
        "volatility_regime_norm",
        "acceptance_ratio",
        "wick_stress"
    ]
    
    # helper to dataframe
    def candles_to_df(candles):
        if not candles: return pd.DataFrame()
        return pd.DataFrame(candles)

    df_m15 = candles_to_df(m15_candles)
    df_m5 = candles_to_df(m5_candles)
    
    # =========================================================================
    # STAGE 1: TIEMPO INSTITUCIONAL
    # =========================================================================
    try:
        tiempo = analyze_tiempo_institucional(
            server_time_utc, d1_direction, h4_direction, h1_direction, has_sweep, news_upcoming
        )
        # Also get detailed killzone block
        kz_block = await feat_get_current_killzone_block(server_time_utc)
        
        tiempo_prob = tiempo["risk"]["combined_risk_multiplier"]
        cumulative_probability *= tiempo_prob
        
        result["chain_stages"]["1_TIEMPO"] = {
            "status": "OK",
            "session": tiempo["session"]["phase"],
            "kz_state": kz_block.get("liquidity_state"),
            "probability": round(tiempo_prob, 3)
        }
        
        all_ml_features.update(tiempo.get("ml_features", {}))
        all_ml_features.update({f"kz_{k}": v for k,v in kz_block.items() if isinstance(v, (int, float))})
        
        chain_log.append(f"T: {tiempo['session']['phase']} prob={tiempo_prob:.2f}")
        chrono_features = tiempo
        
    except Exception as e:
        logger.error(f"[CHAIN] Stage 1 (TIEMPO) failed: {e}")
        result["chain_stages"]["1_TIEMPO"] = {"status": "ERROR"}
        tiempo = {}
        kz_block = {}
        chrono_features = None
        cumulative_probability *= 0.5

    # =========================================================================
    # STAGE 2: MARKET PHYSICS (PVP FEAT + CVD + Breakout)
    # =========================================================================
    try:
        # Use M15 for profile and M5 for CVD/MCI
        physics_pvp = market_physics.calculate_pvp_feat(df_m15)
        physics_cvd = market_physics.calculate_cvd_metrics(df_m5)
        physics_mci = market_physics.calculate_mci(df_m5)
        
        # Estimate breakout probability
        atr_data = get_technical_indicator(df_m15, IndicatorRequest(symbol=symbol, indicator="ATR"))
        breakout_probs = market_physics.estimate_breakout_probability(physics_pvp, physics_cvd, atr_data.get("value", 0.1))
        
        physics_metrics = {
             "pvp": physics_pvp, 
             "cvd": physics_cvd,
             "mci": physics_mci, 
             "breakout": breakout_probs
        }
        
        # Adjust probability based on physics breakout prediction
        physics_prob = 1.0
        if breakout_probs["p_up"] > 0.7 or breakout_probs["p_down"] > 0.7:
             physics_prob = 1.2 # High confidence in institutional intent
        elif abs(physics_pvp["z_score"]) < 0.5:
             physics_prob = 0.8 # Range bound / low interest
            
        cumulative_probability *= physics_prob
        
        result["chain_stages"]["2_PHYSICS"] = {
             "poc": physics_pvp.get("poc"),
             "z_score": physics_pvp.get("z_score"),
             "cvd_accel": physics_cvd.get("cvd_acceleration"),
             "p_breakout": max(breakout_probs.values())
        }
        
        # Inject ML features
        all_ml_features["poc_z_score"] = physics_pvp["z_score"]
        all_ml_features["cvd_acceleration"] = physics_cvd["cvd_acceleration"]
        all_ml_features["cvd_imbalance"] = physics_cvd["imbalance_ratio"]
        
        chain_log.append(f"P: POC={physics_pvp['poc']:.2f} Z={physics_pvp['z_score']:.1f}")
        
    except Exception as e:
        logger.error(f"[CHAIN] Stage 2 (PHYSICS) failed: {e}")
        result["chain_stages"]["2_PHYSICS"] = {"status": "ERROR"}
        physics_metrics = {}

    # =========================================================================
    # STAGE 3: REGIME FSM (State Machine)
    # =========================================================================
    try:
        # Need rudimentary structure context for Regime
        # We'll peek at H4 direction
        structure_context = {
            "trend": h4_direction, 
            "has_sweep": has_sweep or physics_mci.get("is_sweep", False)
        }
        
        # Killzone context for FSM
        temporal_features = {
            "session_heat": kz_block.get("session_heat", 0.0),
            "liquidity_state": kz_block.get("liquidity_state", "OUTSIDE")
        }
        
        regime_data = regime_fsm.detect_regime(physics_metrics, temporal_features, structure_context)
        regime = regime_data["regime"]
        regime_conf = regime_data["confidence"]
        
        # Adjust probability based on regime
        regime_prob = 1.0
        if regime == "MANIPULATION":
            regime_prob = 0.4 # Danger
        elif regime == "EXPANSION_REAL":
            regime_prob = 1.2 # Boost
        elif regime == "ACCUMULATION":
            regime_prob = 0.6 # Low vol
            
        cumulative_probability *= regime_prob
        
        result["chain_stages"]["3_REGIME"] = {
             "regime": regime,
             "confidence": regime_conf,
             "is_trap": regime_data["is_trap"]
        }
        all_ml_features["regime_state"] = regime
        chain_log.append(f"R: {regime}")
        
    except Exception as e:
        logger.error(f"[CHAIN] Stage 3 (REGIME) failed: {e}")
        result["chain_stages"]["3_REGIME"] = {"status": "ERROR"}
        regime = "NEUTRAL"
        regime_data = {"is_trap": False}

    # =========================================================================
    # STAGE 4: FORMA (Structure)
    # =========================================================================
    try:
        forma = analyze_forma(h4_candles, h1_candles, m15_candles, current_price, chrono_features)
        forma_prob = forma.get("alignment_score", 0.5)
        cumulative_probability *= forma_prob
        
        result["chain_stages"]["4_FORMA"] = {"bias": forma["bias_conclusion"]}
        all_ml_features.update(forma.get("ml_features", {}))
        chain_log.append(f"F: {forma['bias_conclusion']}")
        market_structure = forma["analysis"].get("H4", {}).get("trend", "NEUTRAL")
    except Exception as e:
        logger.error(f"[CHAIN] Stage 4 (FORMA) failed: {e}")
        cumulative_probability *= 0.5
        market_structure = "NEUTRAL"

    # =========================================================================
    # STAGE 5: ESPACIO (Zones)
    # =========================================================================
    try:
        espacio = analyze_espacio(
            h1_candles or m15_candles, current_price, market_structure, "H1", chrono_features
        )
        espacio_prob = espacio.get("guidance", {}).get("combined_probability", 0.5)
        cumulative_probability *= espacio_prob
        
        result["chain_stages"]["5_ESPACIO"] = {"zone": espacio["analysis"]["premium_discount"]["zone"]}
        all_ml_features.update(espacio.get("ml_features", {}))
        chain_log.append(f"E: {espacio['analysis']['premium_discount']['zone']}")
        poi_status = "IN_ZONE" if espacio["price_in_zone"] else "WAITING"
    except Exception as e:
         logger.error(f"[CHAIN] Stage 5 (ESPACIO) failed: {e}")
         cumulative_probability *= 0.5
         poi_status = "NEUTRAL"

    # =========================================================================
    # STAGE 6: ACELERACIN (Momentum)
    # =========================================================================
    try:
        proposed_direction = d1_direction if d1_direction != "NEUTRAL" else h4_direction
        aceleracion = analyze_aceleracion(
            m5_candles or m15_candles, poi_status, proposed_direction, 14, chrono_features
        )
        aceleracion_prob = aceleracion.get("execution_probability", 0.5)
        cumulative_probability *= aceleracion_prob
        result["chain_stages"]["6_ACELERACION"] = {"mom_type": aceleracion["analysis"]["momentum_type"]}
        all_ml_features.update(aceleracion.get("ml_features", {}))
        chain_log.append(f"A: {aceleracion['analysis']['momentum_type']}")
    except Exception as e:
        logger.error(f"[CHAIN] Stage 6 (ACELERACION) failed: {e}")
        cumulative_probability *= 0.5

    # =========================================================================
    # STAGE 6.5: MSS-5 SENSORY SYNC (Neural Tensors)
    # =========================================================================
    try:
        from app.skills.feat_sensory import calculate_mss5_tensors
        mss5_data = calculate_mss5_tensors(df_m15)
        
        mss5_tensors = mss5_data["tensors"]
        resonance_score = mss5_data["resonance_score"]
        validation_status = mss5_data["validation"]
        
        # Modulate previous probability by resonance
        cumulative_probability *= resonance_score
        
        # Map to ML Features
        all_ml_features["momentum_kinetic_micro"] = mss5_tensors["kinetic_tension"]
        all_ml_features["entropy_coefficient"] = mss5_tensors["entropy_coeff"]
        all_ml_features["cycle_harmonic_phase"] = mss5_tensors["harmonic_phase"]
        all_ml_features["institutional_mass_flow"] = 1.0 if mss5_tensors["mass_flow"] else 0.0
        all_ml_features["volatility_regime_norm"] = mss5_tensors["volatility_norm"]
        all_ml_features["acceptance_ratio"] = mss5_tensors["acceptance_ratio"]
        all_ml_features["wick_stress"] = mss5_tensors["wick_stress"]
        
        chain_log.append(f"S: {validation_status}")
        
    except Exception as e:
        logger.error(f"[CHAIN] Stage 6.5 (SENSORY) failed: {e}")
        cumulative_probability *= 0.6 # Penalty for sensory failure

    # =========================================================================
    # STAGE 7: FUSION LAYER (MTF Integration)
    # =========================================================================
    try:
        signals = {
            "D1": 0.5 + (0.3 if d1_direction == "BULLISH" else (-0.3 if d1_direction == "BEARISH" else 0)),
            "H4": 0.5 + (0.3 if h4_direction == "BULLISH" else (-0.3 if h4_direction == "BEARISH" else 0)),
            "H1": 0.5 + (0.3 if h1_direction == "BULLISH" else (-0.3 if h1_direction == "BEARISH" else 0)),
            "M15": cumulative_probability,
            "M5": cumulative_probability
        }
        hurst_map = {"D1": 0.55, "H4": 0.55, "H1": 0.50, "M15": 0.50, "M5": 0.50}
        
        # Dynamic Heat-based Attention
        # Heat = normalized volatility (example)
        heat_map = {
            "M5": 0.8 if abs(physics_cvd.get("cvd_acceleration", 0)) > 500 else 0.4,
            "M15": 0.6,
            "H1": 0.5,
            "H4": 0.7 if h4_in_killzone else 0.5,
            "D1": 0.4
        }
        
        fusion_prob = mtf_manager.resolve_conflicts(signals, hurst_map, h4_in_killzone, heat_map)
        
        result["chain_stages"]["7_FUSION"] = {
            "fused_prob": round(fusion_prob, 3),
            "attention": "DYNAMIC_HEAT"
        }
        chain_log.append(f"FUSION: {fusion_prob:.2f}")
    except Exception as e:
        logger.error(f"[CHAIN] Stage 7 (FUSION) failed: {e}")
        fusion_prob = cumulative_probability

    # =========================================================================
    # STAGE 8: TTI (TEMPORAL TREND INTENT) - New Filter
    # =========================================================================
    try:
        tti_data = tti_engine.calculate_intent_score(kz_block, h4_direction, regime_data)
        tti_score = tti_data["tti_score"]
        discount = tti_data["discount_factor"]
        
        # Apply discount to probability
        # If intent is low (e.g. Asia), reduce prob
        final_prob_pre_liq = fusion_prob * discount
        
        # If TTI score is high, it validates the trade
        if tti_score > 0.8:
            final_prob_pre_liq = min(0.99, final_prob_pre_liq * 1.1)

        result["chain_stages"]["8_INTENT"] = {
             "tti_score": tti_score,
             "recommendation": tti_data["recommendation"]
        }
        chain_log.append(f"I: {tti_data['recommendation']}")
        
    except Exception as e:
        logger.error(f"[CHAIN] Stage 8 (INTENT) failed: {e}")
        final_prob_pre_liq = fusion_prob
        tti_data = {"recommendation": "ERROR"}

    # =========================================================================
    # STAGE 9: LIQUIDITY PREFLIGHT
    # =========================================================================
    try:
        from app.skills.liquidity import check_liquidity_preflight
        liquidity_ok = await check_liquidity_preflight(symbol)
        liquidity_penalty = 1.0 if liquidity_ok else 0.3
        result["chain_stages"]["9_LIQUIDITY"] = {"ok": liquidity_ok}
    except Exception as e:
        liquidity_penalty = 1.0

    # =========================================================================
    # FINAL DECISION
    # =========================================================================
    final_probability = final_prob_pre_liq * liquidity_penalty
    final_probability = max(0.0, min(1.0, final_probability))
    
    # Action Logic with TTI + Regime Context
    action = "WAIT"
    size = "NONE"
    
    # 1. Regime Override
    if regime == "MANIPULATION" and final_probability < 0.85:
        action = "WAIT_FOR_CONFIRMATION" # Don't trade manipulation easily
    elif regime == "FIX_EVENT":
        action = "AVOID_FIX"
    elif final_probability >= 0.75:
        if tti_data["recommendation"] == "INSTITUTIONAL_EXECUTE":
            action = "EXECUTE_TWIN"
            size = "FULL"
        else:
            action = "EXECUTE_SCALP"
            size = "HALF"
    elif final_probability >= 0.55:
        action = "EXECUTE_SCALP"
        size = "QUARTER"
    elif final_probability >= 0.40:
        action = "PREPARE"
    
    # Direction
    final_direction = proposed_direction if proposed_direction != "NEUTRAL" else "NEUTRAL"
    
    result["final_decision"] = {
        "action": action,
        "direction": final_direction,
        "probability": round(final_probability, 3),
        "size": size,
        "regime": regime
    }
    result["ml_features"] = all_ml_features
    result["trade_params"] = {
        "symbol": symbol,
        "direction": final_direction,
        "entry_price": current_price,
        # ... sizing logic ...
    }
    
    logger.info(f"[FEAT-CHAIN-v7] {action} prob={final_probability:.2f} Regime={regime} | {' '.join(chain_log)}")
    
    # SENIOR AUDIT RECORD
    try:
        from app.core.auditor_senior import auditor_senior
        auditor_senior.record_execution(result["chain_stages"], result["final_decision"], symbol)
    except Exception as e:
        logger.error(f"Failed to record audit: {e}")

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
    """MCP Tool: Full institutional chain v7 (Physics)."""
    return await execute_feat_chain_institucional(
        symbol, server_time_utc, d1_direction, h4_direction, h1_direction,
        h4_candles, h1_candles, m15_candles, m5_candles, current_price,
        has_sweep, news_upcoming
    )
