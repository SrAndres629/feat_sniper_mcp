import asyncio
import logging
import pandas as pd
from typing import Dict, Any, Optional

# FAIL-FAST: Use centralized MT5 from mt5_conn (no silent mocks)
from app.core.mt5_conn import mt5_conn, mt5, MT5_AVAILABLE

if not MT5_AVAILABLE:
    raise ImportError(
        "MetaTrader5 library not found. This module requires a real MT5 connection. "
        "Install: pip install MetaTrader5 (Windows only)"
    )

from app.models.schemas import TradeOrderRequest, ResponseModel, TradeOrderResponse, ErrorDetail
from app.core.validation import OrderValidator

logger = logging.getLogger("MT5_Bridge.Skills.Execution")

# =============================================================================
# MAPEO INTELIGENTE DE RETCODES (SMART ERRORS)
# =============================================================================
RETCODE_HINTS = {
    mt5.TRADE_RETCODE_REQUOTE: {
        "code": "REQUOTE",
        "message": "El precio ha cambiado.",
        "suggestion": "Aumenta la desviacin (slippage) permitida o intenta en un momento de menor volatilidad."
    },
    mt5.TRADE_RETCODE_REJECT: {
        "code": "ORDER_REJECTED",
        "message": "La orden fue rechazada por el broker.",
        "suggestion": "Revisa si el volumen es vlido o si hay restricciones en el smbolo seleccionado."
    },
    mt5.TRADE_RETCODE_CANCEL: {
        "code": "ORDER_CANCELLED",
        "message": "La orden fue cancelada por el servidor.",
        "suggestion": "Verifica la conexin con el servidor del broker e intenta de nuevo."
    },
    mt5.TRADE_RETCODE_NO_MONEY: {
        "code": "INSUFFICIENT_FUNDS",
        "message": "Margen insuficiente para abrir la posicin.",
        "suggestion": "Reduce el volumen del lote o deposita fondos en la cuenta."
    },
    mt5.TRADE_RETCODE_TRADE_DISABLED: {
        "code": "TRADING_DISABLED",
        "message": "El trading est deshabilitado para esta cuenta o smbolo.",
        "suggestion": "Contacta a tu broker o verifica si el mercado est abierto."
    },
    mt5.TRADE_RETCODE_MARKET_CLOSED: {
        "code": "MARKET_CLOSED",
        "message": "El mercado est cerrado actualmente.",
        "suggestion": "Espera a la apertura de la sesin de trading correspondiente."
    },
    mt5.TRADE_RETCODE_INVALID_STOPS: {
        "code": "INVALID_STOPS",
        "message": "Niveles de Stop Loss (SL) o Take Profit (TP) invlidos.",
        "suggestion": "Asegrate de respetar la distancia mnima de puntos (freeze level) dictada por el broker."
    },
    mt5.TRADE_RETCODE_INVALID_VOLUME: {
        "code": "INVALID_VOLUME",
        "message": "El volumen de la orden no es vlido.",
        "suggestion": "Verifica el lote mnimo, el lote mximo y el step de volumen del smbolo."
    }
}

# Retcodes that may succeed on retry (with fresh price and increased slippage)
TRANSIENT_RETCODES = {mt5.TRADE_RETCODE_REQUOTE}

# Mapeo de tipos de rdenes
ACTION_TO_MT5_TYPE = {
    "BUY": mt5.ORDER_TYPE_BUY,
    "SELL": mt5.ORDER_TYPE_SELL,
    "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
    "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
    "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
    "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP
}

async def send_order(order_data: TradeOrderRequest, urgency_score: float = 0.5) -> ResponseModel[TradeOrderResponse]:
    """
    Ejecuta una orden (Mercado o Pendiente) con validaciones de seguridad avanzadas.
    Instrumentado con OTel y Mtricas de Ejecucin.
    
    Args:
        order_data: Datos de la orden.
        urgency_score: 0.0-1.0 (Output Neuronal). >0.9 = Panic/Breakout (Market), <0.6 = Patient (Limit).
    """
    from app.core.observability import obs_engine, tracer
    import time

    start_time = time.time()
    symbol = order_data.symbol
    action = order_data.action.upper()
    volume = order_data.volume
    price = order_data.price
    sl = order_data.sl
    tp = order_data.tp

    with tracer.start_as_current_span("mt5_send_order") as span:
        span.set_attribute("symbol", symbol)
        span.set_attribute("action", action)
        span.set_attribute("volume", volume)

        # 0.0 CIRCUIT BREAKER CHECK
        from app.services.circuit_breaker import circuit_breaker
        if not circuit_breaker.can_execute():
            obs_engine.track_order(symbol, action, "CIRCUIT_OPEN")
            return ResponseModel(
                status="error",
                error=ErrorDetail(code="CIRCUIT_OPEN", message="System is in emergency isolation mode.", suggestion="Check logs for recent failures.")
            )

        # 0.0.1 TTL VALIDATION (Race Condition Fix)
        from app.core.config import settings
        if order_data.decision_ts:
            age_ms = (time.time() * 1000) - order_data.decision_ts
            if age_ms > settings.DECISION_TTL_MS:
                obs_engine.track_order(symbol, action, "TTL_EXPIRED")
                logger.warning(f"⏰ Order rejected: decision age {age_ms:.0f}ms > TTL {settings.DECISION_TTL_MS}ms")
                return ResponseModel(
                    status="error",
                    error=ErrorDetail(
                        code="TTL_EXPIRED",
                        message=f"Decision too old ({age_ms:.0f}ms > {settings.DECISION_TTL_MS}ms TTL)",
                        suggestion="Reduce latency between ML inference and order execution."
                    )
                )
            span.set_attribute("decision_age_ms", age_ms)

        # 0.2 INSTITUTIONAL RISK VETO
        from app.services.risk_engine import risk_engine
        if not await risk_engine.check_drawdown_limit():
            obs_engine.track_order(symbol, action, "RISK_VETO")
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code="RISK_VETO",
                    message="Operación bloqueada por límite de drawdown diario.",
                    suggestion="Espera a que se recupere la equidad o ajusta el parámetro MAX_DAILY_DRAWDOWN_PERCENT."
                )
            )

        # 0.1 LIQUIDITY PRE-FLIGHT (Sniper 2.0)
        from app.skills.liquidity import check_liquidity_preflight
        if not await check_liquidity_preflight(symbol):
            obs_engine.track_order(symbol, action, "LOW_LIQUIDITY")
            return ResponseModel(
                status="error",
                error=ErrorDetail(code="LOW_LIQUIDITY", message=f"Liquidity for {symbol} is below institutional grade.", suggestion="Avoid entries during market roll-over or news.")
            )

        # 0.2.1 VOLATILITY GUARD (The Quant Flow - Flash Crash Guard)
        try:
            from app.skills.indicators import calculate_atr
            rates = await mt5_conn.execute(mt5.copy_rates_from_pos, symbol, mt5.TIMEFRAME_M1, 0, 50)
            if rates is not None and len(rates) > 20:
                df_rates = pd.DataFrame(rates)
                current_atr = calculate_atr(df_rates)
                avg_atr = (df_rates['high'] - df_rates['low']).mean()
                
                if current_atr > avg_atr * settings.VOLATILITY_THRESHOLD:
                    obs_engine.track_order(symbol, action, "VOLATILITY_VETO")
                    logger.warning(f"🚫 VOLATILITY GUARD: Stress too high ({current_atr:.5f} > {avg_atr * settings.VOLATILITY_THRESHOLD:.5f})")
                    return ResponseModel(
                        status="error",
                        error=ErrorDetail(code="VOLATILITY_VETO", message="Market stress (ATR) exceeds safety threshold.", suggestion="Wait for flash crash or news spike to stabilize.")
                    )
        except Exception as e:
            logger.error(f"VolatilityGuard Error: {e}")

        # 0.2.2 SPREAD FILTER (The Quant Flow - Liquidity Guard)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        if symbol_info and hasattr(symbol_info, 'spread'):
            if symbol_info.spread > settings.SPREAD_MAX_PIPS:
                 obs_engine.track_order(symbol, action, "SPREAD_VETO")
                 logger.warning(f"🚫 SPREAD FILTER: Liquidity too low (Spread: {symbol_info.spread} pts > {settings.SPREAD_MAX_PIPS} limit)")
                 return ResponseModel(
                    status="error",
                    error=ErrorDetail(code="SPREAD_VETO", message=f"Spread too high ({symbol_info.spread} points).", suggestion="Avoid trading during roll-over or low liquidity sessions.")
                )

        # Adaptive Volume Calculation (if volume == 0 or not provided)
        if volume <= 0 and sl:
            tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
            exec_price = tick.ask if action == "BUY" else tick.bid
            symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
            sl_points = abs(exec_price - sl) / symbol_info.point
            
            # 1.0.3 POM: Use calculate_dynamic_lot to internalize CB Levels & Multipliers
            # Fetch volatility from symbol ATR if available
            volatility = 0.0
            try:
                symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
                if symbol_info:
                    volatility = float(symbol_info.spread) / 100.0  # Spread as volatility proxy
            except Exception:
                volatility = 0.0 # Default fallback if spread fetch fails
            
            volume = await risk_engine.calculate_dynamic_lot(
                confidence=urgency_score, # Mapping urgency to confidence for sizing
                volatility=volatility,
                symbol=symbol,
                sl_points=int(sl_points),
                black_swan_multiplier=1.0
            )
            span.set_attribute("adaptive_volume", volume)

        # 1. VALIDACIN INTELIGENTE (SMART GUARDRAILS)
        is_valid, error_msg = await OrderValidator.validate_order(
            symbol=symbol,
            volume=volume,
            action=action,
            price=price,
            sl=sl,
            tp=tp
        )
        
        if not is_valid:
            obs_engine.track_order(symbol, action, "VALIDATION_FAILED")
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code="VALIDATION_FAILED",
                    message=error_msg,
                    suggestion="Corrige los parmetros de la orden basados en los lmites del broker."
                )
            )

        # 2. VALIDACIN DE MARGEN
        has_margin, margin_err = await OrderValidator.validate_margin(symbol, volume, action)
        if not has_margin:
            obs_engine.track_order(symbol, action, "INSUFFICIENT_MARGIN")
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code="INSUFFICIENT_MARGIN",
                    message=margin_err,
                    suggestion="Reduce el volumen o deposita fondos."
                )
            )

        # 3. CALCULO DE DESVIACIN DINMICA (SLIPPAGE)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        spread = symbol_info.spread
        
        # SMART DEVIATION: Ms urgencia = Ms slippage permitido
        base_deviation = max(20, int(spread * 1.5))
        if urgency_score > settings.URGENCY_THRESHOLD:
            dynamic_deviation = base_deviation * 3  # Aceptamos pagar spread por entrar YA
        else:
            dynamic_deviation = base_deviation 

        # 4. SMART EXECUTION LOGIC (Hybrid Routing)
        order_type = ACTION_TO_MT5_TYPE.get(action)
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        current_ask = tick.ask
        current_bid = tick.bid
        
        # Determine Routing based on Urgency
        if "LIMIT" not in action and "STOP" not in action:
            # Es una orden a mercado (potencialmente)
            
            if urgency_score > settings.URGENCY_THRESHOLD:
                 # HIGH URGENCY -> FORCE MARKET EXECUTION
                 exec_price = current_ask if action == "BUY" else current_bid
                 logger.info(f"🔥 HIGH URGENCY ({urgency_score:.2f}) -> FORCING MARKET EXECUTION")
            
            elif urgency_score < 0.5:
                 # LOW URGENCY (< 0.5) -> CONVERT TO LIMIT (Save Spread cost)
                 # Intentamos entrar 'dentro' del spread
                 spread_val = current_ask - current_bid
                 if action == "BUY":
                     # Buy Limit un poco abajo del ask (mid price)
                     exec_price = current_bid + (spread_val * 0.25) 
                     order_type = mt5.ORDER_TYPE_BUY_LIMIT
                     logger.info(f"🐢 LOW URGENCY ({urgency_score:.2f}) -> CONVERTING TO LIMIT @ {exec_price}")
                 else:
                     # Sell Limit un poco arriba del bid
                     exec_price = current_ask - (spread_val * 0.25)
                     order_type = mt5.ORDER_TYPE_SELL_LIMIT
                     logger.info(f"🐢 LOW URGENCY ({urgency_score:.2f}) -> CONVERTING TO LIMIT @ {exec_price}")
            
            else:
                # NORMAL EXECUTION
                exec_price = current_ask if action == "BUY" else current_bid
                
        else:
            # Es pendiente explicita
            if not price:
                 return ResponseModel(status="error", error=ErrorDetail(code="PRICE_REQUIRED", message="Price missing for pending order"))
            exec_price = price

        # Detectar Filling Mode
        filling_mode = mt5.ORDER_FILLING_RETURN
        if "LIMIT" not in action and "STOP" not in action:
            s_filling = symbol_info.filling_mode
            if s_filling & mt5.SYMBOL_FILLING_IOC:
                filling_mode = mt5.ORDER_FILLING_IOC
            elif s_filling & mt5.SYMBOL_FILLING_FOK:
                filling_mode = mt5.ORDER_FILLING_FOK
                
        request = {
            "action": mt5.TRADE_ACTION_DEAL if "LIMIT" not in action and "STOP" not in action else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": exec_price,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "deviation": dynamic_deviation,
            "magic": 234000,
            "comment": order_data.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        logger.info(f"Enviando orden {action} de {volume} lots en {symbol} a {exec_price} (Dev: {dynamic_deviation})")

        # 5. ATOMIC EXECUTION WITH RETRY (Race Condition & REQUOTE Fix)
        is_buy = (action == "BUY" or action == "BUY_LIMIT" or action == "BUY_STOP")
        
        def place_order_atomic(sym, req_dict, buy_flag):
            """
            Atomic tick+order_send: Fetches current price and sends order
            under a single lock acquisition to eliminate race conditions.
            """
            tick = mt5.symbol_info_tick(sym)
            if not tick:
                return None
            # Update price with fresh tick data
            req_dict["price"] = tick.ask if buy_flag else tick.bid
            return mt5.order_send(req_dict)
        
        result = None
        last_retcode = None
        
        for attempt in range(1, settings.MAX_ORDER_RETRIES + 1):
            result = await mt5_conn.execute_atomic(
                place_order_atomic, symbol, request, is_buy
            )
            
            if result is None:
                # Critical MT5 error
                break
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                span.set_attribute("attempts", attempt)
                break  # Success!
            
            last_retcode = result.retcode
            
            # Check if retcode is transient (worth retrying)
            if result.retcode in TRANSIENT_RETCODES and attempt < settings.MAX_ORDER_RETRIES:
                backoff_ms = settings.RETRY_BACKOFF_BASE_MS * (3 ** (attempt - 1))
                # Increase slippage tolerance on each retry
                request["deviation"] = int(dynamic_deviation * (1 + 0.5 * attempt))
                logger.warning(f"🔄 REQUOTE attempt {attempt}/{settings.MAX_ORDER_RETRIES}, "
                             f"backoff {backoff_ms}ms, new deviation: {request['deviation']}")
                await asyncio.sleep(backoff_ms / 1000)
                continue
            
            # Terminal error or max retries reached
            break

        if result is None:
            circuit_breaker.record_failure()
            obs_engine.track_order(symbol, action, "MT5_ERROR")
            error = await mt5_conn.execute(mt5.last_error)
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code=f"MT5_INTERNAL_ERROR_{error[0] if error else 'UNKNOWN'}",
                    message=error[1] if error else "Connection lost",
                    suggestion="Error crtico de comunicacin con MetaTrader 5."
                )
            )

        # 6. MANEJO DE RESULTADO
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode in [mt5.TRADE_RETCODE_ERROR, mt5.TRADE_RETCODE_NO_MONEY]:
                circuit_breaker.record_failure()

            obs_engine.track_order(symbol, action, f"FAILED_{result.retcode}")
            hint = RETCODE_HINTS.get(result.retcode, {
                "code": f"RETCODE_{result.retcode}",
                "message": f"Fallo en la ejecucin: {result.comment}",
                "suggestion": "Consulta el log de MT5 para ms detalles tcnicos."
            })
            
            return ResponseModel(
                status="error",
                error=ErrorDetail(**hint)
            )

        # xito
        circuit_breaker.record_success()
        obs_engine.track_order(symbol, action, "SUCCESS")
        
        duration = time.time() - start_time
        obs_engine.track_latency("full_order_cycle", symbol, duration)
        
        span.set_attribute("ticket", result.order)
        span.set_attribute("exec_price", result.price)

        return ResponseModel(
            status="success",
            data=TradeOrderResponse(
                ticket=result.order,
                symbol=symbol,
                action=action,
                price=result.price,
                volume=result.volume
            )
        )

# =============================================================================
# TWIN-ENGINE HYBRID EXECUTION
# =============================================================================

async def execute_twin_trade(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Protocolo Twin-Entry: Abre 2 rdenes simultneas con diferentes objetivos.
    - Orden A (Scalp): TP corto ($2 target), Magic = MAGIC_SCALP
    - Orden B (Swing): TP largo ($10+ target), Magic = MAGIC_SWING
    """
    from app.core.config import settings
    from app.services.risk_engine import risk_engine
    
    symbol = signal.get("symbol", "XAUUSD")
    direction = signal.get("direction", "BUY").upper()
    confidence = signal.get("confidence", 0)
    
    # Verificar si podemos abrir dual trade
    allocation = await risk_engine.get_capital_allocation()
    
    results = {
        "scalp_ticket": None,
        "swing_ticket": None,
        "mode": "TWIN" if allocation["can_dual"] else "SCALP_ONLY",
        "allocation": allocation
    }
    
    # Obtener precio actual y calcular TPs
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
    
    if not tick or not symbol_info:
        return {"status": "error", "message": "Failed to get market data"}
    
    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value
    
    # Calcular pips para $2 y $10 profit con 0.01 lotes
    # Profit = Pips * Tick_Value * Volume
    # Pips = Target_USD / (Tick_Value * Volume)
    scalp_pips = int(settings.SCALP_TARGET_USD / (tick_value * 0.01)) if tick_value > 0 else 200
    swing_pips = int(settings.SWING_TARGET_USD / (tick_value * 0.01)) if tick_value > 0 else 1000
    
    # SL ajustado para micro cuenta (~15-20 pips)
    sl_pips = 20
    
    exec_price = tick.ask if direction == "BUY" else tick.bid
    
    if direction == "BUY":
        scalp_tp = exec_price + (scalp_pips * point)
        swing_tp = exec_price + (swing_pips * point)
        sl = exec_price - (sl_pips * point)
    else:
        scalp_tp = exec_price - (scalp_pips * point)
        swing_tp = exec_price - (swing_pips * point)
        sl = exec_price + (sl_pips * point)
    
    # --- ORDEN A: SCALP (El Sueldo) ---
    # Smart Sizing: High Confidence (0.9) for Scalp
    scalp_vol = await risk_engine.calculate_dynamic_lot(0.95, 0.01, symbol, sl_pips)
    
    scalp_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": scalp_vol,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": exec_price,
        "sl": sl,
        "tp": scalp_tp,
        "deviation": 20,
        "magic": settings.MAGIC_SCALP,
        "comment": "TWIN_SCALP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    
    scalp_result = await mt5_conn.execute(mt5.order_send, scalp_request)
    if scalp_result and scalp_result.retcode == mt5.TRADE_RETCODE_DONE:
        results["scalp_ticket"] = scalp_result.order
        logger.info(f" SCALP ENTRY: Ticket {scalp_result.order} | Vol: {scalp_vol} | TP: ${settings.SCALP_TARGET_USD}")
    
    # --- ORDEN B: SWING (La Riqueza) --- Solo si hay margen
    if allocation["can_dual"]:
        # Smart Sizing: Slightly lower confidence (0.8) for Swing to conserve risk
        swing_vol = await risk_engine.calculate_dynamic_lot(0.85, 0.01, symbol, sl_pips)
        
        swing_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": swing_vol,
            "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": exec_price,
            "sl": sl,
            "tp": swing_tp,
            "deviation": 20,
            "magic": settings.MAGIC_SWING,
            "comment": "TWIN_SWING",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        swing_result = await mt5_conn.execute(mt5.order_send, swing_request)
        if swing_result and swing_result.retcode == mt5.TRADE_RETCODE_DONE:
            results["swing_ticket"] = swing_result.order
            logger.info(f" SWING ENTRY: Ticket {swing_result.order} | TP: ${settings.SWING_TARGET_USD}")
    else:
        logger.warning(" SURVIVAL MODE: Only Scalp opened due to margin constraints")
    
    results["status"] = "success" if results["scalp_ticket"] else "failed"
    return results

# =============================================================================
# INSTITUTIONAL PENDING ORDER WRAPPERS (Jules' Request)
# =============================================================================

async def place_limit_order(symbol: str, action: str, volume: float, price: float, 
                           sl: float = 0.0, tp: float = 0.0, comment: str = "") -> ResponseModel:
    """
    Coloca una orden LIMIT (Buy Limit / Sell Limit).
    Usage: Para entrar a mejor precio (Pullbacks, FVG retest).
    Validation: Precio debe ser MEJOR que el actual.
    """
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    if not tick: return ResponseModel(status="error", error=ErrorDetail(code="NO_TICK", message="No market data"))
    
    current_price = tick.ask if action == "BUY" else tick.bid
    
    # Validate Limit Logic
    if action == "BUY" and price >= current_price:
        return ResponseModel(status="error", error=ErrorDetail(code="INVALID_LIMIT", message=f"Buy Limit {price} must be BELOW current {current_price}"))
    if action == "SELL" and price <= current_price:
        return ResponseModel(status="error", error=ErrorDetail(code="INVALID_LIMIT", message=f"Sell Limit {price} must be ABOVE current {current_price}"))

    req = TradeOrderRequest(
        symbol=symbol, action=f"{action}_LIMIT", volume=volume, 
        price=price, sl=sl, tp=tp, comment=comment
    )
    # Urgency < 0.5 for Limits (Patient)
    return await send_order(req, urgency_score=0.3)

async def place_stop_order(symbol: str, action: str, volume: float, price: float, 
                          sl: float = 0.0, tp: float = 0.0, comment: str = "") -> ResponseModel:
    """
    Coloca una orden STOP (Buy Stop / Sell Stop).
    Usage: Breakout entries (BOS confirmation).
    Validation: Precio debe ser PEOR que el actual.
    """
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    if not tick: return ResponseModel(status="error", error=ErrorDetail(code="NO_TICK", message="No market data"))
    
    current_price = tick.ask if action == "BUY" else tick.bid
    
    # Validate Stop Logic
    if action == "BUY" and price <= current_price:
        return ResponseModel(status="error", error=ErrorDetail(code="INVALID_STOP", message=f"Buy Stop {price} must be ABOVE current {current_price}"))
    if action == "SELL" and price >= current_price:
        return ResponseModel(status="error", error=ErrorDetail(code="INVALID_STOP", message=f"Sell Stop {price} must be BELOW current {current_price}"))

    req = TradeOrderRequest(
        symbol=symbol, action=f"{action}_STOP", volume=volume, 
        price=price, sl=sl, tp=tp, comment=comment
    )
    # Urgency > 0.6 for Stops (Momentum)
    return await send_order(req, urgency_score=0.7)
