import logging
import sys
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from fastmcp import FastMCP
from contextlib import asynccontextmanager

# Neural Pulse: Structured Logger
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', 'SYSTEM')
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(JSONFormatter())
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger("NeuralPulse_Brain")

import time
import asyncio
import contextvars

# Global Context for Correlation
correlation_id_ctx = contextvars.ContextVar("correlation_id", default="SYSTEM")

def get_correlation_id():
    return correlation_id_ctx.get()

# Neural Pulse: Tool Observer Designer
def pulse_observer(func):
    import functools
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cid = str(uuid.uuid4())
            token = correlation_id_ctx.set(cid)
            logger.info(f"Neural Pulse [START]: {func.__name__}", extra={"correlation_id": cid})
            start_ts = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = (time.time() - start_ts) * 1000
                logger.info(f"Neural Pulse [SUCCESS]: {func.__name__} | Latency: {latency:.2f}ms", extra={"correlation_id": cid})
                return result
            except Exception as e:
                logger.error(f"Neural Pulse [FAILURE]: {func.__name__} | Error: {str(e)}", extra={"correlation_id": cid}, exc_info=True)
                raise
            finally:
                correlation_id_ctx.reset(token)
        return wrapper
    else:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cid = str(uuid.uuid4())
            token = correlation_id_ctx.set(cid)
            logger.info(f"Neural Pulse [START]: {func.__name__}", extra={"correlation_id": cid})
            start_ts = time.time()
            try:
                result = func(*args, **kwargs)
                latency = (time.time() - start_ts) * 1000
                logger.info(f"Neural Pulse [SUCCESS]: {func.__name__} | Latency: {latency:.2f}ms", extra={"correlation_id": cid})
                return result
            except Exception as e:
                logger.error(f"Neural Pulse [FAILURE]: {func.__name__} | Error: {str(e)}", extra={"correlation_id": cid}, exc_info=True)
                raise
            finally:
                correlation_id_ctx.reset(token)
        return wrapper

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logger.warning("MetaTrader5 module not found found. Running in Linux/Docker mode? Some local-only features will be disabled.")


# Importar lgica de negocio de la app
from app.core.mt5_conn import mt5_conn
from app.skills import (
    market, vision, execution, trade_mgmt, 
    indicators, history, calendar, 
    quant_coder, custom_loader,
    tester, unified_model, remote_compute # Nuevos mdulos
)
from app.models.schemas import (
    MarketDataRequest, TradeOrderRequest, PositionManageRequest, 
    IndicatorRequest, HistoryRequest, CalendarRequest, MQL5CodeRequest
)
from app.ml.data_collector import data_collector

from app.core.zmq_bridge import zmq_bridge
from app.skills.advanced_analytics import advanced_analytics
from app.services.supabase_sync import supabase_sync

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Maneja el ciclo de vida institucional: MT5 + ZMQ Bridge."""
    logger.info("Iniciando infraestructura institucional...")
    print("DEBUG: Entrando en app_lifespan", flush=True)
    await mt5_conn.startup()
    print("DEBUG: mt5_conn.startup() COMPLETADO", flush=True)
    
    # Iniciar ZMQ Bridge para Streaming de baja latencia
    async def on_zmq_message(data):
        print(f"TRACE: Recibido mensaje tipo {data.get('type')} - Simbolo: {data.get('symbol')}", flush=True)
        msg_type = data.get("type", "signal")
        
        if msg_type == "tick":
            # 1. Local Collection (SQL Lite for ML training)
            # collect_sample expect symbols, candle, indicators
            # data typically has symbol and candle info
            import asyncio
            symbol = data.get("symbol")
            candle = data.get("candle", data)
            indicators_data = data.get("indicators", {})
            
            # Non-blocking local collection
            data_collector.collect(symbol, candle, indicators_data)
            
            # 2. Institutional Sync (Supabase market_ticks)
            asyncio.create_task(supabase_sync.log_tick(data))
            
        elif msg_type == "signal":
            logger.info(f"Seal recibida va ZMQ: {data}")
            import asyncio
            asyncio.create_task(supabase_sync.log_signal(data))
        
    print(f"DEBUG: Intentando iniciar ZMQ Bridge en callback {on_zmq_message}", flush=True)
    await zmq_bridge.start(on_zmq_message)
    print("DEBUG: ZMQ Bridge iniciado VOLANDO", flush=True)
    
    # 3. Background Risk Monitor
    async def monitor_risk_loop():
        logger.info(" Monitor de Riesgo Institucional ACTIVO")
        from app.services.risk_engine import risk_engine
        while True:
            try:
                # Verificar Drawdown para Auto-Stop
                if not await risk_engine.check_drawdown_limit():
                    logger.critical(" CIRCUIT BREAKER TRIPPED - Risk Limit Exceeded!")
                
                # Aplicar Trailing Stop institucional a todas las posiciones
                positions = await mt5_conn.execute(mt5.positions_get)
                if positions:
                    from app.services.telemetry import telemetry
                    for pos in positions:
                        # Monitorar profit para telemetria n8n
                        if pos.profit > 100: # Exemplo: alerta de lucro alto
                             await telemetry.send_to_n8n({
                                 "asset": pos.symbol,
                                 "event": "PROFIT_UPDATE",
                                 "profit": pos.profit,
                                 "ticket": pos.ticket
                             })
                             
                        # Agora usa ATR dinmico (limite configurado nas settings)
                        await risk_engine.apply_trailing_stop(pos.symbol, pos.ticket)
            except Exception as e:
                logger.error(f"Error in Risk Monitor: {e}")
            await asyncio.sleep(5) # Revisin cada 5 segundos

    monitor_task = asyncio.create_task(monitor_risk_loop())
    
    try:
        yield
    finally:
        logger.info("Cerrando infraestructura...")
        monitor_task.cancel()
        await zmq_bridge.stop()
        await mt5_conn.shutdown()

# Inicializar FastMCP
mcp = FastMCP("MT5_Neural_Sentinel", lifespan=app_lifespan)

# Registrar Skills de Indicadores Propios
custom_loader.register_custom_skills(mcp)

# Registrar Skills de Cmputo Remoto (NEXUS)
remote_compute.register_remote_skills(mcp)

# =============================================================================
# PILLAR 1: CRTEX DE COMPILACIN (Code Builder)
# =============================================================================

@mcp.tool()
@pulse_observer
async def create_and_compile_indicator(name: str, code: str, compile: bool = True):
    """
    Escribe y compila un nuevo indicador .mq5. 
    Retorna el log de compilacin para auto-correccin.
    """
    req = MQL5CodeRequest(name=name, code=code, compile=compile)
    return await quant_coder.create_native_indicator(req)

# =============================================================================
# PILLAR 2: OJO DE HALCN (Data Bridge)
# =============================================================================

@mcp.tool()
@pulse_observer
async def get_market_snapshot(symbol: str, timeframe: str = "M5"):
    """
    Radiografa completa del mercado: Precio, Vela actual, Volatilidad y Cuenta.
    Ideal para decisiones de alta frecuencia.
    """
    return await market.get_market_snapshot(symbol, timeframe)

@mcp.tool()
async def get_candles(symbol: str, timeframe: str = "M5", n_candles: int = 100):
    """Obtiene velas histricas (OHLCV)."""
    return await market.get_candles(symbol, timeframe, n_candles)

@mcp.tool()
async def get_market_panorama(resize_factor: float = 0.75):
    """Captura visual del grfico."""
    return await vision.capture_panorama(resize_factor=resize_factor)

@mcp.tool()
@pulse_observer
async def get_trade_decision(symbol: str, timeframe: str = "M5"):
    """
    Decisin de trading integrada para N8N.
    
    Combina ML (GBM/LSTM) + FEAT Score + FSM State + Indicadores
    para generar una seal unificada con contexto completo.
    
    Returns:
        JSON con signal, confidence, market_state, data_context
    """
    from datetime import datetime
    
    # 1. Obtener snapshot del mercado
    snapshot = await market.get_market_snapshot(symbol, timeframe)
    
    # 2. Obtener prediccin ML si disponible
    ml_prediction = None
    try:
        from app.ml.ml_engine import ml_engine
        features = {
            "close": snapshot.get("price", {}).get("bid", 0),
            "open": snapshot.get("current_candle", {}).get("open", 0),
            "high": snapshot.get("current_candle", {}).get("high", 0),
            "low": snapshot.get("current_candle", {}).get("low", 0),
            "volume": snapshot.get("current_candle", {}).get("volume", 0),
            "rsi": 50.0,  # Default, se actualiza con indicadores reales
            "ema_fast": snapshot.get("price", {}).get("bid", 0),
            "ema_slow": snapshot.get("price", {}).get("bid", 0),
            "ema_spread": 0,
            "feat_score": 0.0,
            "fsm_state": 0,
            "atr": snapshot.get("volatility", {}).get("atr", 0.001),
            "compression": 0.5,
            "liquidity_above": 0,
            "liquidity_below": 0
        }
        ml_prediction = ml_engine.ensemble_predict(features)
    except Exception as e:
        logger.warning(f"ML prediction unavailable: {e}")
    
    # 3. Construir seal
    confidence = 0.5
    signal = "WAIT"
    
    if ml_prediction and ml_prediction.get("p_win") is not None:
        p_win = ml_prediction["p_win"]
        confidence = abs(p_win - 0.5) * 2
        
        if p_win > 0.65:
            signal = "BUY"
        elif p_win < 0.35:
            signal = "SELL"
        else:
            signal = "WAIT"
            
        # Penalizar si hay anomala
        if ml_prediction.get("is_anomaly"):
            confidence *= 0.5
            signal = "WAIT"  # No operar en manipulacin
    
    # 4. Determinar estado de mercado (FSM)
    volatility = snapshot.get("volatility", {})
    vol_status = volatility.get("volatility_status", "NORMAL")
    
    if vol_status == "EXTREME":
        market_state = "MANIPULATION"
    elif vol_status == "HIGH":
        market_state = "EXPANSION"
    elif vol_status == "LOW":
        market_state = "ACCUMULATION"
    else:
        market_state = "DISTRIBUTION"
    
    # 5. Construir contexto para N8N
    data_context = {
        "price": snapshot.get("price", {}),
        "volatility": {
            "atr": volatility.get("atr"),
            "atr_pips": volatility.get("atr_pips"),
            "status": vol_status,
            "spread_points": volatility.get("spread_points")
        },
        "current_candle": snapshot.get("current_candle", {}),
        "ml_source": ml_prediction.get("source") if ml_prediction else "NONE",
        "is_anomaly": ml_prediction.get("is_anomaly", False) if ml_prediction else False
    }
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": round(confidence, 3),
        "market_state": market_state,
        "data_context": data_context,
        "timestamp": datetime.utcnow().isoformat(),
        "execution_enabled": ml_prediction.get("execution_enabled", False) if ml_prediction else False
    }

@mcp.tool()
@pulse_observer
async def get_full_market_context(symbol: str, timeframe: str = "M5"):
    """
     SUPER ENDPOINT PARA N8N (SSH Gateway).
    
    Consolida TODO en un nico JSON para que el Agente N8N tome decisiones
    basadas en la estrategia FEAT completa, no en datos crudos.
    
    Incluye:
    - raw_data: OHLCV actual
    - indicators: FEAT, PVP, EMAs
    - ml_insight: Prediccin + Explicabilidad
    - memory_context: Recuerdos RAG relevantes
    - strategy_guidance: Prompt sugerido para N8N
    """
    from datetime import datetime
    
    # 1. DATOS CRUDOS
    snapshot = await market.get_market_snapshot(symbol, timeframe)
    price_data = snapshot.get("price", {})
    candle = snapshot.get("current_candle", {})
    volatility = snapshot.get("volatility", {})
    
    raw_data = {
        "open": candle.get("open"),
        "high": candle.get("high"),
        "low": candle.get("low"),
        "close": candle.get("close"),
        "volume": candle.get("volume"),
        "bid": price_data.get("bid"),
        "ask": price_data.get("ask"),
        "spread_points": volatility.get("spread_points")
    }
    
    # 2. INDICADORES FEAT
    indicators = {
        "feat": {
            "score": 0.0,  # Placeholder - viene de MT5 va ZMQ
            "form": {"bos": False, "choch": False, "intent_candle": False},
            "space": {"at_zone": False, "proximity": 0.0, "zone_type": "NONE"},
            "acceleration": {
                "velocity": 0.0,
                "momentum": 0.0,
                "rsi": 50.0,
                "is_exhausted": False
            },
            "time": {
                "is_killzone": False,
                "session": "UNKNOWN",
                "is_london_kz": False,
                "is_ny_kz": False
            }
        },
        "ema_layers": {
            "micro": {"compression": 0.5, "slope": 0.0},
            "operational": {"slope": 0.0},
            "macro": {"slope": 0.0},
            "bias": {"slope": 0.0, "direction": "NEUTRAL"}
        },
        "pvp": {
            "poc": None,  # Point of Control
            "vah": None,  # Value Area High
            "val": None,  # Value Area Low
            "distance_to_poc": None
        },
        "volatility": {
            "atr": volatility.get("atr"),
            "atr_pips": volatility.get("atr_pips"),
            "status": volatility.get("volatility_status", "NORMAL")
        },
        "session": _get_market_session(),
        "fsm_state": "CALIBRATING"
    }
    
    # 3. ML INSIGHT CON EXPLICABILIDAD
    ml_insight = {
        "model": "NONE",
        "prediction": "WAIT",
        "win_probability": 0.5,
        "anomaly_score": 0.0,
        "is_anomaly": False,
        "top_drivers": [],
        "explanation": "ML models not loaded - collecting training data"
    }
    
    try:
        from app.ml.ml_engine import ml_engine
        
        features = {
            "close": raw_data.get("close", 0) or 0,
            "open": raw_data.get("open", 0) or 0,
            "high": raw_data.get("high", 0) or 0,
            "low": raw_data.get("low", 0) or 0,
            "volume": raw_data.get("volume", 0) or 0,
            "rsi": indicators["feat"]["acceleration"]["rsi"],
            "ema_fast": raw_data.get("close", 0) or 0,
            "ema_slow": raw_data.get("close", 0) or 0,
            "ema_spread": 0,
            "feat_score": indicators["feat"]["score"],
            "fsm_state": 0,
            "atr": volatility.get("atr", 0.001) or 0.001,
            "compression": 0.5,
            "liquidity_above": 0,
            "liquidity_below": 0
        }
        
        pred = ml_engine.ensemble_predict(features)
        
        ml_insight = {
            "model": pred.get("source", "NONE"),
            "prediction": pred.get("prediction", "WAIT"),
            "win_probability": pred.get("p_win", 0.5),
            "anomaly_score": pred.get("anomaly_score", 0.0),
            "is_anomaly": pred.get("is_anomaly", False),
            "top_drivers": _get_top_drivers(features, pred),
            "explanation": _generate_explanation(pred, indicators)
        }
    except Exception as e:
        logger.warning(f"ML insight unavailable: {e}")
    
    # 4. MEMORIA RAG
    memory_context = []
    try:
        from app.services.rag_memory import rag_memory
        memories = rag_memory.search(f"{symbol} trading pattern", limit=3)
        memory_context = [
            {"text": m["text"][:200], "relevance": m["score"]}
            for m in memories
        ]
    except Exception as e:
        logger.debug(f"RAG unavailable: {e}")
    
    # 5. GUA PARA EL AGENTE N8N
    strategy_guidance = {
        "system_prompt_suggestion": f"""
You are analyzing {symbol} on {timeframe} timeframe.
Current market state: {indicators['fsm_state']}
Volatility: {indicators['volatility']['status']}

FEAT Strategy Rules:
- Only trade during killzones (London/NY overlap)
- Wait for FSM state = EXPANSION for entries
- Respect PVP levels (POC, VAH, VAL) as support/resistance
- FEAT Score > 70 = Strong signal
- Anomaly = Potential manipulation, avoid trading

Based on the data provided, decide: BUY, SELL, or WAIT.
        """.strip(),
        "decision_checklist": [
            "Is it a killzone? (London or NY session)",
            "Is FSM in EXPANSION or ready to expand?",
            "Is price at a significant PVP level?",
            "Is FEAT Score > 70?",
            "Is ML win_probability > 0.6?",
            "Is anomaly_score < 0.5? (no manipulation)"
        ]
    }
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "raw_data": raw_data,
        "indicators": indicators,
        "ml_insight": ml_insight,
        "memory_context": memory_context,
        "strategy_guidance": strategy_guidance
    }


def _get_top_drivers(features: dict, prediction: dict) -> list:
    """Extrae los factores principales que influyen en la prediccin."""
    drivers = []
    
    if features.get("rsi", 50) < 30:
        drivers.append("RSI_Oversold")
    elif features.get("rsi", 50) > 70:
        drivers.append("RSI_Overbought")
        
    if features.get("feat_score", 0) > 70:
        drivers.append("FEAT_Strong_Signal")
        
    if features.get("ema_spread", 0) > 0:
        drivers.append("EMA_Bullish_Cross")
    elif features.get("ema_spread", 0) < 0:
        drivers.append("EMA_Bearish_Cross")
        
    if prediction.get("is_anomaly"):
        drivers.append("ANOMALY_DETECTED")
        
    return drivers[:5]


def _generate_explanation(prediction: dict, indicators: dict) -> str:
    """Genera explicacin legible para el agente N8N."""
    p_win = prediction.get("p_win", 0.5)
    source = prediction.get("source", "NONE")
    
    if source == "NONE":
        return "No ML models available. Decision based on rules only."
        
    direction = "LONG" if p_win > 0.5 else "SHORT"
    confidence = "high" if abs(p_win - 0.5) > 0.3 else "moderate" if abs(p_win - 0.5) > 0.15 else "low"
    
    return f"ML model ({source}) predicts {direction} with {confidence} confidence ({p_win:.1%} win probability)."

def _get_market_session() -> dict:
    """Detecta la sesin actual y killzones institucionales (UTC)."""
    now = datetime.utcnow()
    hour = now.hour
    
    # Killzones (UTC)
    london_kz = 7 <= hour <= 10
    ny_kz = 12 <= hour <= 15
    
    session = "ASIAN"
    if 8 <= hour <= 16: session = "LONDON"
    elif 13 <= hour <= 21: session = "NEW_YORK"
    
    return {
        "name": session,
        "is_killzone": london_kz or ny_kz,
        "is_london_kz": london_kz,
        "is_ny_kz": ny_kz,
        "time_utc": now.strftime("%H:%M")
    }

# =============================================================================
# PILLAR 3: SIMULADOR DE SOMBRAS (Backtest Automator)
# =============================================================================

@mcp.tool()
@pulse_observer
async def run_shadow_backtest(
    expert_name: str, 
    symbol: str, 
    period: str = "H1", 
    days: int = 30,
    deposit: int = 10000
):
    """
    Ejecuta una prueba de estrategia en segundo plano usando el Strategy Tester de MT5.
    """
    # Calcular fechas
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    return await tester.run_strategy_test(
        expert_name=expert_name,
        symbol=symbol,
        period=period,
        date_from=start_date.strftime("%Y.%m.%d"),
        date_to=end_date.strftime("%Y.%m.%d"),
        deposit=deposit
    )

@mcp.tool()
async def execute_twin_entry(symbol: str = "XAUUSD", direction: str = "BUY"):
    """
    Ejecuta la estrategia Hybrid Twin-Engine.
    Abre 2 operaciones simultneas: Scalp ($2 TP) + Swing ($10+ TP).
    Si no hay margen suficiente, solo abre el Scalp.
    """
    from app.skills.execution import execute_twin_trade
    
    signal = {
        "symbol": symbol,
        "direction": direction.upper(),
        "confidence": 0.90
    }
    
    result = await execute_twin_trade(signal)
    return result

@mcp.tool()
async def get_twin_engine_status():
    """
    Obtiene el estado actual del motor Twin-Engine.
    Incluye asignacin de capital Scalp vs Swing.
    """
    from app.services.risk_engine import risk_engine
    
    allocation = await risk_engine.get_capital_allocation()
    return {
        "engine_mode": "TWIN" if allocation["can_dual"] else "SCALP_ONLY",
        "scalp_capital": allocation["scalp_capital"],
        "swing_capital": allocation["swing_capital"],
        "max_positions": allocation["max_positions"],
        "equity": allocation["equity"]
    }

# =============================================================================
# PILLAR 4: INYECCIN ESTRATGICA (Unified Model)
# =============================================================================

@mcp.tool()
@pulse_observer
async def query_unified_brain(sql_query: str):
    """
    Consulta la base de datos de conocimiento unificado (Unified Model).
    Solo permite SELECT.
    """
    return await unified_model.unified_db.query_custom_sql(sql_query)

# =============================================================================
# EXECUTION & MANAGEMENT SKILLS
# =============================================================================

@mcp.tool()
async def execute_trade(symbol: str, action: str, volume: float, price: float = None, sl: float = None, tp: float = None, comment: str = "MCP_Order"):
    """Ejecuta rdenes: BUY, SELL, LIMIT, STOP."""
    req = TradeOrderRequest(symbol=symbol, action=action, volume=volume, price=price, sl=sl, tp=tp, comment=comment)
    result = await execution.send_order(req)
    return result.dict()

@mcp.tool()
async def manage_trade(ticket: int, action: str, volume: float = None, sl: float = None, tp: float = None):
    """Gestiona posiciones: CLOSE, MODIFY_SL_TP, DELETE."""
    req = PositionManageRequest(ticket=ticket, action=action, volume=volume, sl=sl, tp=tp)
    result = await trade_mgmt.manage_position(req)
    return result.dict()

@mcp.tool()
async def get_account_status():
    """Estado financiero de la cuenta."""
    return await market.get_account_metrics()

@mcp.tool()
async def get_economic_calendar():
    """Eventos econmicos de alto impacto prximos."""
    req = CalendarRequest(importance="HIGH", days_forward=1)
    return await calendar.get_economic_calendar(req)

@mcp.tool()
async def get_financial_performance():
    """
    Obtiene el rendimiento financiero detallado para el Dashboard.
    Incluye PnL Diario, Ganancia por Hora y Diario de Operaciones.
    """
    import datetime
    from app.services.risk_engine import risk_engine
    
    # 1. Mtricas de Cuenta
    account = await market.get_account_metrics()
    
    # 2. Rendimiento Hoy (UTC)
    now = datetime.datetime.now()
    start_of_day = datetime.datetime(now.year, now.month, now.day)
    
    deals = await mt5_conn.execute(mt5.history_deals_get, start_of_day, now + datetime.timedelta(hours=1))
    
    daily_pnl = 0.0
    hourly_pnl = 0.0
    journal = []
    
    one_hour_ago = now - datetime.timedelta(hours=1)
    
    if deals:
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT: # Solo trades cerrados
                profit = deal.profit + deal.swap + deal.commission
                daily_pnl += profit
                
                # Hora del deal
                deal_time = datetime.datetime.fromtimestamp(deal.time)
                if deal_time > one_hour_ago:
                    hourly_pnl += profit
                
                journal.append({
                    "id": deal.ticket,
                    "time": deal_time.strftime("%H:%M:%S"),
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                    "profit": round(profit, 2)
                })

    return {
        "status": "success",
        "equity": account.get("equity"),
        "balance": account.get("balance"),
        "daily_pnl": round(daily_pnl, 2),
        "hourly_pnl": round(hourly_pnl, 2),
        "drawdown_percent": round((1 - account.get("equity") / account.get("balance")) * 100, 2) if account.get("balance",0) > 0 else 0,
        "journal": journal[-10:] # ltimos 10 trades
    }

@mcp.tool()
async def get_profit_velocity():
    """
    Calcula la velocidad de ganancia ($/Hora) de la sesin actual.
    Frmula: (Beneficio Total Sesin) / (Horas Transcurridas desde el inicio del da).
    """
    import datetime
    perf = await get_financial_performance()
    
    now = datetime.datetime.now()
    start_of_day = datetime.datetime(now.year, now.month, now.day)
    hours_elapsed = (now - start_of_day).total_seconds() / 3600.0
    
    # Mnimo 1 hora para evitar divisiones agresivas al inicio del da
    hours_elapsed = max(1.0, hours_elapsed)
    
    velocity = perf.get("daily_pnl", 0) / hours_elapsed
    
    return {
        "profit_velocity_usd_hr": round(velocity, 2),
        "total_session_hours": round(hours_elapsed, 2),
        "bias": "ACCELERATING" if velocity > 0 else "DECELERATING"
    }

# =============================================================================
# PILLAR 5: MEMORIA INFINITA (RAG Memory)
# =============================================================================

@mcp.tool()
async def remember(text: str, category: str = "general"):
    """
    Almacena informacin en memoria permanente RAG.
    Las memorias persisten entre reinicios del contenedor.
    
    Categoras sugeridas: analysis, trade, news, pattern, lesson, config
    """
    from app.services.rag_memory import rag_memory
    doc_id = rag_memory.store(text, category=category)
    return {"status": "stored", "id": doc_id, "category": category}

@mcp.tool()
async def recall(query: str, limit: int = 5, category: str = None):
    """
    Busca informacin relevante en memoria RAG usando bsqueda semntica.
    Retorna los documentos ms similares a la query.
    """
    from app.services.rag_memory import rag_memory
    results = rag_memory.search(query, k=limit, category=category)
    return {
        "query": query,
        "count": len(results),
        "memories": results
    }

@mcp.tool()
async def forget(category: str):
    """
    Elimina todas las memorias de una categora especfica.
    til para limpiar informacin obsoleta.
    """
    from app.services.rag_memory import rag_memory
    count = rag_memory.forget(category=category)
    return {"status": "deleted", "count": count, "category": category}

@mcp.tool()
async def memory_stats():
    """
    Obtiene estadsticas de la memoria RAG.
    """
    from app.services.rag_memory import rag_memory
    return {
        "total_memories": rag_memory.count(),
        "categories": rag_memory.get_categories()
    }

# =============================================================================
# PILLAR 6: SYSTEM ADMIN (Auto-Diagnstico)
# =============================================================================

@mcp.tool()
@pulse_observer
async def system_check():
    """
    Revisa la salud del servidor (CPU, RAM, Disco).
    til para auto-diagnstico antes de operaciones pesadas.
    """
    from app.skills.system_ops import system_health_check
    return await system_health_check()

@mcp.tool()
@pulse_observer
async def system_environment():
    """
    Obtiene informacin del entorno de ejecucin.
    """
    from app.skills.system_ops import get_environment_info
    return await get_environment_info()

@mcp.tool()
@pulse_observer
async def system_cleanup():
    """
    Limpia caches de Python para liberar memoria.
    """
    from app.skills.system_ops import cleanup_cache
    return await cleanup_cache()

# =============================================================================
# PILLAR 7: QUANTUM LEAP ML (Stochastic Predictions)
# =============================================================================

@mcp.tool()
@pulse_observer
async def ml_predict(features: dict, symbol: str = "BTCUSD"):
    """
    Genera prediccin ML usando ensemble GBM+LSTM.
    Shadow Mode por defecto (no ejecuta, solo predice).
    
    Args:
        features: Dict con close, rsi, ema_fast, ema_slow, volume, feat_score, fsm_state, etc.
        symbol: Par de divisas (Asset Identity Protocol).
    """
    from app.ml.ml_engine import predict
    return await predict(features, symbol)

@mcp.tool()
@pulse_observer
async def ml_status():
    """
    Estado del sistema ML: modelos cargados, modo, etc.
    """
    from app.ml.ml_engine import get_ml_status
    return await get_ml_status()

@mcp.tool()
@pulse_observer
async def ml_collect_sample(symbol: str, candle: dict, indicators: dict):
    """
    Recolecta muestra para training dataset con Oracle labeling.
    
    Args:
        symbol: Par de divisas
        candle: Dict con open, high, low, close, volume
        indicators: Dict con RSI, EMAs, FEAT score, FSM state, etc.
    """
    from app.ml.data_collector import collect_sample
    return await collect_sample(symbol, candle, indicators)

@mcp.tool()
@pulse_observer
async def ml_train():
    """
    Entrena modelos GBM y LSTM si hay datos suficientes.
    Requiere 1000+ muestras etiquetadas.
    """
    from app.ml.train_models import train_all
    return train_all()

@mcp.tool()
@pulse_observer
async def ml_enable_execution(enable: bool = True):
    """
    (PELIGROSO) Activa/desactiva ejecucin real de rdenes.
    Solo usar despus de verificar predicciones en Shadow Mode.
    """
    from app.ml.ml_engine import enable_execution
    return await enable_execution(enable)

@mcp.tool()
@pulse_observer
async def get_system_health():
    """
     HEALTH PULSE (Institutional Status)
    Radiografa completa de la salud del ecosistema FEAT NEXUS.
    """
    from app.core.mt5_conn import mt5_conn
    from app.services.supabase_sync import supabase_sync
    from app.core.zmq_bridge import zmq_bridge
    
    health = {
        "status": "NOMINAL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brain": {
            "uptime": "Active",
            "mode": "Linux/Docker" if not sys.stdin.isatty() else "Windows/Local",
            "observer": "Neural Pulse ACTIVE"
        },
        "connectivity": {
            "mt5_bridge": "ONLINE" if mt5_conn.available else "OFFLINE",
            "zmq_market_data": "RUNNING" if zmq_bridge.running else "STOPPED",
            "persistence_cloud": "SYNCED" if supabase_sync.client else "LOCAL_ONLY"
        },
        "resource_pulse": {
            "p99_latency_ms": "< 10ms",
            "correlation_tracking": "ENABLED"
        }
    }
    
    # Check for systemic warnings
    if not mt5_conn.available or not supabase_sync.client:
        health["status"] = "DEGRADED"
        
    return health

# =============================================================================
# FEAT-DEEP MULTI-TEMPORAL INTELLIGENCE TOOLS
# =============================================================================

@mcp.tool()
@pulse_observer
async def get_market_state_tensor(symbol: str = "XAUUSD"):
    """
     FEAT-DEEP: Returns the Multi-Temporal Market State Tensor.
    Layers: Macro (H4), Structural (H1/M15), Execution (M5/M1).
    Includes alignment score and kill zone status.
    """
    from app.ml.data_collector import fetch_multi_tf_data
    
    data = await fetch_multi_tf_data(symbol)
    
    if data.get("tensor"):
        return {
            "symbol": symbol,
            "tensor": data["tensor"],
            "status": "ACTIVE"
        }
    
    return {
        "symbol": symbol,
        "error": "No multi-TF data available. Collect data first.",
        "status": "NO_DATA"
    }


@mcp.tool()
@pulse_observer
async def get_h4_veto_status(symbol: str = "XAUUSD", proposed_signal: str = "BUY"):
    """
     FEAT-DEEP VETO RULE: Checks if a M1/M5 signal is allowed.
    Returns ALLOW or VETO based on H4 trend alignment.
    
    Philosophy: M1 is noise, H4 is truth. No counter-trend trading.
    """
    from app.ml.data_collector import get_h4_bias
    from app.ml.ml_engine import ml_engine
    
    # Get current H4 bias
    bias_data = await get_h4_bias(symbol)
    h4_trend = bias_data.get("H4_Trend", "NEUTRAL")
    
    # Update ML Engine macro bias cache
    ml_engine.update_macro_bias(symbol, h4_trend)
    
    # Apply Veto Rule
    should_trade, reason = ml_engine.apply_feat_veto(symbol, proposed_signal)
    
    return {
        "symbol": symbol,
        "proposed_signal": proposed_signal,
        "h4_trend": h4_trend,
        "decision": "ALLOW" if should_trade else "VETO",
        "reason": reason,
        "alignment_score": bias_data.get("alignment_score", 0),
        "kill_zone": bias_data.get("kill_zone"),
        "in_ny_kz": bias_data.get("in_ny_kz", False)
    }


@mcp.tool()
@pulse_observer
async def get_kill_zone_status():
    """
     FEAT-DEEP: Returns current Kill Zone status.
    NY: 07:00-11:00 UTC-4 (Best for XAUUSD)
    """
    from app.skills.liquidity_detector import get_current_kill_zone, is_in_kill_zone
    
    return {
        "current_kill_zone": get_current_kill_zone(),
        "in_ny_session": is_in_kill_zone("NY"),
        "in_london_session": is_in_kill_zone("LONDON"),
        "in_asia_session": is_in_kill_zone("ASIA"),
        "recommendation": "OPTIMAL" if is_in_kill_zone("NY") else "REDUCED_RISK"
    }


@mcp.tool()
@pulse_observer
async def get_liquidity_pools(symbol: str = "XAUUSD"):
    """
     FEAT-DEEP: Detects institutional liquidity pools.
    Returns unmitigated Swing Highs/Lows that act as price magnets.
    """
    from app.ml.data_collector import fetch_multi_tf_data
    from app.skills.liquidity_detector import detect_liquidity_pools
    
    data = await fetch_multi_tf_data(symbol)
    
    if not data.get("H1").empty:
        liq = detect_liquidity_pools(data["H1"])
        return {
            "symbol": symbol,
            "liquidity_above": liq.get("liquidity_above", 0),
            "liquidity_below": liq.get("liquidity_below", 0),
            "total_pools": liq.get("total_pools", 0),
            "pools": liq.get("pools", [])[:5]  # Top 5 pools
        }
    
    return {"symbol": symbol, "error": "No H1 data for liquidity detection"}


# =============================================================================
# FEAT MODULAR NEURAL TRAINING SYSTEM
# =============================================================================

@mcp.tool()
@pulse_observer
async def feat_check_tiempo(
    server_time_gmt: str = None,
    h4_candle: str = "NEUTRAL",
    news_in_minutes: int = 999
):
    """
     FEAT Module T: Tiempo (Kill Switch #1)
    Verifica Kill Zones, alineacin H4, y filtro de noticias.
    """
    from app.skills.feat_tiempo import analyze_tiempo
    return analyze_tiempo(server_time_gmt, h4_candle, news_in_minutes)


@mcp.tool()
@pulse_observer
async def feat_check_tiempo_advanced(
    server_time_utc: str = None,
    news_event_upcoming: bool = False,
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL"
):
    """
     FEAT Module T ADVANCED: Ciclo Diario de Liquidez XAU/USD
    
    Analiza fases de liquidez institucional por hora:
    - ASIA_OVERNIGHT, PRE_LONDON, LONDON_KILLZONE, NY_KILLZONE, etc.
    - Intensidad: LOW/MEDIUM/HIGH/VERY_HIGH
    - Modos: WAIT/PREPARE/EXECUTE/MANAGE
    """
    from app.skills.feat_tiempo import analyze_tiempo_advanced
    return analyze_tiempo_advanced(server_time_utc, news_event_upcoming, h4_direction, h1_direction)


@mcp.tool()
@pulse_observer
async def feat_generate_chrono_features(
    server_time_utc: str = None,
    news_upcoming: bool = False,
    current_spread_pips: float = None
):
    """
     FEAT CHRONO-ANALYST: Genera features numricas para ML.
    
    NO hay kill switches - solo probabilidades y multiplicadores de riesgo.
    Incluye: ciclo semanal, liquidez, volatilidad, proximidad H4.
    """
    from app.skills.feat_tiempo import generate_chrono_features
    return generate_chrono_features(server_time_utc, news_upcoming, current_spread_pips)


@mcp.tool()
@pulse_observer
async def feat_analyze_tiempo_institucional(
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    has_sweep: bool = False,
    news_upcoming: bool = False
):
    """
     FEAT TIEMPO INSTITUCIONAL: Anlisis completo para GC/XAU.
    
    Integra:
    - Sesiones: Globex, Asia/SGE, London, NY Overlap
    - LBMA/SGE Fixes (AM/PM benchmarks)
    - Alignment D1+H4+H1 con size multiplier
    - Entry Templates: Confirmation/Post-Fix/Sweep
    
    Retorna checklist institucional completo.
    """
    from app.skills.feat_tiempo import analyze_tiempo_institucional
    return analyze_tiempo_institucional(server_time_utc, d1_direction, h4_direction, h1_direction, has_sweep, news_upcoming)


# =============================================================================
# KILLZONE MICRO-TEMPORAL INTELLIGENCE
# =============================================================================

@mcp.tool()
@pulse_observer
async def feat_get_current_killzone_block(server_time_utc: str = None):
    """
     KILLZONE BLOCK: Obtiene el bloque actual de 15 minutos (09:00-13:00 Bolivia).
    
    16 bloques con:
    - session_heat (0.0-1.0)
    - expansion_probability
    - liquidity_state (FSM)
    - action_recommendation
    
    PEAK BLOCK: 09:30-09:44 
    """
    from app.skills.killzone_intelligence import feat_get_current_killzone_block as get_block
    return await get_block(server_time_utc)


@mcp.tool()
@pulse_observer
async def feat_generate_temporal_ml_features(
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    current_volume_ratio: float = 1.0
):
    """
     TEMPORAL ML FEATURES: Genera vector de features para red neuronal.
    
    Incluye:
    - session_heat, expansion_prob (priors)
    - alignment_factor (-1.0 to +1.0)
    - posterior_probability (Bayesian)
    - liquidity_state one-hot
    """
    from app.skills.killzone_intelligence import generate_temporal_ml_features
    return generate_temporal_ml_features(
        server_time_utc, d1_direction, h4_direction, h1_direction, current_volume_ratio
    )


@mcp.tool()
@pulse_observer
async def feat_check_h1_confirmation(
    h1_closed_outside: bool,
    minutes_since_close: int,
    retest_occurred: bool = False,
    retest_rejected: bool = False,
    current_volume_ratio: float = 1.0,
    in_killzone: bool = False
):
    """
     H1 CONFIRMATION: Verifica nivel de confirmacin H1.
    
    Reglas:
    - Close fuera del nivel
    - Retest 5-30 min con rechazo
    - Volume  1.3x
    - Killzone boost
    
    Retorna: level, score, proceed
    """
    from app.skills.killzone_intelligence import check_h1_confirmation
    return check_h1_confirmation(
        h1_closed_outside, minutes_since_close, retest_occurred,
        retest_rejected, current_volume_ratio, in_killzone
    )


@mcp.tool()
@pulse_observer
async def feat_update_bayesian_prior(block: str, was_successful: bool):
    """
     UPDATE PRIOR: Actualiza prior bayesiano basado en resultado.
    
    Llamar despus de cada trade para aprendizaje adaptativo.
    Si accuracy < 40% por 5 das  reduce prior automticamente.
    """
    from app.ml.temporal_features import feat_update_bayesian_prior as update_prior
    return await update_prior(block, was_successful)


@mcp.tool()
@pulse_observer
async def feat_get_bayesian_priors():
    """
     GET PRIORS: Obtiene todos los priors bayesianos actuales.
    
    Muestra expansion_prob y confidence por cada bloque de 15-min.
    """
    from app.ml.temporal_features import feat_get_all_priors as get_priors
    return await get_priors()






@mcp.tool()
@pulse_observer
async def feat_analyze_forma(
    h4_candles: list = None,
    h1_candles: list = None,
    m15_candles: list = None
):
    """
     FEAT Module F: Forma (Market Structure)
    Detecta tendencia, BOS/CHoCH, y fases Wyckoff.
    """
    from app.skills.feat_forma import analyze_forma
    return analyze_forma(h4_candles, h1_candles, m15_candles)


@mcp.tool()
@pulse_observer
async def feat_analyze_forma_advanced(
    h4_candles: list = None,
    h1_candles: list = None,
    m15_candles: list = None,
    chrono_features: dict = None
):
    """
     FEAT Module F ADVANCED: Estructura Chrono-Aware
    
    Integra contexto temporal para validar BOS/CHoCH:
    - Detecta trampas de Lunes (INDUCTION)
    - SWEEP detection (mechas sin cierre)
    - Spring/Upthrust Wyckoff
    """
    from app.skills.feat_forma import analyze_forma
    return analyze_forma(h4_candles, h1_candles, m15_candles, None, chrono_features)


@mcp.tool()
@pulse_observer
async def feat_generate_structure_features(
    h4_candles: list = None,
    h1_candles: list = None,
    m15_candles: list = None,
    chrono_features: dict = None
):
    """
     FEAT STRUCTURE ML: Genera features numricas de estructura.
    
    Output listo para red neuronal: alignment_score, trend flags, event flags.
    """
    from app.skills.feat_forma import generate_structure_features
    return generate_structure_features(h4_candles, h1_candles, m15_candles, chrono_features)



@mcp.tool()
@pulse_observer
async def feat_map_espacio(
    candles: list,
    current_price: float,
    market_structure: str = "NEUTRAL"
):
    """
     FEAT Module E: Espacio (Liquidity Zones)
    Detecta FVG, Order Blocks, y Premium/Discount zones.
    """
    from app.skills.feat_espacio import analyze_espacio
    return analyze_espacio(candles, current_price, market_structure)


@mcp.tool()
@pulse_observer
async def feat_map_espacio_advanced(
    candles: list,
    current_price: float,
    market_structure: str = "NEUTRAL",
    chrono_features: dict = None
):
    """
     FEAT Module E ADVANCED: Zonas Chrono-Aware
    
    Integra contexto temporal para validar calidad de zonas:
    - Zonas de Kill Zone tienen +20% score
    - Zonas de Lunes (INDUCTION) tienen -40% score
    """
    from app.skills.feat_espacio import analyze_espacio
    return analyze_espacio(candles, current_price, market_structure, "H1", chrono_features)

@mcp.tool()
@pulse_observer
async def feat_generate_liquidity_features(
    candles: list,
    current_price: float,
    chrono_features: dict = None
):
    """
    FEAT LIQUIDITY ML: Genera features de liquidez.
    """
    from app.skills.feat_espacio import generate_liquidity_features
    return generate_liquidity_features(candles, current_price, chrono_features)


# =============================================================================
# MARKET PHYSICS & REGIME ANALYSIS
# =============================================================================

@mcp.tool()
@pulse_observer
async def feat_analyze_physics(
    m15_candles: list,
    m5_candles: list
):
    """
    FEAT PHYSICS: Analiza Masas y Movimientos.
    
    Retorna:
    - PVP Vectorial (POC, VAH, VAL, Skew)
    - MCI (Manipulation Index)
    - Liquidity Primitives (Pools, Sweeps)
    """
    from app.skills.market_physics import market_physics
    import pandas as pd
    
    df_m15 = pd.DataFrame(m15_candles) if m15_candles else pd.DataFrame()
    df_m5 = pd.DataFrame(m5_candles) if m5_candles else pd.DataFrame()
    
    return {
        "pvp": market_physics.calculate_pvp_feat(df_m15),
        "cvd": market_physics.calculate_cvd_metrics(df_m5),
        "mci": market_physics.calculate_mci(df_m5)
    }


@mcp.tool()
@pulse_observer
async def feat_get_energy_map(
    m15_candles: list,
    bins: int = 50
):
    """
    FEAT ENERGY MAP: Genera el tensor de energía institucional (Density * Kinetic * Flow).
    """
    import pandas as pd
    from nexus_core.features import feat_features
    df = pd.DataFrame(m15_candles)
    return feat_features.generate_energy_map(df, bins)


@mcp.tool()
@pulse_observer
async def feat_get_risk_parameters(
    current_features: dict,
    model_name: str = "gbm_XAUUSD_v1"
):
    """
    FEAT RISK ENGINE: Retorna TP/SL dinámicos y probabilidad para cBot (C#).
    """
    from nexus_brain.model import nexus_brain
    import os
    
    model_path = os.path.join("models", f"{model_name}.joblib")
    if not nexus_brain.model and os.path.exists(model_path):
        nexus_brain.load_model(model_path)
        
    return nexus_brain.predict_probability(current_features)


@mcp.tool()
@pulse_observer
async def feat_get_structure_analysis(
    candles: list,
    timeframe: str = "M15"
):
    """
    FEAT STRUCTURE ENGINE: Analiza la estructura institucional (FourJarvis).
    Calcula scores de Forma (F), Espacio (E), Aceleración (A) y Tiempo (T).
    """
    import pandas as pd
    from nexus_core.structure_engine import structure_engine
    df = pd.DataFrame(candles)
    # Ensure tick_time exists for Phase T
    if 'tick_time' not in df.columns and 'timestamp' in df.columns:
        df['tick_time'] = df['timestamp']
    return structure_engine.compute_feat_index(df).iloc[-1].to_dict()


@mcp.tool()
@pulse_observer
async def feat_analyze_acceleration(
    candles: list,
    atr_window: int = 14
):
    """
    FEAT ACCELERATION: Analiza el momento e intención final (Gatillo).
    Retorna score de aceleración, tipo (breakout/rejection/climax) y trigger.
    """
    import pandas as pd
    from nexus_core.acceleration import acceleration_engine
    df = pd.DataFrame(candles)
    return acceleration_engine.compute_acceleration_features(df).iloc[-1].to_dict()


@mcp.tool()
@pulse_observer
async def feat_detect_regime(
    physics_metrics: dict,
    temporal_features: dict,
    trend_context: str = "NEUTRAL"
):
    """
    FEAT REGIME FSM: Clasifica Estado del Mercado (0-1).
    
    Estados:
    - ACCUMULATION (Low Entropy)
    - MANIPULATION (Trap)
    - EXPANSION_REAL (Trend)
    - EXPANSION_FAKE (Failed Breakout)
    """
    from app.skills.feat_regime import regime_fsm
    structure_context = {"trend": trend_context, "has_sweep": False} # Basic proxy
    return regime_fsm.detect_regime(physics_metrics, temporal_features, structure_context)


@mcp.tool()
@pulse_observer
async def feat_validate_aceleracion(
    recent_candles: list,
    poi_status: str = "NEUTRAL",
    proposed_direction: str = None
):
    """
    FEAT Module A: Aceleracion (Momentum Validation)
    Valida Body/Wick, volumen, y detecta fakeouts.
    """
    from app.skills.feat_aceleracion import analyze_aceleracion
    return analyze_aceleracion(recent_candles, poi_status, proposed_direction)


@mcp.tool()
@pulse_observer
async def feat_validate_aceleracion_advanced(
    recent_candles: list,
    proposed_direction: str = None,
    chrono_features: dict = None
):
    """
    FEAT Module A ADVANCED: Momentum Chrono-Aware
    
    Execution probability ajustada por ciclo temporal:
    - Kill Zone: +15% momentum score
    - INDUCTION (Lunes): -30% momentum score
    - Fakeout y Exhaustion detection
    """
    from app.skills.feat_aceleracion import analyze_aceleracion
    return analyze_aceleracion(recent_candles, "NEUTRAL", proposed_direction, 14, chrono_features)


@mcp.tool()
@pulse_observer
async def feat_generate_momentum_features(
    recent_candles: list,
    proposed_direction: str = None,
    chrono_features: dict = None
):
    """
    FEAT MOMENTUM ML: Genera features de momentum para red neuronal.
    """
    from app.skills.feat_aceleracion import generate_momentum_features
    return generate_momentum_features(recent_candles, proposed_direction, chrono_features)



@mcp.tool()
@pulse_observer
async def feat_full_chain(
    symbol: str = "XAUUSD",
    h4_candles: list = None,
    h1_candles: list = None,
    m15_candles: list = None,
    m5_candles: list = None,
    current_price: float = None
):
    """
    FEAT COMPLETE CHAIN: Ejecuta T -> F -> E -> A en secuencia.
    Si cualquier modulo falla, la cadena se detiene (Kill Switch).
    Retorna senal de trading si todo pasa.
    """
    from app.skills.feat_chain import execute_feat_chain_institucional
    return await execute_feat_chain_institucional(
        symbol=symbol,
        h4_candles=h4_candles,
        h1_candles=h1_candles,
        m15_candles=m15_candles,
        m5_candles=m5_candles,
        current_price=current_price
    )


@mcp.tool()
@pulse_observer
async def feat_full_chain_institucional(
    symbol: str = "XAUUSD",
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    h4_candles: list = None,
    h1_candles: list = None,
    m15_candles: list = None,
    m5_candles: list = None,
    current_price: float = None,
    has_sweep: bool = False,
    news_upcoming: bool = False
):
    """
    FEAT CHAIN INSTITUCIONAL: Pipeline completo MIP.
    
    6 Stages:
    1. TIEMPO -> Session, fixes, D1/H4/H1 alignment
    2. FORMA -> BOS/CHoCH, sweeps, structure  
    3. ESPACIO -> FVG, OB, Premium/Discount
    4. ACELERACION -> Momentum, fakeout, volume
    5. FUSION -> MTF weighted probability
    6. LIQUIDITY -> DoM preflight
    
    Retorna probabilidad 0.0-1.0 y trade_params si >0.55
    """
    from app.skills.feat_chain import execute_feat_chain_institucional
    return await execute_feat_chain_institucional(
        symbol, server_time_utc, d1_direction, h4_direction, h1_direction,
        h4_candles, h1_candles, m15_candles, m5_candles, current_price,
        has_sweep, news_upcoming
    )

@mcp.tool()
async def feat_deep_audit(auto_repair: bool = False) -> Dict[str, Any]:
    """
    MASTER AUDIT: Deep inspection of FEAT NEXUS ecosystem.
    Checks: Connectivity, Intelligence (ML), Sensors (MSS-5), and Data Flow.
    """
    from app.skills.skill_deep_auditor import run_deep_audit
    return run_deep_audit(auto_repair)

@mcp.tool()
async def feat_analyze_acceptance(
    open_p: float, high: float, low: float, close_p: float, zone_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    KIV PROTOCOL: Analyze Acceptance Ratio (Body vs Wick).
    If zone_price is provided, validates Conquest vs Probe.
    """
    from app.skills.feat_kiv import kiv_engine
    candle = {"open": open_p, "high": high, "low": low, "close": close_p}
    zones = {"resistance": zone_price} if zone_price else {}
    return kiv_engine.validate_intent(candle, zones)

if __name__ == "__main__":
    import os
    import sys
    
    # Detectar si estamos en Docker/Linux (sin TTY) o en Windows (con TTY)
    is_docker = not sys.stdin.isatty() or os.environ.get("DOCKER_MODE", "").lower() == "true"
    
    if is_docker:
        logger.info("Modo Docker detectado - Iniciando servidor SSE en puerto 8000...")
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        logger.info("Modo Windows detectado - Iniciando servidor STDIO...")
        mcp.run()
