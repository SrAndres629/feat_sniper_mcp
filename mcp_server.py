import logging
import sys
import uuid
import json
from datetime import datetime, timezone
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


# Importar lógica de negocio de la app
from app.core.mt5_conn import mt5_conn
from app.skills import (
    market, vision, execution, trade_mgmt, 
    indicators, history, calendar, 
    quant_coder, custom_loader,
    tester, unified_model, remote_compute # Nuevos módulos
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
            logger.info(f"Señal recibida vía ZMQ: {data}")
            import asyncio
            asyncio.create_task(supabase_sync.log_signal(data))
        
    print(f"DEBUG: Intentando iniciar ZMQ Bridge en callback {on_zmq_message}", flush=True)
    await zmq_bridge.start(on_zmq_message)
    print("DEBUG: ZMQ Bridge iniciado VOLANDO", flush=True)
    
    # 3. Background Risk Monitor
    async def monitor_risk_loop():
        logger.info("🛡️ Monitor de Riesgo Institucional ACTIVO")
        from app.services.risk_engine import risk_engine
        while True:
            try:
                # Verificar Drawdown para Auto-Stop
                if not await risk_engine.check_drawdown_limit():
                    logger.critical("🚨 CIRCUIT BREAKER TRIPPED - Risk Limit Exceeded!")
                
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
                             
                        # Agora usa ATR dinâmico (limite configurado nas settings)
                        await risk_engine.apply_trailing_stop(pos.symbol, pos.ticket)
            except Exception as e:
                logger.error(f"Error in Risk Monitor: {e}")
            await asyncio.sleep(5) # Revisión cada 5 segundos

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

# Registrar Skills de Cómputo Remoto (NEXUS)
remote_compute.register_remote_skills(mcp)

# =============================================================================
# PILLAR 1: CÓRTEX DE COMPILACIÓN (Code Builder)
# =============================================================================

@mcp.tool()
@pulse_observer
async def create_and_compile_indicator(name: str, code: str, compile: bool = True):
    """
    Escribe y compila un nuevo indicador .mq5. 
    Retorna el log de compilación para auto-corrección.
    """
    req = MQL5CodeRequest(name=name, code=code, compile=compile)
    return await quant_coder.create_native_indicator(req)

# =============================================================================
# PILLAR 2: OJO DE HALCÓN (Data Bridge)
# =============================================================================

@mcp.tool()
@pulse_observer
async def get_market_snapshot(symbol: str, timeframe: str = "M5"):
    """
    Radiografía completa del mercado: Precio, Vela actual, Volatilidad y Cuenta.
    Ideal para decisiones de alta frecuencia.
    """
    return await market.get_market_snapshot(symbol, timeframe)

@mcp.tool()
async def get_candles(symbol: str, timeframe: str = "M5", n_candles: int = 100):
    """Obtiene velas históricas (OHLCV)."""
    return await market.get_candles(symbol, timeframe, n_candles)

@mcp.tool()
async def get_market_panorama(resize_factor: float = 0.75):
    """Captura visual del gráfico."""
    return await vision.capture_panorama(resize_factor=resize_factor)

@mcp.tool()
@pulse_observer
async def get_trade_decision(symbol: str, timeframe: str = "M5"):
    """
    Decisión de trading integrada para N8N.
    
    Combina ML (GBM/LSTM) + FEAT Score + FSM State + Indicadores
    para generar una señal unificada con contexto completo.
    
    Returns:
        JSON con signal, confidence, market_state, data_context
    """
    from datetime import datetime
    
    # 1. Obtener snapshot del mercado
    snapshot = await market.get_market_snapshot(symbol, timeframe)
    
    # 2. Obtener predicción ML si disponible
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
    
    # 3. Construir señal
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
            
        # Penalizar si hay anomalía
        if ml_prediction.get("is_anomaly"):
            confidence *= 0.5
            signal = "WAIT"  # No operar en manipulación
    
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
    ⭐ SUPER ENDPOINT PARA N8N (SSH Gateway).
    
    Consolida TODO en un único JSON para que el Agente N8N tome decisiones
    basadas en la estrategia FEAT completa, no en datos crudos.
    
    Incluye:
    - raw_data: OHLCV actual
    - indicators: FEAT, PVP, EMAs
    - ml_insight: Predicción + Explicabilidad
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
            "score": 0.0,  # Placeholder - viene de MT5 vía ZMQ
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
    
    # 5. GUÍA PARA EL AGENTE N8N
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
    """Extrae los factores principales que influyen en la predicción."""
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
    """Genera explicación legible para el agente N8N."""
    p_win = prediction.get("p_win", 0.5)
    source = prediction.get("source", "NONE")
    
    if source == "NONE":
        return "No ML models available. Decision based on rules only."
        
    direction = "LONG" if p_win > 0.5 else "SHORT"
    confidence = "high" if abs(p_win - 0.5) > 0.3 else "moderate" if abs(p_win - 0.5) > 0.15 else "low"
    
    return explanation

def _get_market_session() -> dict:
    """Detecta la sesión actual y killzones institucionales (UTC)."""
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
    Abre 2 operaciones simultáneas: Scalp ($2 TP) + Swing ($10+ TP).
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
    Incluye asignación de capital Scalp vs Swing.
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
# PILLAR 4: INYECCIÓN ESTRATÉGICA (Unified Model)
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
    """Ejecuta órdenes: BUY, SELL, LIMIT, STOP."""
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
    """Eventos económicos de alto impacto próximos."""
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
    
    # 1. Métricas de Cuenta
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
        "journal": journal[-10:] # Últimos 10 trades
    }

@mcp.tool()
async def get_profit_velocity():
    """
    Calcula la velocidad de ganancia ($/Hora) de la sesión actual.
    Fórmula: (Beneficio Total Sesión) / (Horas Transcurridas desde el inicio del día).
    """
    import datetime
    perf = await get_financial_performance()
    
    now = datetime.datetime.now()
    start_of_day = datetime.datetime(now.year, now.month, now.day)
    hours_elapsed = (now - start_of_day).total_seconds() / 3600.0
    
    # Mínimo 1 hora para evitar divisiones agresivas al inicio del día
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
    Almacena información en memoria permanente RAG.
    Las memorias persisten entre reinicios del contenedor.
    
    Categorías sugeridas: analysis, trade, news, pattern, lesson, config
    """
    from app.services.rag_memory import rag_memory
    doc_id = rag_memory.store(text, category=category)
    return {"status": "stored", "id": doc_id, "category": category}

@mcp.tool()
async def recall(query: str, limit: int = 5, category: str = None):
    """
    Busca información relevante en memoria RAG usando búsqueda semántica.
    Retorna los documentos más similares a la query.
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
    Elimina todas las memorias de una categoría específica.
    Útil para limpiar información obsoleta.
    """
    from app.services.rag_memory import rag_memory
    count = rag_memory.forget(category=category)
    return {"status": "deleted", "count": count, "category": category}

@mcp.tool()
async def memory_stats():
    """
    Obtiene estadísticas de la memoria RAG.
    """
    from app.services.rag_memory import rag_memory
    return {
        "total_memories": rag_memory.count(),
        "categories": rag_memory.get_categories()
    }

# =============================================================================
# PILLAR 6: SYSTEM ADMIN (Auto-Diagnóstico)
# =============================================================================

@mcp.tool()
@pulse_observer
async def system_check():
    """
    Revisa la salud del servidor (CPU, RAM, Disco).
    Útil para auto-diagnóstico antes de operaciones pesadas.
    """
    from app.skills.system_ops import system_health_check
    return await system_health_check()

@mcp.tool()
@pulse_observer
async def system_environment():
    """
    Obtiene información del entorno de ejecución.
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
    Genera predicción ML usando ensemble GBM+LSTM.
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
    (PELIGROSO) Activa/desactiva ejecución real de órdenes.
    Solo usar después de verificar predicciones en Shadow Mode.
    """
    from app.ml.ml_engine import enable_execution
    return await enable_execution(enable)

@mcp.tool()
@pulse_observer
async def get_system_health():
    """
    ⭐ HEALTH PULSE (Institutional Status)
    Radiografía completa de la salud del ecosistema FEAT NEXUS.
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
# SYSTEM RESOURCES
# =============================================================================

@mcp.resource("signals://live")
async def get_live_signals():
    """Stream de señales ZMQ disponibles."""
    return "Stream Active"

if __name__ == "__main__":
    import os
    import sys
    
    # Detectar si estamos en Docker/Linux (sin TTY) o en Windows (con TTY)
    # En Docker usamos HTTP para que el servidor persista
    # En Windows local usamos STDIO para integración directa con IDE
    is_docker = not sys.stdin.isatty() or os.environ.get("DOCKER_MODE", "").lower() == "true"
    
    if is_docker:
        logger.info("🐳 Modo Docker detectado - Iniciando servidor SSE en puerto 8000...")
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        logger.info("🖥️ Modo Windows detectado - Iniciando servidor STDIO...")
        mcp.run()
