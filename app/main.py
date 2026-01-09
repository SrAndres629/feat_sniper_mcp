import logging
try:
    import MetaTrader5 as mt5
except ImportError:
    from unittest.mock import MagicMock
    mt5 = MagicMock()
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from app.core.config import settings
from app.core.mt5_conn import mt5_conn
from app.models.schemas import (
    ResponseModel, 
    MarketDataRequest, MarketDataResponse,
    AccountMetrics,
    PanoramaRequest, VisionResponse,
    TradeOrderRequest, TradeOrderResponse,
    VolatilityRequest, VolatilityMetrics,
    PositionManageRequest, PositionActionResponse,
    IndicatorRequest, HistoryRequest, CalendarRequest, MQL5CodeRequest,
    ErrorDetail
)
from app.skills import market, vision, execution, trade_mgmt, indicators, history, calendar, quant_coder

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MT5_Bridge.Gateway")

# =============================================================================
# LIFESPAN MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la aplicación:
    - Conecta con MT5 al iniciar.
    - Desconecta al cerrar.
    """
    logger.info("Iniciando MT5 Neural Bridge Gateway...")
    success = await mt5_conn.startup()
    if not success:
        logger.error("No se pudo establecer conexión inicial con MT5. El servidor operará en modo degradado.")
    
    yield
    
    await mt5_conn.shutdown()

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="MT5 Neural Bridge API",
    description="Gateway de alta disponibilidad para trading algorítmico e IA.",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware para CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MIDDLEWARE & TRACEABILITY
# =============================================================================

@app.middleware("http")
async def add_correlation_id_header(request: Request, call_next):
    correlation_id = request.headers.get(settings.CORRELATION_ID_HEADER) or "REQ_" + datetime.utcnow().strftime("%y%m%d%H%M%S")
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers[settings.CORRELATION_ID_HEADER] = correlation_id
    return response

# =============================================================================
# EXCEPTION HANDLER (GLOBAL ENVELOPE)
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Captura cualquier error no controlado y lo devuelve en el formato Envelope.
    Garantiza que n8n nunca reciba un error plano.
    """
    c_id = getattr(request.state, "correlation_id", "UNKNOWN")
    logger.exception(f"[{c_id}] Error no controlado en {request.url.path}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "correlation_id": c_id,
            "data": None,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": f"{type(exc).__name__}: {str(exc)}",
                "suggestion": "Revisa los logs del servidor para más detalles."
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=ResponseModel[dict])
async def health_check():
    """Verifica el estado de la conexión con MetaTrader 5."""
    connected = await mt5_conn.execute(mt5.terminal_info) is not None
    
    return ResponseModel(
        status="success" if connected else "error",
        data={"mt5_connected": connected},
        error=None if connected else ErrorDetail(
            code="MT5_DISCONNECTED",
            message="El terminal MT5 no está respondiendo.",
            suggestion="Asegúrate de que MT5 esté abierto y con sesión iniciada."
        )
    )

@app.post("/market/candles", response_model=ResponseModel[MarketDataResponse])
async def get_candles(request: MarketDataRequest):
    """Obtiene datos OHLC con sistema de caché."""
    result = await market.get_candles(
        symbol=request.symbol,
        timeframe=request.timeframe,
        n_candles=request.n_candles,
        output_format=request.output_format
    )
    
    if result["status"] == "error":
        return ResponseModel(
            status="error",
            error=ErrorDetail(
                code=result.get("mt5_error", ["DATA_ERROR"])[1],
                message=result["message"],
                suggestion="Verifica el símbolo y el timeframe."
            )
        )
    
    return ResponseModel(status="success", data=MarketDataResponse(**result))

@app.get("/market/account", response_model=ResponseModel[AccountMetrics])
async def get_account_metrics():
    """Obtiene métricas clave de la cuenta para gestión de riesgo."""
    result = await market.get_account_metrics()
    
    if result["status"] == "error":
        return ResponseModel(
            status="error",
            error=ErrorDetail(
                code="ACCOUNT_ERROR",
                message=result["message"],
                suggestion="Verifica la sesión en MT5."
            )
        )
        
    return ResponseModel(status="success", data=AccountMetrics(**result))

@app.post("/vision/panorama", response_model=ResponseModel[VisionResponse])
async def capture_panorama(request: PanoramaRequest):
    """Captura la pantalla actual de MT5 para análisis visual."""
    result = await vision.capture_panorama(resize_factor=request.resize_factor)
    
    if result["status"] == "error":
        return ResponseModel(
            status="error",
            error=ErrorDetail(
                code=result["error_code"],
                message=result["message"],
                suggestion=result["suggestion"]
            )
        )
        
    return ResponseModel(status="success", data=VisionResponse(**result))

@app.post("/market/create_indicator", response_model=ResponseModel[dict])
async def create_indicator(request: MQL5CodeRequest):
    """Crea, guarda y compila un nuevo indicador nativo (.mq5) en MetaTrader 5."""
    result = await quant_coder.create_native_indicator(request)
    if result["status"] == "error":
        return ResponseModel(status="error", error=ErrorDetail(code="CODER_ERROR", message=result["message"], suggestion="Revisa los permisos de escritura."))
    return ResponseModel(status="success", data=result)

@app.post("/market/indicators", response_model=ResponseModel[dict])
async def get_indicators(request: IndicatorRequest):
    """Calcula indicadores técnicos (RSI, Moving Average, MACD, etc)."""
    result = await indicators.get_technical_indicator(request)
    if result["status"] == "error":
        return ResponseModel(
            status="error",
            error=ErrorDetail(code="INDICATOR_ERROR", message=result["message"], suggestion="Verifica los parámetros del indicador.")
        )
    return ResponseModel(status="success", data=result)

@app.post("/market/calendar", response_model=ResponseModel[dict])
async def get_calendar(request: CalendarRequest):
    """Obtiene el calendario económico de MT5 para análisis fundamental."""
    result = await calendar.get_economic_calendar(request)
    return ResponseModel(status="success", data=result)

@app.post("/account/history", response_model=ResponseModel[dict])
async def get_history(request: HistoryRequest):
    """Obtiene el historial de trading y métricas de rendimiento (Win Rate, Profit Factor)."""
    result = await history.get_trade_history(request)
    return ResponseModel(status="success", data=result)

@app.post("/trade/order", response_model=ResponseModel[TradeOrderResponse])
async def place_order(request: TradeOrderRequest):
    """Envía una orden al mercado (Market o Pending) con validaciones pre-flight."""
    return await execution.send_order(request)

@app.post("/market/volatility", response_model=ResponseModel[VolatilityMetrics])
async def get_volatility(request: VolatilityRequest):
    """Calcula métricas de ATR y Spread."""
    result = await market.get_volatility_metrics(
        symbol=request.symbol,
        timeframe=request.timeframe,
        period=request.period
    )
    
    if result["status"] == "error":
        return ResponseModel(
            status="error",
            error=ErrorDetail(
                code="VOLATILITY_ERROR",
                message=result["message"],
                suggestion="Verifica el símbolo y que haya suficiente historial."
            )
        )
        
    return ResponseModel(status="success", data=VolatilityMetrics(**result))

@app.post("/trade/manage", response_model=ResponseModel[PositionActionResponse])
async def manage_position(request: PositionManageRequest):
    """Gestiona una posición abierta (Cierre o Modificación)."""
    return await trade_mgmt.manage_position(request)

# =============================================================================
# INSTITUTIONAL ANALYTICS & RISK
# =============================================================================

@app.post("/analytics/performance", response_model=ResponseModel[dict])
async def get_performance(request: HistoryRequest):
    """Auditoría cuantitativa de rendimiento (Sharpe, Win Rate, etc)."""
    from app.services.analytics import analytics_engine
    
    # 1. Obtener historial raw
    history_data = await history.get_trade_history(request)
    if history_data["status"] == "error":
        return ResponseModel(status="error", error=ErrorDetail(code="HISTORY_ERROR", message=history_data["message"], suggestion="Revisa el terminal MT5."))
    
    # 2. Procesar métricas institucionales
    metrics = await analytics_engine.get_performance_metrics(history_data.get("deals", []))
    return ResponseModel(status="success", data=metrics)

@app.get("/risk/metrics", response_model=ResponseModel[dict])
async def get_risk_status():
    """Estado actual de exposición y riesgo de la cuenta."""
    from app.services.risk_engine import risk_engine
    
    exposure = await risk_engine.get_total_exposure()
    drawdown_ok = await risk_engine.check_drawdown_limit()
    
    return ResponseModel(
        status="success",
        data={
            "total_exposure_percent": round(exposure, 2),
            "drawdown_limit_healthy": drawdown_ok,
            "max_drawdown_allowed": settings.MAX_DAILY_DRAWDOWN_PERCENT
        }
    )

# =============================================================================
# SNIPER 2.0 ADVANCED SKILLS
# =============================================================================

@app.post("/sniper/setup_score", response_model=ResponseModel[dict])
async def get_ml_score(request: IndicatorRequest):
    """Clasificación de setups con ML-Lite y detección de anomalías."""
    from app.skills.ml_sniper import ml_sniper
    result = await ml_sniper.score_setup(request.symbol, request.timeframe)
    return ResponseModel(status="success", data=result)

@app.get("/sniper/liquidity/{symbol}", response_model=ResponseModel[dict])
async def get_liquidity(symbol: str):
    """Análisis de profundidad de mercado (DoM) institucional."""
    from app.skills.liquidity import liquidity_engine
    result = await liquidity_engine.get_market_depth(symbol)
    return ResponseModel(status="success", data=result)

@app.get("/system/health", response_model=ResponseModel[dict])
async def get_system_health():
    """Monitoreo de Circuit Breaker y estado de latencia p99."""
    from app.services.circuit_breaker import circuit_breaker
    
    return ResponseModel(
        status="success",
        data=circuit_breaker.get_status()
    )
