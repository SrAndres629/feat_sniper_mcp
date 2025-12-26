import logging
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
# EXCEPTION HANDLER (GLOBAL ENVELOPE)
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Captura cualquier error no controlado y lo devuelve en el formato Envelope.
    Garantiza que n8n nunca reciba un error plano.
    """
    logger.exception(f"Error no controlado en {request.url.path}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
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
    import MetaTrader5 as mt5
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
