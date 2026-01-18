from .foundation import SubsystemCheck, SubsystemState
import logging

logger = logging.getLogger("Health.Checks")

async def check_ml_readiness() -> SubsystemCheck:
    try:
        from app.ml.ml_engine import ml_engine
        st = ml_engine.get_status()
        if not st.get("anomaly_fitted", False): return SubsystemCheck("ML Engine", SubsystemState.WARMING_UP, "Anomaly detector not fitted", metadata=st)
        return SubsystemCheck("ML Engine", SubsystemState.READY, f"V{st.get('v','?')} - {len(st.get('symbols_registered',[]))} symbols", metadata=st)
    except ImportError: return SubsystemCheck("ML Engine", SubsystemState.NOT_READY, "Module missing")
    except Exception as e: return SubsystemCheck("ML Engine", SubsystemState.ERROR, str(e))

async def check_physics_engine() -> SubsystemCheck:
    try:
        from app.skills.market_physics import MarketPhysics, market_physics
        if hasattr(MarketPhysics, 'MIN_DELTA_T'): return SubsystemCheck("Physics Engine", SubsystemState.READY, f"MIN_DELTA_T={MarketPhysics.MIN_DELTA_T}s", metadata={"window": market_physics.window_size})
        return SubsystemCheck("Physics Engine", SubsystemState.DEGRADED, "P0-1 fix missing")
    except ImportError: return SubsystemCheck("Physics Engine", SubsystemState.NOT_READY, "Module missing")
    except Exception as e: return SubsystemCheck("Physics Engine", SubsystemState.ERROR, str(e))

async def check_feat_gates() -> SubsystemCheck:
    try:
        from app.services.spread_filter import spread_filter
        from app.services.volatility_guard import volatility_guard
        if spread_filter and volatility_guard: return SubsystemCheck("FEAT Gates", SubsystemState.READY, "Active")
        return SubsystemCheck("FEAT Gates", SubsystemState.DEGRADED, "Partial")
    except ImportError as e: return SubsystemCheck("FEAT Gates", SubsystemState.NOT_READY, str(e))
