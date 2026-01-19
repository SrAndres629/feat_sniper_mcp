import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("SMOKE_TEST")

# Importar Módulos Nucleares (La Nueva Arquitectura)
from app.ml.feat_processor import feat_processor
from app.ml.ml_engine import ml_engine
from nexus_core.mtf_engine import mtf_engine
from nexus_core.acceleration import acceleration_engine
from app.services.risk import risk_engine

async def run_smoke_test():
    logger.info("🔥 INICIANDO SMOKE TEST (Prueba de Flujo Completo)...")
    
    # 1. Generar Datos Sintéticos (Simulando MT5)
    # Creamos un DataFrame que parece una secuencia de ticks convertida a OHLC
    data = {
        'open': np.linspace(2000, 2010, 50),
        'high': np.linspace(2001, 2011, 50),
        'low': np.linspace(1999, 2009, 50),
        'close': np.linspace(2000.5, 2010.5, 50),
        'volume': np.random.randint(100, 500, 50),
        'tick_volume': np.random.randint(100, 500, 50),
        'time': pd.date_range(start="2024-01-01", periods=50, freq="1min")
    }
    df = pd.DataFrame(data)
    current_tick = {'close': 2010.5, 'volume': 300, 'time': datetime.now().timestamp()}
    
    try:
        # 2. Prueba de Motores Independientes
        logger.info("⚙️ Probando Nexus Acceleration (Physics)...")
        acc_feats = acceleration_engine.compute_acceleration_features(df)
        assert not acc_feats.empty, "Acceleration Engine devolvió DF vacío"
        logger.info("✅ Acceleration OK")

        logger.info("⚙️ Probando MTF Engine (Institutional)...")
        # Simulamos un diccionario de candles perezoso
        candles_map = {"M1": df, "M5": df, "H1": df, "H4": df} 
        mtf_score = await mtf_engine.analyze_all_timeframes(candles_map, 2010.5)
        logger.info(f"✅ MTF Score: {mtf_score.composite_score:.2f} (Action: {mtf_score.action})")

        logger.info("⚙️ Probando ML Engine (Neural)...")
        # Hidratación rápida
        ml_engine.hydrate("XAUUSD", df['close'].tolist(), [])
        prediction = await ml_engine.predict_async("XAUUSD", current_tick)
        logger.info(f"✅ ML Prediction: {prediction.get('prediction')} (Conf: {prediction.get('p_win', 0):.2f})")

        # 3. Prueba de Integración (Risk Gate)
        logger.info("🛡️ Probando Risk Engine (The Vault)...")
        # Simulamos que queremos abrir un trade basado en ML
        risk_check = await risk_engine.check_trading_veto("XAUUSD", "BUY", 2010.5)
        logger.info(f"✅ Risk Veto Status: {risk_check.get('status')}")

        logger.info("🌟 RESULTADO: EL SISTEMA ESTÁ VIVO Y FLUYENDO.")
        return True

    except Exception as e:
        logger.error(f"❌ FALLO EL SMOKE TEST: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
