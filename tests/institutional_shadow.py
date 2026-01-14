import asyncio
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from nexus_brain.inference_api import neural_api
from app.core.config import settings

# Patch settings for Shadow Mode
settings.LAYER_BIAS_PERIOD = 200 # Warmup razonable para el test
settings.SHADOW_MODE = True
settings.EXECUTION_ENABLED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [INSTITUTIONAL_SHADOW] - %(levelname)s - %(message)s"
)
logger = logging.getLogger("institutional_shadow")

async def run_institutional_shadow_test(duration_hours: float = 1.0):
    logger.info(f"🚀 Launching Institutional Shadow Mode (FEAT Monitoring)")
    logger.info(f"Duration: {duration_hours} hour(s) | Pillars: Shape, Space, Acceleration, Time")
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    
    stats = {
        "total_signals": 0,
        "vetos_physics": 0,
        "vetos_time": 0,
        "vetos_space": 0,
        "inference_latency_avg": 0.0,
        "drift_alerts": 0
    }
    
    tick_count = 0
    
    try:
        while time.time() < end_time:
            tick_count += 1
            
            # Simular tick dinámico con volatilidad variable
            # Simulamos un mercado que tiende a lateralizar pero con picos de aceleración
            vol_mult = 1.0 if (tick_count % 100 < 80) else 5.0 # Picos de volatilidad cada 100 ticks
            base_price = 2600.0 + (np.sin(tick_count/20.0) * 10)
            
            market_tick = {
                "bid": base_price + np.random.normal(0, 0.2 * vol_mult),
                "high": base_price + 0.5 * vol_mult,
                "low": base_price - 0.5 * vol_mult,
                "tick_volume": 100 + np.random.randint(0, 50) * vol_mult
            }
            
            # Inferencia
            result = await neural_api.predict_next_candle(market_tick)
            
            if result.get("status") == "Warming up":
                if tick_count % 20 == 0:
                    logger.info(f"System Warming Up... ({tick_count}/{settings.LAYER_BIAS_PERIOD} ticks)")
            else:
                stats["total_signals"] += 1
                latency = result.get("latency_ms", 0.0)
                stats["inference_latency_avg"] = (stats["inference_latency_avg"] * (stats["total_signals"] - 1) + latency) / stats["total_signals"]
                
                drift = result.get("drift_metrics", {}).get("drift_score", 0.0)
                if drift > 0.8: stats["drift_alerts"] += 1
                
                execute = result.get("execute_trade", False)
                veto = result.get("veto_reason", "None")
                
                if not execute and veto != "None":
                    if "Physics" in veto: stats["vetos_physics"] += 1
                    elif "Time" in veto: stats["vetos_time"] += 1
                    elif "Space" in veto: stats["vetos_space"] += 1
                
                if tick_count % 10 == 0 or execute:
                    status_icon = "🔵" if execute else "🚫" if veto != "None" else "➖"
                    logger.info(
                        f"{status_icon} Tick {tick_count:04d} | P_Win: {result.get('p_win', 0):.4f} | "
                        f"Veto: {veto[:20]}... | Lat: {latency:.2f}ms"
                    )

            await asyncio.sleep(0.1) # Simulación acelerada (10 ticks / seg)
            
            if tick_count % 500 == 0:
                logger.info(f"--- Session Stats: Signals={stats['total_signals']} | Physics_Vetos={stats['vetos_physics']} | Avg_Lat={stats['inference_latency_avg']:.2f}ms ---")

        # Reporte Final
        logger.info("\n" + "="*50)
        logger.info("INSTITUTIONAL SHADOW MODE FINAL REPORT")
        logger.info("="*50)
        logger.info(f"Total Ticks: {tick_count}")
        logger.info(f"Total Signals Generated: {stats['total_signals']}")
        logger.info(f"Physics Vetos (Consenso): {stats['vetos_physics']}")
        logger.info(f"Drift Alerts: {stats['drift_alerts']}")
        logger.info(f"Average Inference Latency: {stats['inference_latency_avg']:.2f}ms")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Shadow Mode Error: {e}")

import numpy as np
if __name__ == "__main__":
    asyncio.run(run_institutional_shadow_test(duration_hours=0.05)) # 3 minutos para validación rápida en este turno
