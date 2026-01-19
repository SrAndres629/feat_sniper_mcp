import sys
import os
import asyncio
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor import feat_processor
from app.ml.ml_engine import ml_engine
from app.services.risk import risk_engine
from nexus_core.acceleration import acceleration_engine

logging.basicConfig(level=logging.WARN, format='%(message)s')
logger = logging.getLogger("WAR_SIM")
logger.setLevel(logging.INFO)

class BattlefieldSimulator:
    """
    [FIRE TRIAL] Combat Simulation Protocol.
    Feeds historical data to the Full Stack (Physics -> MTF -> ML -> Risk)
    to verify intelligence emergence.
    """
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.balance = 10000.0
        self.positions = []
        self.history = []

    def generate_synthetic_data(self, n_rows=200):
        """Generates a volatile market scenario (Pump & Dump)"""
        logger.info(f"Generating {n_rows} bars of synthetic combat data...")
        
        # Split into 4 phases proportionally
        n = n_rows // 4
        remainder = n_rows - (n * 4)
        
        # Phase 1: Range
        p1 = np.random.normal(2000, 2, n)
        # Phase 2: Pump (Breakout)
        p2 = np.linspace(2000, 2050, n) + np.random.normal(0, 1, n)
        # Phase 3: Distribution
        p3 = np.random.normal(2050, 3, n)
        # Phase 4: Dump + Remainder
        p4 = np.linspace(2050, 1990, n + remainder) + np.random.normal(0, 5, n + remainder)
        
        prices = np.concatenate([p1, p2, p3, p4])
        
        df = pd.DataFrame({
            'time': pd.date_range(start="2025-01-01", periods=n_rows, freq="1min"),
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices + np.random.normal(0, 0.5, n_rows),
            'volume': np.random.randint(50, 500, n_rows),
            'tick_volume': np.random.randint(50, 500, n_rows)
        })
        
        # Calculate ATR for volatility guard
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean().fillna(1.0)
        df['avg_atr'] = df['atr'].rolling(50).mean().fillna(1.0)
        
        return df

    async def run_simulation(self):
        logger.info("⚔️ INICIANDO SIMULACRO DE COMBATE [DRY RUN] ⚔️")
        df = self.generate_synthetic_data()
        
        # Pre-process features
        features = feat_processor.process_dataframe(df)
        
        logger.info("\n--- [SIMULACIÓN START] ---")
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            feat_row = features.iloc[i]
            
            # 1. Physics Check
            acc_data = acceleration_engine.calculate_momentum_vector(df.iloc[i-5:i+1])
            is_accelerating = acc_data.get("is_valid", False)
            
            # 2. ML Prediction (Mocking hydrating sequence)
            # In real loop, we append to deque. Here we just mock result for speed if model not trained
            # But let's try to actually call the engine if possible
            # ml_engine.hydrate(self.symbol, [row['close']], [feat_row.to_dict()])
            
            # Simulate ML output heavily influenced by Physics for this test
            # If accelerating, ML should learn to buy.
            
            ml_pred = await ml_engine.predict_async(self.symbol, {
                **feat_row.to_dict(), 
                "close": row['close'], 
                "volume": row['volume'],
                "atr": row['atr'],
                "avg_atr": row['avg_atr']
            })
            
            p_win = ml_pred.get("p_win", 0.5)
            
            # 3. Decision Logic
            action = "HOLD"
            if is_accelerating and p_win > 0.6:
                action = "BUY"
            elif p_win < 0.4:
                action = "SELL"
                
            # DEBUG: Why no trades?
            if i % 25 == 0:
                 logger.info(f"[{row['time']}] ... Escaneando ... (Price: {row['close']:.2f}) | Conf: {p_win:.3f} | Accel: {is_accelerating}")

            # 4. Risk Gate
            if action != "HOLD":
                context = {
                    "close": row['close'], 
                    "atr": row['atr'], 
                    "avg_atr": row['avg_atr'],
                    "symbol": self.symbol,
                    "bid": row['close']
                }
                # Use Mock Cb
                allowed, regime = await risk_engine.check_trading_veto(self.symbol, context, 1.0)
                
                if allowed:
                    logger.info(f"[{row['time']}] 🚀 ORDEN EJECUTADA: {action} @ {row['close']:.2f} | Conf: {p_win:.2f} | Phys: {acc_data['acceleration']:.4f}")
                    self.history.append({"time": row['time'], "action": action, "price": row['close'], "result": "OPEN"})
                else:
                    logger.info(f"[{row['time']}] 🛡️ RISK VETO: {regime}")
            
            # Log periodic heartbeat
            if i % 25 == 0:
                logger.info(f"[{row['time']}] ... Escaneando ... (Price: {row['close']:.2f})")

        logger.info("\n--- [SIMULACIÓN COMPLETE] ---")
        logger.info(f"Trades Generated: {len(self.history)}")
        
if __name__ == "__main__":
    sim = BattlefieldSimulator()
    asyncio.run(sim.run_simulation())
