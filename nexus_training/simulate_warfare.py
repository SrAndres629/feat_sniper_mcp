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

# Strategic Cortex Hardware
from app.ml.strategic_cortex import policy_agent, state_encoder, StrategicAction
from app.ml.rlaif_critic import rlaif_critic
from nexus_core.microstructure import micro_scanner

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
        self.balance = 20.0 # Small Account Survival Trial
        self.positions = []
        self.history = []
        self.training_steps = 0

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

    async def run_simulation(self, episodes=5):
        logger.info(f"⚔️ INICIANDO GIMNASIO RL: {episodes} episodios de entrenamiento ⚔️")
        
        for ep in range(episodes):
            df = self.generate_synthetic_data(n_rows=500)
            features = feat_processor.process_dataframe(df)
            
            logger.info(f"\n--- EPISODIO {ep+1}/{episodes} START ---")
            
            # Reset account for Episode
            self.balance = 20.0
            
            for i in range(50, len(df)):
                row = df.iloc[i]
                feat_row = features.iloc[i]
                
                # 1. Physics & Microstructure (Simulated)
                prices = df['close'].iloc[i-50:i+1].values
                # Simulate some ticks for Microscanner
                mock_ticks = [
                    {'bid': row['close']-0.1, 'ask': row['close']+0.1, 'bid_vol': 100, 'ask_vol': 50}
                    for _ in range(20)
                ]
                micro_state = micro_scanner.process_tick_batch(mock_ticks, prices)
                
                # 2. Strategic Cortex Decision
                neural_probs = {'scalp': 0.75, 'day': 0.5, 'swing': 0.2} # Mock
                
                state_vec = state_encoder.encode(
                    account_state={"balance": self.balance, "phase_name": "SURVIVAL"},
                    microstructure=micro_scanner.get_dict(),
                    neural_probs=neural_probs,
                    physics_state={"titanium": "NEUTRAL", "feat_composite": 50.0}
                )
                
                # Agent makes a decision
                action, prob, value = policy_agent.select_action(state_vec)
                
                # 3. Simulate Outcome & Get Reward
                # (Simple model: if we buy/twin and price goes up in 10 bars, positive)
                future_price = df['close'].iloc[min(i+10, len(df)-1)]
                price_change_pct = (future_price - row['close']) / row['close']
                
                pnl = 0.0
                if action != StrategicAction.HOLD:
                    pnl = price_change_pct * 10.0 # Leveraged effect for P&L
                
                # RLAIF Critic gives the reward
                trade_result = {"profit": pnl, "type": action.name}
                context = {
                    "feat_score": feat_row.get('L1_Mean', 50.0),
                    "entropy": micro_state.entropy_score,
                    "drawdown_pct": max(0, 20.0 - self.balance) / 20.0
                }
                
                reward = rlaif_critic.critique_trade(trade_result, context)
                
                # RECORD EXPERIENCE
                policy_agent.record_experience(state_vec, action.value, reward, value, prob)
                
                self.balance += pnl
                
                # 4. Training (Every N steps)
                self.training_steps += 1
                if self.training_steps % 64 == 0:
                    logger.info("🧠 CORTEX: Updating Policy Network from Experience Buffer...")
                    # policy_agent.update() # Placeholder for actual PPO step logic if implemented
                
                if self.balance < 5: # Ruin check
                    logger.warning(f"💀 ACCOUNT BLOWN @ Step {i} | Restarting Episode...")
                    break
            
            logger.info(f"EPISODE {ep+1} COMPLETE. Final Balance: ${self.balance:.2f}")

        logger.info("\n🏆 WARFARE TRAINING COMPLETE.")

        
if __name__ == "__main__":
    sim = BattlefieldSimulator()
    asyncio.run(sim.run_simulation())
