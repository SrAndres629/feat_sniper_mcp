import sys
import os
import asyncio
import logging
import pandas as pd
import numpy as np
import torch
import random
import time
import json
import argparse
from datetime import datetime

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor import feat_processor
from app.ml.ml_engine import ml_engine
from app.services.risk import risk_engine
from nexus_core.acceleration import acceleration_engine

# Strategic Cortex Hardware
from app.ml.strategic_cortex import policy_agent, state_encoder, StrategicAction
from nexus_core.neural_health import neural_health
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
        self.status_file = "data/simulation_status.json"
        os.makedirs("data", exist_ok=True)

    def write_status(self, current_ep: int, total_ep: int, balance: float, running: bool = True):
        """Writes simulation status to file for dashboard display."""
        status = {
            "current_episode": current_ep,
            "total_episodes": total_ep,
            "current_balance": balance,
            "running": running,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.status_file, 'w') as f:
            json.dump(status, f)

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
        
        # [HERD RADAR] Stochastic Herd Model (PhD Logic)
        # Features: 1) Momentum Lag, 2) Contrarian Persistence (averaging down), 3) Random Noise
        
        sent_long = np.zeros(n_rows)
        # Initial state (balanced)
        current_long = 50.0
        
        for i in range(1, n_rows):
            # Calculate price velocity (lagged)
            # If price went UP last 5 periods, retail starts selling (contrarian)
            # but with a LAG.
            window = 10
            if i > window:
                price_change = (prices[i] - prices[i-window]) / prices[i-window]
                
                # Deterministic drift: Price UP -> Retail sells (long goes DOWN)
                # We model "averaging down": Retail sells into rallies (contrarian)
                # Phase 8: Volatility-Induced Panic (PhD Logic)
                # Volatility increases noise and reduces persistence (Panic)
                current_vol = df['atr'].iloc[i]
                vol_ratio = current_vol / df['avg_atr'].iloc[i] if df['avg_atr'].iloc[i] > 0 else 1.0
                
                # Persistence drops as volatility rises (Panic / Capitulation)
                persistence = np.clip(0.95 - (vol_ratio * 0.05), 0.5, 0.98)
                
                # Noise increases with volatility
                stochastic_noise = np.random.normal(0, 1.5 * vol_ratio)
                
                # Drift: Price UP -> Retail sells (contrarian drift)
                # But during extreme vol, they might "Panic Buy" the top or "Panic Sell" the bottom (Momentum Chase)
                if vol_ratio > 2.5:
                    # Panic Momentum: They flip and follow the trend at the worst possible time
                    drift = price_change * 500.0  
                else:
                    # Normal Contrarian bias
                    drift = -price_change * 1000.0 
                
                current_long = (current_long * persistence) + (drift * (1 - persistence)) + stochastic_noise
            
            sent_long[i] = np.clip(current_long, 10, 90)
            
        df['mock_long_pct'] = sent_long
        df['mock_short_pct'] = 100 - sent_long
        
        # Calculate ATR for volatility guard
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean().fillna(1.0)
        df['avg_atr'] = df['atr'].rolling(50).mean().fillna(1.0)
        
        return df

    async def run_imitation_bootcamp(self, n_samples=500):
        """
        [PHASE 1] COLD START PROTECTION
        Teaches the Agent the baseline FEAT Sniper rules via Imitation Learning.
        """
        logger.info(f"🎓 INICIANDO BOOTCAMP DE IMITACIÓN: {n_samples} muestras...")
        
        df = self.generate_synthetic_data(n_rows=n_samples + 100)
        states = []
        teacher_actions = []
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            prices = df['close'].iloc[i-50:i+1].values
            
            # 1. State Encoding
            # We simulate the microstructure for the teacher context
            micro_state_mock = {
                "entropy_score": 0.3 if i % 10 < 3 else 0.7, # Alternating focus
                "ofi_z_score": 1.5 if i % 20 < 10 else -1.5,
                "hurst": 0.65
            }
            
            state_vec = state_encoder.encode(
                account_state={"balance": 20.0, "phase_name": "SURVIVAL"},
                microstructure=micro_state_mock,
                neural_probs={'scalp': 0.8 if micro_state_mock['entropy_score'] < 0.4 else 0.4},
                physics_state={"titanium": "TITANIUM_SUPPORT" if i % 30 < 5 else "NEUTRAL", "feat_composite": 75.0 if i % 20 < 10 else 40.0}
            )
            
            # 2. TEACHER LOGIC (Deterministic Legacy Rules)
            feat_score = (state_vec.feat_composite * 100)
            entropy = state_vec.entropy_score
            
            if feat_score > 80 and entropy < 0.35:
                action = StrategicAction.TWIN_SNIPER
            elif feat_score > 60 and entropy < 0.5:
                action = StrategicAction.STANDARD
            elif entropy > 0.65:
                action = StrategicAction.DEFENSIVE
            else:
                action = StrategicAction.HOLD
                
            states.append(state_vec)
            teacher_actions.append(action)
            
            if len(states) >= n_samples:
                break
                
        # Train Agent to replicate Teacher
        policy_agent.pretrain(states, teacher_actions, epochs=10)
        policy_agent.save_weights("models/strategic_policy_bootcamp.pth")

    async def run_simulation(self, episodes=5):
        # 1. Bootcamp - Pre-training
        await self.run_imitation_bootcamp(n_samples=500)
        
        logger.info(f"\n⚔️ INICIANDO GIMNASIO RL: {episodes} episodios de entrenamiento ⚔️")
        
        for ep in range(episodes):
            df = self.generate_synthetic_data(n_rows=500)
            
            logger.info(f"\n--- EPISODIO {ep+1}/{episodes} START ---")
            
            self.balance = 20.0
            
            for i in range(50, len(df)):
                row = df.iloc[i]
                
                prices = df['close'].iloc[i-50:i+1].values
                mock_ticks = [
                    {'bid': row['close']-0.1, 'ask': row['close']+0.1, 'bid_vol': 100, 'ask_vol': 50}
                    for _ in range(20)
                ]
                micro_state = micro_scanner.process_tick_batch(mock_ticks, prices)
                
                # [HERD RADAR] Inject Mock Sentiment into State
                mock_sentiment = {
                    "long_pct": row['mock_long_pct'],
                    "short_pct": row['mock_short_pct'],
                    "contrarian_score": (row['mock_short_pct'] - row['mock_long_pct']) / 100.0,
                    "liquidity_above": 1.0 if row['mock_short_pct'] > 60 else 0.0,
                    "liquidity_below": 1.0 if row['mock_long_pct'] > 60 else 0.0
                }
                
                state_vec = state_encoder.encode(
                    account_state={"balance": self.balance, "phase_name": "SURVIVAL"},
                    microstructure=micro_scanner.get_dict(),
                    neural_probs={'scalp': 0.7, 'day': 0.5, 'swing': 0.2},
                    physics_state={"titanium": "NEUTRAL", "feat_composite": 50.0},
                    sentiment_state=mock_sentiment # Pass mock sentiment to encoder
                )
                
                action, prob, value = policy_agent.select_action(state_vec)
                
                # [NEURAL HEALTH] Track prediction for drift analysis
                sim_trade_id = f"SIM-{ep}-{i}"
                if action != StrategicAction.HOLD:
                    neural_health.log_prediction(sim_trade_id, prob, action.name)
                
                # [IRON FORGE] ADVERSARIAL GYM
                # 1. Latency Simulation (The Brain vs The Wire)
                # In real life, 200ms lag kills the perfect entry.
                latency = random.uniform(0.05, 0.300) # 50ms to 300ms
                time.sleep(latency) 
                
                # 2. Slippage / Spread Stretcher (The Broker vs You)
                # Broker widens spread or slips you on high vol
                tick_size = 0.01
                slippage_ticks = random.choice([-1, 0, 1, 2]) # Biased slightly against
                slippage = slippage_ticks * tick_size
                
                execution_price = row['close'] + slippage if action != StrategicAction.HOLD else row['close']
                
                # Simulate price movement from EXECUTION PRICE (not theoretical close)
                future_price = df['close'].iloc[min(i+10, len(df)-1)]
                price_change_pct = (future_price - execution_price) / execution_price
                
                # Simple PnL: Action scaling
                pnl = 0.0
                if action == StrategicAction.TWIN_SNIPER: pnl = price_change_pct * 30.0 # Aggressive
                elif action == StrategicAction.STANDARD: pnl = price_change_pct * 15.0
                elif action == StrategicAction.DEFENSIVE: pnl = price_change_pct * 5.0
                
                # RLAIF Critic
                trade_result = {"profit": pnl, "type": action.name}
                reward = rlaif_critic.critique_trade(trade_result, {
                    "feat_score": 50.0, 
                    "entropy": micro_state.entropy_score,
                    "drawdown_pct": max(0, 20.0 - self.balance) / 20.0,
                    "slippage_paid": slippage # Feed this pain to the critic if possible
                })
                
                # [ADVERSARIAL] Penalty for High Entropy Entries
                if micro_state.entropy_score > 0.6 and action != StrategicAction.HOLD:
                    reward -= 2.0 # Explicit punishment for gambling in noise
                
                # [MACRO DISCIPLINE] Mock News Window - PEDAGOGICAL PUNISHMENT
                # Every 60-120 steps, simulate a "news event" window (10 steps long)
                # If agent trades during this window, MASSIVE penalty (even if trade would profit)
                mock_news_window = (i % random.randint(60, 120)) < 10
                if mock_news_window and action != StrategicAction.HOLD:
                    # Agent tried to trade during "news time" - PUNISH SEVERELY
                    reward -= 5.0  # Massive penalty for news disobedience
                    if i % 50 == 0:
                        logger.warning(f"⚠️ Step {i}: Agent traded during mock NEWS WINDOW! Penalty -5.0")
                
                # [CONTRARIAN REWARD] Reward following the liquidity bias
                if action != StrategicAction.HOLD:
                    # If we BUY while retail is SHORT (contrarian bullish)
                    if action in [StrategicAction.TWIN_SNIPER, StrategicAction.STANDARD] and mock_sentiment["contrarian_score"] > 0.2:
                        reward += 1.5 # Reward for hunting liquidity ABOVE
                    # If we SELL while retail is LONG (contrarian bearish)
                    elif action == StrategicAction.DEFENSIVE and mock_sentiment["contrarian_score"] < -0.2:
                        reward += 1.5 # Reward for hunting liquidity BELOW
                    # If we follow the herd (BUY while retail is LONG)
                    elif action in [StrategicAction.TWIN_SNIPER, StrategicAction.STANDARD] and mock_sentiment["contrarian_score"] < -0.4:
                        reward -= 2.0 # Heavy penalty for herd FOMO

                
                # Record (Simplified storage)
                policy_agent.record_experience(state_vec, action, reward, state_vec, False)
                
                self.balance += pnl
                
                # [NEURAL HEALTH] Resolve trade result
                if action != StrategicAction.HOLD:
                    neural_health.resolve_prediction(sim_trade_id, pnl)
                
                if self.balance < 5:
                    logger.warning(f"💀 RUIN @ Step {i}")
                    break
            
            logger.info(f"EPISODE {ep+1} DONE. Balance: ${self.balance:.2f}")
            self.write_status(ep + 1, episodes, self.balance, running=True)

        self.write_status(episodes, episodes, self.balance, running=False)
        logger.info("\n🏆 WARFARE TRAINING COMPLETE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEAT Sniper Battlefield Simulator")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    args = parser.parse_args()
    
    sim = BattlefieldSimulator()
    asyncio.run(sim.run_simulation(episodes=args.episodes))
