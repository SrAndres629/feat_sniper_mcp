import unittest
import sys
import os
import logging
from datetime import datetime, timedelta

# Logs
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Add root
sys.path.append(os.getcwd())

from app.skills.market_physics import market_physics
from app.skills.feat_chain import feat_full_chain_institucional

class TestFEATPipeline(unittest.IsolatedAsyncioTestCase):
    async def test_full_feat_sequence_valid(self):
        print("\n[INFO] STARTING DETERMINISTIC FEAT SIMULATION...")

        # 1. Warmup Physics (Stable Baseline)
        # 60 seconds of stable market.
        base_price = 2000.0
        start_time = datetime(2024, 1, 1, 9, 0, 0) # 09:00 AM
        
        for i in range(60):
            current_time = start_time + timedelta(seconds=i)
            market_physics.ingest_tick({
                'tick_volume': 100.0, 
                'bid': base_price
            }, force_timestamp=current_time.timestamp())
            
        print("[OK] Physics Warmed Up (Delta T = 1.0s).")

        # 2. Simulate Spike (1 sec later)
        # Vol: 1000 (10x Avg). Price: +5.0 (Velocity = 5.0/1.0 = 5.0).
        # Accel = Ratio(10) * |Vel(5)| = 50.
        # Threshold: Mean(0) + 2*Std(0) = 0? (Warmup had velocity 0).
        # If Velocity=0 in warmup, Accel=0.
        # Threshold should be low. 50 > 0. Valid.
        
        spike_time = start_time + timedelta(seconds=60) # 09:01:00 (Valid Killzone Logic?)
        # 09:01 is NY Killzone? No. 
        # Killzones: London 02-05, NY 07-11. 
        # 09:01 is INSIDE NY Killzone (07-11). OK.
        
        tick_data = {
            'tick_volume': 1000.0, 
            'bid': base_price + 5.0,
            'structure': 'BULLISH_CHOCK',
            'simulated_time': spike_time
        }
        
        print(f"[TIME] Trigger Time: {spike_time.strftime('%H:%M:%S')}")

        # 3. Execute Analysis
        is_valid = await feat_full_chain_institucional.analyze(tick_data, tick_data['bid'])
        
        print(f"[RESULT] Chain Result: {is_valid}")
        
        self.assertTrue(is_valid, "Chain REJECTED valid High-Sigma Setup.")

if __name__ == '__main__':
    unittest.main()
