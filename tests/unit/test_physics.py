import unittest
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from app.skills.market_physics import market_physics

class TestMarketPhysics(unittest.TestCase):
    def test_acceleration_trigger(self):
        print("\n🧪 STARTING PHYSICS SIMULATION...")
        
        # 1. Warmup (Flat Market)
        # Vol = 100, Price moves 0.1 per tick
        base_price = 2000.0
        for i in range(60):
            tick = {
                'tick_volume': 100.0,
                'bid': base_price + (i * 0.1)
            }
            market_physics.ingest_tick(tick)
            
        print("✅ Warmup Complete (60 ticks). Baseline Established.")
        
        # 2. Inject Anomaly (The Sniper Trigger)
        # Vol = 800 (8x mean), Price Jump = 5.0 (50x speed)
        spike_tick = {
            'tick_volume': 800.0,
            'bid': base_price + 6 + 5.0 
        }
        
        print(f"⚡ Injecting Spike: Vol={spike_tick['tick_volume']}, Price={spike_tick['bid']}")
        regime = market_physics.ingest_tick(spike_tick)
        
        # 3. Assertions
        if regime:
            print(f"📊 Result: AccelScore={regime.acceleration_score:.2f}, IsAccel={regime.is_accelerating}")
            self.assertTrue(regime.is_accelerating, "FAILED: Acceleration not detected on massive spike.")
            self.assertTrue(regime.vol_z_score > 3.0, "FAILED: Z-Score too low.")
        else:
            self.fail("Regime returned None (not enough data?)")

if __name__ == '__main__':
    unittest.main()
