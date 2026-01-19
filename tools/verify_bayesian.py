import sys
import os
import datetime
import pytz
import numpy as np

sys.path.append(os.getcwd())

from app.ml.feat_processor.chronos_tensor import ChronosTensorFactory

def run_test():
    print("🧠 BAYESIAN CHRONOS: PROBABILITY FIELD DIAGNOSTIC")
    print("================================================")
    
    factory = ChronosTensorFactory()
    
    # Test Times (Bolivia based)
    # We need to construct UTC times that map to these Bolivia times.
    # Bolivia is UTC-4.
    
    # 1. London Raid: 02:30 Bolivia -> 06:30 UTC
    t_raid = datetime.datetime(2026, 1, 20, 6, 30, tzinfo=datetime.timezone.utc)
    
    # 2. NY Open: 09:30 Bolivia -> 13:30 UTC
    t_open = datetime.datetime(2026, 1, 20, 13, 30, tzinfo=datetime.timezone.utc)
    
    # 3. Lunch: 13:00 Bolivia -> 17:00 UTC
    t_lunch = datetime.datetime(2026, 1, 20, 17, 00, tzinfo=datetime.timezone.utc)
    
    scenarios = [
        ("LONDON RAID (02:30)", t_raid),
        ("NY OPEN (09:30)", t_open),
        ("LUNCH DEAD ZONE (13:00)", t_lunch)
    ]
    
    for name, t in scenarios:
        print(f"\n[{name}]")
        payload = factory.process(t)
        
        p_manip = payload['prob_manipulation'][0]
        p_exp = payload['prob_expansion'][0]
        p_liq = payload['prob_liquidity'][0]
        
        print(f"   ► P(Manipulation): {p_manip:.4f}")
        print(f"   ► P(Expansion):    {p_exp:.4f}")
        print(f"   ► P(Liquidity):    {p_liq:.4f}")
        
        # Validations
        if "RAID" in name:
            if p_manip > 0.8: print("   ✅ SUCCESS: High Manipulation Prior.")
            else: print("   ❌ FAIL: Low Manipulation Prior.")
            
        if "OPEN" in name:
            if p_exp > 0.8: print("   ✅ SUCCESS: High Expansion Prior.")
            else: print("   ❌ FAIL: Low Expansion Prior.")
            
        if "LUNCH" in name:
            if p_exp < 0.3 and p_manip < 0.3: print("   ✅ SUCCESS: Dead Zone confirmed.")
            else: print("   ⚠️ NOTE: Residual probability present.")

    print("================================================")

if __name__ == "__main__":
    run_test()
