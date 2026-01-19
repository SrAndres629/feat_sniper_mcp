import sys
import os
import pandas as pd
import numpy as np
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor.time import TimeFeatureProcessor

def verify_integration():
    print("🌍 [TRIDENT] MACRO TENSOR INTEGRATION CHECK")
    print("-------------------------------------------")
    
    try:
        processor = TimeFeatureProcessor()
        
        # Mock Data
        timestamp = datetime.datetime.now(datetime.timezone.utc)
        closes = pd.Series([100.0, 101.0, 102.0])
        volumes = pd.Series([1000.0, 1200.0, 1100.0])
        opens = pd.Series([99.0, 100.0, 101.0])
        
        print("   ► Processing Time Snapshot...")
        tensors = processor.process(timestamp, closes, volumes, opens)
        
        print(f"   ► Keys Detected: {len(tensors.keys())}")
        
        required_keys = ["macro_safe", "macro_caution", "macro_danger", "macro_position_multiplier"]
        
        missing = [k for k in required_keys if k not in tensors]
        
        if not missing:
            print("   ✅ SUCCESS: All Macro Tensors Present.")
            print(f"   ► Sample Safe: {tensors['macro_safe']}")
            print(f"   ► Sample Multiplier: {tensors['macro_position_multiplier']}")
        else:
            print(f"   ❌ FAIL: Missing Keys: {missing}")
            
    except Exception as e:
        print(f"   ❌ FAIL: Crash during processing. {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_integration()
