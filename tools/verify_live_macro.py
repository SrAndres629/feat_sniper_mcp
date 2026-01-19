import sys
import os
import datetime
sys.path.append(os.getcwd())

from nexus_core.fundamental_engine.engine import FundamentalEngine
from nexus_core.fundamental_engine.calendar_client import CalendarClient
from nexus_core.fundamental_engine.risk_modulator import DEFCON

def verify_live_macro():
    print("🌍 MACRO SENTINEL: LIVE DATA VERIFICATION")
    print("=" * 60)
    
    # Initialize with ForexFactory provider
    engine = FundamentalEngine(calendar_provider="forexfactory")
    
    # 1. Test Data Fetching
    print("\n[STEP 1] Fetching Live Data from ForexFactory...")
    try:
        events = engine.calendar.get_upcoming_events(hours_ahead=168) # 1 week
        print(f"   ✅ Successfully fetched {len(events)} events for the upcoming week.")
        
        if len(events) > 0:
            print("\n   Upcoming High Impact Events:")
            high_impact = [e for e in events if e.impact.name == "HIGH"]
            for e in high_impact[:5]:
                print(f"   - [{e.timestamp}] {e.currency}: {e.event_name}")
    except Exception as e:
        print(f"   ❌ Failed to fetch live data: {e}")
        return

    # 2. Test Proximity & DEFCON
    print("\n[STEP 2] Calculating Current System Risk (DEFCON)...")
    result = engine.check_event_proximity(currencies=["USD", "EUR", "GBP", "JPY"])
    
    print(f"   Current DEFCON: {result['defcon'].name}")
    print(f"   Macro Regime: {result['macro_regime']}")
    print(f"   Position Multiplier: {result['position_multiplier']:.4f}")
    
    if result['next_event']:
        e = result['next_event']
        print(f"   Next Relevant Event: {e.event_name} ({e.currency}) in {result['minutes_until']:.1f} minutes.")
    else:
        print("   No imminent high-impact events detected.")

    # 3. Validation
    if result['defcon'] != DEFCON.DEFCON_5:
        print("\n   ⚠️ SYSTEM ALERT: Risk reduction active due to upcoming macro events.")
    else:
        print("\n   ✅ SYSTEM CLEAR: Normal trading conditions.")

    print("\n" + "=" * 60)
    print("🏆 LIVE MACRO VERIFICATION COMPLETE.")

if __name__ == "__main__":
    verify_live_macro()
