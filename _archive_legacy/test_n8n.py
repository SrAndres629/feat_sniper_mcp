import asyncio
import os
import sys

sys.path.append(os.getcwd())

from app.services.n8n_bridge import n8n_bridge

async def test_n8n_connection():
    print("\n🔗 CHECKING N8N LINK...")
    print("=======================")
    
    # Check if configured
    print(f"Current Config: {n8n_bridge.webhook_url or 'NOT SET'}")
    
    if not n8n_bridge.enabled:
        print("❌ N8N Bridge is DISABLED. Check keys/urls.")
        return

    # Send Test Event
    payload = {"status": "READY", "version": "5.0"}
    print(f"Sending Payload: {payload}")
    
    success = await n8n_bridge.send_system_event("SYSTEM_STARTUP", "READY", payload)
    
    if success:
        print("✅ SUCCESS: n8n acknowledged (200 OK).")
    else:
        print("❌ FAIL: n8n unreachable or error.")

if __name__ == "__main__":
    asyncio.run(test_n8n_connection())
