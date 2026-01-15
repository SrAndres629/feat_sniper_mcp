import asyncio
import os
import sys

# Force path
sys.path.append(os.getcwd())

from app.core.config import settings
from supabase import create_client, Client

async def check_table():
    print("--- UNIT TEST: DB CONNECTION ---")
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY
    
    if not url or not key:
        print("❌ Supabase Config Missing")
        return

    try:
        client: Client = create_client(url, key)
        print(f"✅ Client Created: {url[:15]}...")
        
        # Try to select 1 row from bot_activity_log
        # We don't insert, just check if it errors
        print("QUERY: bot_activity_log (HEAD)...")
        res = client.table("bot_activity_log").select("*").limit(1).execute()
        print(f"✅ Connection Successful. Rows: {len(res.data)}")
        
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        # Check specific error code
        if "relation" in str(e) and "does not exist" in str(e):
             print("   -> Table 'bot_activity_log' DOES NOT EXIST.")

if __name__ == "__main__":
    asyncio.run(check_table())
