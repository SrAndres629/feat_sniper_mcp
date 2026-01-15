
import asyncio
import sys
import os
import json

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def prove_mcp():
    print("=== MCP TELEMETRY CHECK ===")
    try:
        import mcp_server
        # We need to initialize the necessary components or mock the connection if possible
        # But calling the tool directly relies on global state in mcp_server
        # Ideally, we should use the verify script approach where we just invoke the tool function
        
        if hasattr(mcp_server, 'market_get_telemetry'):
             print("Invoking market_get_telemetry...")
             tool_func = mcp_server.market_get_telemetry
             if hasattr(tool_func, 'fn'): tool_func = tool_func.fn # Unwrap
             
             # This might fail if MT5 is not actually connected in THIS process
             # But the user asked to "prove I can use the mcp". 
             # If the MCP is running as a server, I should refer to the running server?
             # But I am running code in the same environment.
             # The existing architecture runs mcp_server as a standalone process.
             # Importing it here initializes a NEW instance. 
             # If that new instance connects to MT5, great.
             
             res = await tool_func(symbol="XAUUSD", timeframe="M5")
             print(json.dumps(res, indent=2, default=str))
        else:
             print("Tool not found.")
             
    except Exception as e:
        print(f"MCP Check Failed: {e}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(prove_mcp())
