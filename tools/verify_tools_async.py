
import asyncio
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("--- ASYNC VERIFICATION START ---")
try:
    import mcp_server
    print("[OK] mcp_server imported")
except ImportError as e:
    print(f"[FAIL] mcp_server import failed: {e}")
    sys.exit(1)

async def check_tools():
    mcp = mcp_server.mcp
    print(f"MCP Object: {mcp}")
    
    # Try different ways to list tools depending on library version
    tools = []
    
    # Method 1: list_tools()
    if hasattr(mcp, 'list_tools'):
        print("Found list_tools method.")
        if asyncio.iscoroutinefunction(mcp.list_tools):
            tools = await mcp.list_tools()
        else:
            tools = mcp.list_tools()
    
    # Method 2: _tools list
    elif hasattr(mcp, '_tools'):
        print("Found _tools attribute.")
        tools = mcp._tools
        
    # Method 3: registry
    elif hasattr(mcp, 'tool_registry'):
         print("Found tool_registry.")
         tools = mcp.tool_registry
         
    print(f"Tools Found: {len(tools)}")
    if len(tools) > 0:
        print("--- TOOLS LIST ---")
        for t in tools:
            # t might be a Tool object or dict
            name = getattr(t, 'name', str(t))
            print(f"- {name}")
        print("------------------")
        return True
        print("[WARNING] No tools found via introspection.")
        
    # Try calling a tool directly to verify access
    try:
        print("Attempting to call 'sys_audit_status' direct...")
        if hasattr(mcp_server, 'sys_audit_status'):
            func = mcp_server.sys_audit_status
            if asyncio.iscoroutinefunction(func):
                res = await func()
            else:
                res = func()
            print(f"Tool Result: {res}")
            return True
        else:
            print("sys_audit_status not found in mcp_server module.")
            return False
    except Exception as e:
        print(f"Tool Execution Failed: {e}")
        return False


if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(check_tools())
    except Exception as e:
        print(f"[ERROR] Async check failed: {e}")
    finally:
        print("--- ASYNC VERIFICATION END ---")
