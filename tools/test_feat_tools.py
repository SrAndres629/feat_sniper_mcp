
import asyncio
import sys
import os
import json

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

async def run_demo():
    print("=== FEAT TOOL VERIFICATION ===")
    
    try:
        import mcp_server
        print("[OK] MCP Server Imported")
    except ImportError as e:
        print(f"[FAIL] Import Error: {e}")
        return

    # Helper to call tool
    async def call_tool(tool_obj, name, **kwargs):
        print(f"\n--- Testing: {name} ---")
        print(f"Type: {type(tool_obj)}")
        
        func = None
        # Unwrapping strategies for FastMCP / Decorators
        if hasattr(tool_obj, 'fn'): 
            func = tool_obj.fn
        elif hasattr(tool_obj, '__wrapped__'):
            func = tool_obj.__wrapped__
        elif hasattr(tool_obj, 'func'):
            func = tool_obj.func
        elif asyncio.iscoroutinefunction(tool_obj):
            func = tool_obj
        
        if func:
            print(f"Callable found: {func}")
            try:
                res = await func(**kwargs)
                print(f"RESULT ({name}):")
                print(json.dumps(res, indent=2, default=str))
                return True
            except Exception as e:
                print(f"EXECUTION ERROR: {e}")
                return False
        else:
            print(f"COULD NOT EXTRACT CALLABLE from {tool_obj}")
            print(f"Dir: {dir(tool_obj)}")
            return False

    # 1. Test sys_audit_status
    if hasattr(mcp_server, 'sys_audit_status'):
        await call_tool(mcp_server.sys_audit_status, "sys_audit_status")
    else:
        print("[MISSING] sys_audit_status not found")

    # 2. Test brain_run_inference
    if hasattr(mcp_server, 'brain_run_inference'):
        # Mock context data
        ctx = {"close": 2000.50, "bid": 2000.40, "features": [0.5, 0.1, 0.9, 0.2]}
        await call_tool(mcp_server.brain_run_inference, "brain_run_inference", context_data=ctx)
    else:
        print("[MISSING] brain_run_inference not found")

    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_demo())
