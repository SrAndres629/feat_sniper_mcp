
import subprocess
import time
import sys
import os

# Command to test: cmd /c npx -y @supabase/mcp-server-supabase@latest --access-token ...
token = "sbp_659e917b703436a28010d9fbd4cc2e6e016087f2"
cmd = ["cmd", "/c", "npx", "-y", "@supabase/mcp-server-supabase@latest", "--access-token", token]

print(f"Testing Launch: {' '.join(cmd)}")

try:
    # Launch process with pipes
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False # We are calling cmd explicitly
    )
    
    # Wait 5 seconds to see if it crashes
    time.sleep(5)
    
    poll = proc.poll()
    if poll is None:
        print("RESULT: SUCCESS (Process still running)")
        proc.terminate()
    else:
        print(f"RESULT: FAILED (Exit Code: {poll})")
        stdout, stderr = proc.communicate()
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")

except Exception as e:
    print(f"RESULT: CRASH ({e})")
