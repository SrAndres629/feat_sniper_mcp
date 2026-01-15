
import sys
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio

# Hardcode path to n8n mcp
MOCK_PATH = r"c:\Users\acord\OneDrive\Desktop\Bot\n8n_dev_mcp"
sys.path.append(MOCK_PATH)

# Inject Auth from User Config
os.environ["N8N_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxNjViMDA5ZS1jNTM5LTRlOGUtOGUyNi1mODZlOGQyMTRjNWMiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzY3NTI2NjY2fQ.QRYJ85hOm4Ar3A9xH6Bq_UsIQipn9hgxVN0yL83qJDM"
os.environ["N8N_BASE_URL"] = "http://localhost:5678/api/v1"

print("--- N8N ARCHITECT VERIFICATION ---")

try:
    from app.main import mcp, app
    print("✅ Import Successful: app.main")
except ImportError as e:
    print(f"❌ Import Failed: {e}")
    sys.exit(1)

print(f"MCP Name: {mcp.name}")

# Direct Attribute Check (Plan B)
expected_tools = [
    "workflow_architect_expert",
    "operational_surgeon_expert", 
    "system_oracle_expert",
    "infrastructure_guardian_expert"
]

found_count = 0
import app.main
for t_name in expected_tools:
    if hasattr(app.main, t_name):
        print(f" - Found Function: {t_name}")
        found_count += 1
    else:
        print(f" - MISSING: {t_name}")

if found_count > 0:
    print("RESULT: PASS")
else:
    print("RESULT: FAIL")
