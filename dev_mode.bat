@echo off
echo [FEAT-SNIPER] Starting in DEV MODE (Hot Reload Active)...
echo [INFO] Skipping build. Code changes in local folder will be reflected immediately.
docker compose up -d
echo [OK] Containers started. Logs:
docker compose logs -f mcp-brain
