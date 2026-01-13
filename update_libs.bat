@echo off
echo [FEAT-SNIPER] Updating Dependencies (Deep Rebuild)...
echo [INFO] This will reinstall pip packages from requirements.txt.
docker compose up -d --build
echo [OK] Rebuild complete.
