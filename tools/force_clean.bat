@echo off
echo ==========================================
echo   NEXUS EMERGENCY CLEANER (PANIC BUTTON)
echo ==========================================
echo.
echo [1/3] Matando procesos fantasma en Puerto 5555 (ZMQ Pub)...
powershell -Command "Get-NetTCPConnection -LocalPort 5555 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"

echo [2/3] Matando procesos fantasma en Puerto 5556 (ZMQ Sub)...
powershell -Command "Get-NetTCPConnection -LocalPort 5556 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"

echo [3/3] Ejecutando purga general de Python...
taskkill /F /IM python.exe /T 

echo.
echo [DONE] Sistema purgado. Ya puedes reiniciar nexus.bat sin miedo a instancias dobles.
pause
