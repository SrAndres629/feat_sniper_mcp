@echo off
title FEAT KILL SWITCH
color 0C
echo.
echo [KILL SWITCH] DETECTING ROGUE PROCESSES...
echo.

REM Kill Python Brains (Targeted)
taskkill /F /IM python.exe /FI "WINDOWTITLE eq FEAT NEXUS*" >nul 2>&1

REM Kill Dashboard (Chrome)
echo [BROWSER] CLOSING DASHBOARD...
powershell -NoProfile -Command "$browser='chrome'; Get-Process $browser -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -match 'FEAT NEXUS' -or $_.MainWindowTitle -match 'dashboard.html' } | Stop-Process -Force"

echo.
echo [SYSTEM] ALL SYSTEMS OFFLINE.
echo.
pause
