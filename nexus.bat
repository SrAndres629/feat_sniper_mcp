@echo off
setlocal
cd /d "%~dp0"
title NEXUS COMMAND CENTER - SENIOR AUDITOR MODE
color 0A
cls

echo ==============================================================================
echo      NN   NN  EEEEEEE  XX   XX  UU   UU   SSSSS   PRIMME
echo      NNN  NN  EE        XX XX   UU   UU  SS       P    P
echo      NN N NN  EEEEE      XXX    UU   UU   SSSSS   PPPPPP
echo      NN  NNN  EE        XX XX   UU   UU       SS  P
echo      NN   NN  EEEEEEE  XX   XX   UUUUU    SSSSS   P
echo.
echo      [ MASTER AUDIT + AUTO-HEALING PROTOCOL v3.0 ]
echo ==============================================================================
echo.

echo [FASE 1] Verificacion de Software e Integridad...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] Python no detectado.
    goto ERROR
)
echo [OK] Python Environment Detectado.

echo [FASE 2] Inicializando Orquestador Maestro...
echo.

python nexus_control.py start
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo [FATAL] Fallo critico en el orquestador. Revisa AUDIT_REPORT.md
    pause
    color 0A
    goto END
)

echo.
echo [SUCCESS] Sistema Estable. Reporte generado en AUDIT_REPORT.md
goto END

:ERROR
color 0C
echo [ERROR] Deteniendo ejecucion por fallo de prerrequisitos.
pause
goto END

:END
pause
