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
echo      [ MASTER AUDIT AND AUTO-HEALING PROTOCOL v3.1 ]
echo ==============================================================================
echo.

echo [FASE 1] Verificacion de Software e Integridad...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL] Python no detectado.
    goto ERROR
)
echo [OK] Python Environment Detectado.

echo [INFO] Inicializando entorno institucional...
echo [INFO] Cargando configuraciones de seguridad...

:: Command Line Argument Handler
if /I "%1"=="start" goto START
if /I "%1"=="audit" goto AUDIT

choice /C SAQ /N /M "[S] Start System  [A] Audit Only  [Q] Quit : "

if errorlevel 3 goto QUIT
if errorlevel 2 goto AUDIT
if errorlevel 1 goto START

:START
cls
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

:AUDIT
cls
echo [EXEC] Ejecutando Auditoria Profunda (Sin Arranque)...
python nexus_control.py audit
echo.
pause
goto END

:QUIT
echo [INFO] Saliendo...
goto END

:ERROR
color 0C
echo [ERROR] Deteniendo ejecucion por fallo de prerrequisitos.
pause
goto END

:END
pause
