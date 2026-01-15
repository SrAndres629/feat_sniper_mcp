@echo off
setlocal

:: Define MQL5 Source Directory (Relative to this script)
set "MQL_DIR=%~dp0FEAT_Sniper_Master_Core"

:: Common Installation Paths for MetaEditor
set "COMPILER_PATHS=C:\Program Files\LiteFinance MT5 Terminal\metaeditor64.exe;C:\Program Files\MetaTrader 5\metaeditor64.exe;C:\Program Files (x86)\MetaTrader 5\metaeditor64.exe"

set "COMPILER="

:: Find Compiler
for %%p in ("%COMPILER_PATHS:;=" "%") do (
    if exist %%p (
        set "COMPILER=%%~p"
        goto :FOUND
    )
)

:FOUND
if "%COMPILER%"=="" (
    echo [ERROR] metaeditor64.exe not found in common paths.
    echo Please compile manually or add MetaTrader 5 to PATH.
    exit /b 1
)

echo [INFO] Found Compiler: "%COMPILER%"

:: Array of Indicators to Compile
set "INDICATORS=FEAT_Visualizer.mq5 InstitutionalPVP.mq5 Ping.mq5"

for %%i in (%INDICATORS%) do (
    echo [COMPILE] Compiling %%i...
    "%COMPILER%" /compile:"%MQL_DIR%\%%i" /log:"%MQL_DIR%\compile_%%i.log"
    
    if exist "%MQL_DIR%\%%~ni.ex5" (
        echo [SUCCESS] %%i compiled successfully.
    ) else (
        echo [FAIL] %%i compilation failed. Check log: %MQL_DIR%\compile_%%i.log
    )
)

echo [DONE] Compilation process finished.
exit /b 0
