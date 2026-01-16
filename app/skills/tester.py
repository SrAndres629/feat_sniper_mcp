import logging
import os
import subprocess
import configparser
from typing import Dict, Any, Optional

# FAIL-FAST: Use centralized MT5 from mt5_conn (no silent mocks)
from app.core.config import settings
from app.core.mt5_conn import mt5_conn, mt5, MT5_AVAILABLE

if not MT5_AVAILABLE:
    raise ImportError(
        "MetaTrader5 library not found. This module requires a real MT5 connection. "
        "Install: pip install MetaTrader5 (Windows only)"
    )

logger = logging.getLogger("MT5_Bridge.Skills.Tester")

async def run_strategy_test(
    expert_name: str, 
    symbol: str, 
    period: str = "H1", 
    date_from: str = "2025.01.01",
    date_to: str = "2025.01.31",
    model: int = 1, # 0: StartBar, 1: OHLC, 2: EveryTick, 3: RealTicks, 4: MathCalc
    deposit: int = 10000,
    leverage: int = 100,
    optimization: int = 0 # 0: Disabled, 1: Slow, 2: Fast
) -> Dict[str, Any]:
    """
    Automates the MT5 Strategy Tester by creating an INI file and running terminal64.exe via CLI.
    """
    
    # 1. Get Terminal Path
    term_info = await mt5_conn.execute(mt5.terminal_info)
    if not term_info:
        return {"status": "error", "message": "MT5 Terminal Info not available."}
    
    terminal_exe = os.path.join(term_info.path, "terminal64.exe")
    if not os.path.exists(terminal_exe):
        return {"status": "error", "message": f"Terminal executable not found at {terminal_exe}"}

    # 2. Prepare INI Config
    # We write to a temporary INI file in the common path or app path
    
    report_name = f"Report_{expert_name}_{symbol}_{period}.html"
    report_path = os.path.join(term_info.path, report_name) # Save in terminal root for simplicity
    
    config = configparser.ConfigParser()
    config.optionxform = str # Preserve case
    
    config["Tester"] = {
        "Expert": expert_name,
        "Symbol": symbol,
        "Period": period,
        "Deposit": str(deposit),
        "Leverage": f"1:{leverage}",
        "Model": str(model), # 1=OHLC, 0=EveryTick
        "ExecutionMode": "0", # Normal
        "Optimization": str(optimization), 
        "DateFrom": date_from,
        "DateTo": date_to,
        "Report": report_name,
        "ReplaceReport": "1",
        "ShutdownTerminal": "1", # Close after test
        "Visual": "0", # Non-visual mode
    }
    
    # Common INI setup
    config["Common"] = {
        "Login": str(settings.MT5_LOGIN),
        "Password": settings.MT5_PASSWORD,
        "Server": settings.MT5_SERVER,
        "CertPassword": ""
    }

    ini_filename = "mcp_tester.ini"
    ini_path = os.path.join(term_info.path, ini_filename)
    
    try:
        with open(ini_path, "w") as configfile:
            config.write(configfile)
    except Exception as e:
        return {"status": "error", "message": f"Failed to write INI config: {str(e)}"}

    # 3. Execute Terminal
    # Command: terminal64.exe /config:mcp_tester.ini
    
    logger.info(f"Launching Strategy Tester with config: {ini_path}")
    
    try:
        # We assume the terminal will close itself due to ShutdownTerminal=1
        # But we verify if it's already running. If we run portable/config, it launches a new instance usually.
        # Warning: This blocks until terminal closes if we use check=True
        
        result = subprocess.run(
            [terminal_exe, f"/config:{ini_path}"],
            capture_output=True,
            text=True,
            timeout=300 # 5 min timeout safety
        )
        
        # 4. Check for Report
        if os.path.exists(report_path):
             return {
                "status": "success",
                "message": "Test completed.",
                "report_path": report_path,
                "stdout": result.stdout,
                "note": "Parse the HTML report for specific metrics (Profit, DD, etc.)"
            }
        else:
            return {
                "status": "warning",
                "message": "Terminal executed but report was not found. Check Journal.",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Strategy Tester timed out (5 mins)."}
    except Exception as e:
        return {"status": "error", "message": f"Execution failed: {str(e)}"}
