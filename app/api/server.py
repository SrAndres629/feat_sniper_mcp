"""
FEAT SNIPER: FastAPI Server
============================
Central API server for all system operations.
Dashboard and external tools communicate through this interface.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    SimulationRequest, SimulationStatus, SystemStatus, 
    PerformanceReport, CommandResponse, RiskProfileRequest, LogEntry
)
from .workers import worker_manager

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [API] | %(levelname)s | %(message)s'
)
logger = logging.getLogger("API.Server")

# --- CONNECTED WEBSOCKET CLIENTS ---
connected_clients: List[WebSocket] = []


@asynccontextmanager
async def api_lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info("🚀 FEAT SNIPER API Server Starting...")
    yield
    logger.info("👋 API Server Shutting Down...")
    # Cleanup: stop any running simulations
    if worker_manager.simulation.is_running():
        await worker_manager.simulation.stop()


# --- FASTAPI APP ---
app = FastAPI(
    title="FEAT Sniper API",
    description="API-driven control interface for the FEAT Sniper trading system.",
    version="1.0.0",
    lifespan=api_lifespan
)

# CORS Middleware for Dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/api/status", response_model=SystemStatus, tags=["System"])
async def get_system_status():
    """Get overall system health and status."""
    sim_status = worker_manager.simulation.get_status()
    
    # Try to read live state for account info
    account_balance = 0.0
    account_equity = 0.0
    positions_count = 0
    mt5_connected = False
    engine_state = "IDLE"
    
    live_state_path = "data/live_state.json"
    if os.path.exists(live_state_path):
        try:
            with open(live_state_path, 'r') as f:
                live_state = json.load(f)
                account = live_state.get("account", {})
                account_balance = account.get("balance", 0.0)
                account_equity = account.get("equity", 0.0)
                positions_count = live_state.get("positions_count", 0)
                mt5_connected = live_state.get("mt5_connected", False)
                engine_state = live_state.get("engine_state", "IDLE")
        except:
            pass
    
    return SystemStatus(
        daemon_running=True,
        engine_state=engine_state,
        api_version="1.0.0",
        uptime_seconds=worker_manager.get_uptime_seconds(),
        simulation_active=sim_status.get("running", False),
        mt5_connected=mt5_connected,
        positions_count=positions_count,
        account_balance=account_balance,
        account_equity=account_equity
    )


@app.post("/api/emergency/close-all", response_model=CommandResponse, tags=["System"])
async def emergency_close_all():
    """EMERGENCY: Close all open positions immediately."""
    logger.warning("🚨 EMERGENCY CLOSE ALL TRIGGERED via API")
    
    # Write command for daemon to process
    cmd_file = "data/app_commands.json"
    commands = []
    if os.path.exists(cmd_file):
        try:
            with open(cmd_file, 'r') as f:
                commands = json.load(f)
        except:
            pass
    
    commands.append({"action": "PANIC_CLOSE_ALL", "params": {}})
    with open(cmd_file, 'w') as f:
        json.dump(commands, f)
    
    return CommandResponse(
        success=True,
        message="Emergency close command sent. Check system status."
    )


@app.post("/api/risk/update", response_model=CommandResponse, tags=["System"])
async def update_risk_profile(request: RiskProfileRequest):
    """Update the risk factor for trading."""
    logger.info(f"📈 Risk factor update: {request.risk_factor}")
    
    # Write command for daemon
    cmd_file = "data/app_commands.json"
    commands = []
    if os.path.exists(cmd_file):
        try:
            with open(cmd_file, 'r') as f:
                commands = json.load(f)
        except:
            pass
    
    commands.append({"action": "SET_RISK_FACTOR", "params": {"value": request.risk_factor}})
    with open(cmd_file, 'w') as f:
        json.dump(commands, f)
    
    return CommandResponse(
        success=True,
        message=f"Risk factor set to {request.risk_factor}",
        data={"risk_factor": request.risk_factor}
    )


# ============================================================================
# SIMULATION ENDPOINTS
# ============================================================================

@app.post("/api/simulation/start", response_model=CommandResponse, tags=["Simulation"])
async def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """Start a training simulation."""
    if worker_manager.simulation.is_running():
        raise HTTPException(status_code=409, detail="Simulation already running")
    
    success = await worker_manager.simulation.start(
        episodes=request.episodes,
        mode=request.mode.value
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start simulation")
    
    return CommandResponse(
        success=True,
        message=f"Simulation started with {request.episodes} episodes",
        data={"episodes": request.episodes, "mode": request.mode.value}
    )


@app.post("/api/simulation/stop", response_model=CommandResponse, tags=["Simulation"])
async def stop_simulation():
    """Stop a running simulation."""
    if not worker_manager.simulation.is_running():
        raise HTTPException(status_code=404, detail="No simulation running")
    
    success = await worker_manager.simulation.stop()
    
    return CommandResponse(
        success=success,
        message="Simulation stopped" if success else "Failed to stop simulation"
    )


@app.get("/api/simulation/status", response_model=SimulationStatus, tags=["Simulation"])
async def get_simulation_status():
    """Get current simulation status and progress."""
    status = worker_manager.simulation.get_status()
    
    # Calculate elapsed and remaining time
    elapsed = 0.0
    remaining = 0.0
    if status.get("start_time"):
        try:
            start = datetime.fromisoformat(status["start_time"])
            elapsed = (datetime.now() - start).total_seconds()
            
            if status.get("current_episode", 0) > 0:
                avg_per_episode = elapsed / status["current_episode"]
                remaining_episodes = status.get("total_episodes", 0) - status["current_episode"]
                remaining = avg_per_episode * remaining_episodes
        except:
            pass
    
    return SimulationStatus(
        running=status.get("running", False),
        current_episode=status.get("current_episode", 0),
        total_episodes=status.get("total_episodes", 0),
        current_balance=status.get("current_balance", 20.0),
        start_time=status.get("start_time"),
        elapsed_seconds=elapsed,
        estimated_remaining_seconds=remaining,
        last_update=status.get("last_update")
    )


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/analytics/performance", response_model=PerformanceReport, tags=["Analytics"])
async def get_performance_analytics():
    """Get historical performance metrics."""
    journal_path = "data/trade_journal.json"
    
    if not os.path.exists(journal_path):
        return PerformanceReport()  # Return empty report
    
    try:
        with open(journal_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        
        closed = [e for e in entries if e.get("status") == "CLOSED"]
        
        if not closed:
            return PerformanceReport()
        
        # Calculate metrics
        wins = [e for e in closed if e.get("result") == "WIN"]
        losses = [e for e in closed if e.get("result") == "LOSS"]
        
        win_rate = (len(wins) / len(closed)) * 100 if closed else 0
        total_pnl = sum(e.get("pnl_pips", 0) for e in closed)
        avg_win = sum(e.get("pnl_pips", 0) for e in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(e.get("pnl_pips", 0) for e in losses) / len(losses)) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Equity curve
        equity = [20.0]
        for e in closed:
            equity.append(equity[-1] + e.get("pnl_pips", 0) * 0.1)
        
        # Exit reasons
        exit_stats = {}
        for e in closed:
            reason = e.get("exit_reason", "UNKNOWN")
            exit_stats[reason] = exit_stats.get(reason, 0) + 1
        
        return PerformanceReport(
            total_trades=len(closed),
            win_rate=win_rate,
            profit_factor=profit_factor,
        # [DOCTORAL] Calculate Sharpe Ratio correctly using Daily Returns
        # Technical Debt Removal: Do not treat trade series as time series.
        # 1. Aggregate PnL by Day
        daily_pnl = {}
        for e in closed:
            ts = e.get("close_time", "")
            if ts:
                # Extract YYYY-MM-DD
                try:
                    date_str = ts.split("T")[0]
                    daily_pnl[date_str] = daily_pnl.get(date_str, 0.0) + e.get("pnl_pips", 0)
                except:
                    pass
        
        if len(daily_pnl) > 1:
            daily_returns = list(daily_pnl.values())
            mean_daily = sum(daily_returns) / len(daily_returns)
            std_daily = (sum((x - mean_daily) ** 2 for x in daily_returns) / (len(daily_returns) - 1)) ** 0.5
            # Annualize: Mean * 252, Std * sqrt(252) -> Sharpe * sqrt(252)
            sharpe = (mean_daily / std_daily) * (252 ** 0.5) if std_daily > 0 else 0.0
        else:
            sharpe = 0.0

        # Calculate Max Drawdown (Standard Equity High watermark)
        peak = equity[0]
        max_dd = 0.0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd    
        
        return PerformanceReport(
            total_trades=len(closed),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_pnl_pips=total_pnl,
            avg_trade_duration_minutes=sum(e.get("duration_minutes", 0) for e in closed) / len(closed) if closed else 0,
            exit_reasons=exit_stats,
            equity_curve=equity
        )
        
    except Exception as e:
        logger.error(f"Error loading analytics: {e}")
        return PerformanceReport()


# ============================================================================
# MODELS ENDPOINTS
# ============================================================================

@app.post("/api/models/reload", response_model=CommandResponse, tags=["Models"])
async def reload_models():
    """Hot-reload neural network weights."""
    logger.info("🧠 Model reload requested via API")
    
    cmd_file = "data/app_commands.json"
    commands = []
    if os.path.exists(cmd_file):
        try:
            with open(cmd_file, 'r') as f:
                commands = json.load(f)
        except:
            pass
    
    commands.append({"action": "RELOAD_MODELS", "params": {}})
    with open(cmd_file, 'w') as f:
        json.dump(commands, f)
    
    return CommandResponse(
        success=True,
        message="Model reload command sent"
    )


# ============================================================================
# LOGS / STREAMING ENDPOINTS
# ============================================================================

@app.get("/api/logs/recent", tags=["Logs"])
async def get_recent_logs(limit: int = 50):
    """Get recent log entries."""
    logs = worker_manager.simulation.get_logs(max_entries=limit)
    return {"logs": logs}


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming."""
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(connected_clients)}")
    
    try:
        while True:
            # Send any new logs
            logs = worker_manager.simulation.get_logs(max_entries=10)
            if logs:
                await websocket.send_json({"type": "logs", "data": logs})
            
            # Send simulation status periodically
            status = worker_manager.simulation.get_status()
            await websocket.send_json({"type": "status", "data": status})
            
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
