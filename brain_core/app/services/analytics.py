import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import MetaTrader5 as mt5
from app.core.mt5_conn import mt5_conn

logger = logging.getLogger("MT5_Bridge.Services.Analytics")

class AnalyticsEngine:
    """
    Institutional Analytics Engine for Strategy Audition.
    """

    @staticmethod
    async def get_performance_metrics(history_deals: List[Any]) -> Dict[str, Any]:
        """
        Calculates Sharpe Ratio, Win Rate, and Profit Factor from raw history.
        """
        if not history_deals:
            return {"status": "no_data", "sharpe": 0, "win_rate": 0, "profit_factor": 0}

        df = pd.DataFrame(history_deals)
        # Filters deals that are actual profit/loss
        profits = df[df['entry'] != mt5.DEAL_ENTRY_IN]['profit'].values
        
        if len(profits) == 0:
            return {"status": "no_closed_trades", "sharpe": 0}

        # Win Rate
        wins = profits[profits > 0]
        losses = profits[profits < 0]
        win_rate = (len(wins) / len(profits)) * 100
        
        # Profit Factor
        total_won = np.sum(wins)
        total_lost = abs(np.sum(losses))
        profit_factor = total_won / total_lost if total_lost > 0 else total_won

        # Sharpe Ratio (Simulated daily returns)
        avg_ret = np.mean(profits)
        std_ret = np.std(profits)
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

        return {
            "status": "success",
            "trades_total": len(profits),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_net_profit": round(float(np.sum(profits)), 2)
        }

    @staticmethod
    async def run_monte_carlo(current_results: List[float], iterations: int = 1000) -> Dict[str, Any]:
        """
        Simulates strategy outcomes by shuffling returns.
        """
        if len(current_results) < 5:
            return {"error": "Not enough data for Monte Carlo"}

        results = []
        for _ in range(iterations):
            sim = np.random.choice(current_results, size=len(current_results), replace=True)
            results.append(np.sum(sim))
        
        return {
            "p50_expected_profit": round(float(np.percentile(results, 50)), 2),
            "p95_var": round(float(np.percentile(results, 5)), 2), # Value at Risk
            "max_simulated_loss": round(float(np.min(results)), 2)
        }

analytics_engine = AnalyticsEngine()
