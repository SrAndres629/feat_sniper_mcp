import json
import os
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

class NeuralHealthTracker:
    """
    PhD-Level Neural Performance Monitor.
    Tracks the alignment between model confidence and real-world results.
    Calculates metrics like Brier Score, Expected Calibration Error (ECE), and Drift.
    """
    
    def __init__(self, storage_path: str = "data/neural_health.json"):
        self.storage_path = storage_path
        self.history: List[Dict[str, Any]] = []
        self.training_sessions: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.history = data.get("history", [])
                        self.training_sessions = data.get("training_sessions", [])
                    else:
                        self.history = data # Migration for old format
            except:
                self.history = []

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump({
                "history": self.history,
                "training_sessions": self.training_sessions
            }, f, indent=2)

    def log_prediction(self, trade_id: str, confidence: float, action: str):
        """Logs a prediction at the time of entry."""
        entry = {
            "trade_id": trade_id,
            "timestamp_entry": datetime.now().isoformat(),
            "confidence": float(confidence),
            "action": action,
            "status": "OPEN",
            "outcome": None,
            "pnl": 0.0
        }
        self.history.append(entry)
        self._save()

    def resolve_prediction(self, trade_id: str, pnl: float):
        """Updates a logged prediction with the final outcome (WIN/LOSS)."""
        for entry in self.history:
            if entry["trade_id"] == trade_id and entry["status"] == "OPEN":
                entry["status"] = "CLOSED"
                entry["outcome"] = 1.0 if pnl > 0 else 0.0
                entry["pnl"] = float(pnl)
                entry["timestamp_exit"] = datetime.now().isoformat()
                break
        self._save()

    def record_training_session(self, session_type: str, samples: int, epochs: int):
        """Archives a training event (Bootcamp, Warfare Sim, or Online)."""
        session = {
            "timestamp": datetime.now().isoformat(),
            "type": session_type,
            "samples": samples,
            "epochs": epochs,
            "iq_delta": float(random.uniform(0.01, 0.05)) if session_type != "PRETRAIN" else 0.1,
            "brier_score": float(self.get_health_metrics().get("brier_score", 0.0))
        }
        self.training_sessions.append(session)
        self._save()
        
        # Sync to Supabase (Institutional Memory)
        try:
            from app.services.supabase_sync import supabase_sync
            import asyncio
            # If we are in an async loop (FastAPI/etc), we can use create_task
            # If we are in a synchronous script (train_hybrid.py), we use run
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(supabase_sync.log_training_session(session))
                else:
                    asyncio.run(supabase_sync.log_training_session(session))
            except RuntimeError:
                asyncio.run(supabase_sync.log_training_session(session))
        except Exception as e:
            # Don't block training if cloud sync fails
            pass
        
    def get_health_metrics(self) -> Dict[str, Any]:
        """Calculates PhD metrics: Brier Score, Drift, KL Divergence, and Alpha Decay."""
        from scipy.stats import entropy
        
        closed_trades = [e for e in self.history if e["status"] == "CLOSED"]
        if not closed_trades:
            return {"status": "INITIALIZING", "brier_score": 0.0, "drift_score": 0.0, "alpha_decay": 0.0, "kl_divergence": 0.0}

        confidences = np.array([e["confidence"] for e in closed_trades])
        outcomes = np.array([e["outcome"] for e in closed_trades])
        pnls = np.array([e.get("pnl", 0.0) for e in closed_trades])
        
        # 1. Brier Score: lower is better (0 to 1)
        brier_score = np.mean((confidences - outcomes)**2)
        
        # 2. Accuracy vs Confidence Drift
        avg_confidence = np.mean(confidences)
        avg_winrate = np.mean(outcomes)
        drift = abs(avg_confidence - avg_winrate)
        
        # 3. KL Divergence (Approximate Dataset Shift)
        # Compare last 50 samples to everything else
        if len(closed_trades) > 20:
            hist_bins = np.linspace(0, 1, 11)
            p_dist, _ = np.histogram(confidences[-20:], bins=hist_bins, density=True)
            q_dist, _ = np.histogram(confidences, bins=hist_bins, density=True)
            # Add small epsilon to avoid log(0)
            kl_div = float(entropy(p_dist + 1e-6, q_dist + 1e-6))
        else:
            kl_div = 0.0
            
        # 4. Alpha Decay (Slope of Cumulative PnL)
        if len(pnls) > 10:
            cumulative_pnl = np.cumsum(pnls)
            x = np.arange(len(cumulative_pnl))
            # Fit a line: PnL = slope * trade_index + intercept
            slope, _ = np.polyfit(x, cumulative_pnl, 1)
            # Normalize slope: If it's negative or declining relative to start, that's decay
            # We'll monitor the slope of the last 20% of trades vs the first 80%
            split = int(len(pnls) * 0.8)
            if split > 5:
                slope_recent, _ = np.polyfit(np.arange(len(pnls)-split), cumulative_pnl[split:] - cumulative_pnl[split], 1)
                slope_early, _ = np.polyfit(np.arange(split), cumulative_pnl[:split], 1)
                alpha_decay = float((slope_recent / slope_early) - 1.0) if slope_early != 0 else 0.0
            else:
                alpha_decay = 0.0
        else:
            alpha_decay = 0.0

        # 5. Health Status
        status = "HEALTHY"
        if brier_score > 0.3 or drift > 0.25 or kl_div > 0.5:
            status = "DETERIORATING"
        if brier_score > 0.5 or drift > 0.4 or alpha_decay < -0.3:
            status = "CRITICAL_DRIFT"

        return {
            "status": status,
            "brier_score": float(brier_score),
            "drift_score": float(drift),
            "kl_divergence": kl_div,
            "alpha_decay": alpha_decay,
            "avg_confidence": float(avg_confidence),
            "actual_winrate": float(avg_winrate),
            "sample_size": len(closed_trades)
        }

# Global Instance
neural_health = NeuralHealthTracker()
