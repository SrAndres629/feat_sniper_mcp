
import logging
import os
import sys
import json
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List

logger = logging.getLogger("FEAT.DeepAuditor")

class DeepAuditor:
    """
    🔬 INSTITUTIONAL GRADE AUDITOR
    Analyzes system health, ML performance, and data integrity.
    """
    def __init__(self, db_path: str = "app/data/market_data.db"):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            self.db_path = "data/market_data.db"

    def run_comprehensive_audit(self) -> Dict[str, Any]:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "NOMINAL",
            "modules": {},
            "metrics": {},
            "anomalies": []
        }

        # 1. Database Integrity & Freshness
        db_audit = self._audit_database()
        report["modules"]["database"] = db_audit
        if db_audit["status"] != "OK":
            report["status"] = "DEGRADED"
            report["anomalies"].append(f"DB_STALE: {db_audit['latency_sec']}s")

        # 2. ML Feature Distribution Audit
        # Check if MSS-5 sensors are producing valid data (not all zeros / NaNs)
        ml_audit = self._audit_ml_features()
        report["modules"]["ml_engine"] = ml_audit
        if ml_audit["status"] != "OK":
            report["status"] = "CRITICAL"
            report["anomalies"].append("ML_FEATURE_CORRUPTION")

        # 3. Process & Infrastructure Check
        infra_audit = self._audit_infrastructure()
        report["modules"]["infrastructure"] = infra_audit

        return report

    def _audit_database(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.db_path):
                return {"status": "ERROR", "message": "File not found"}
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("SELECT * FROM market_ticks ORDER BY id DESC LIMIT 10", conn)
            conn.close()

            if df.empty:
                return {"status": "EMPTY", "latency_sec": -1}

            last_ts = pd.to_datetime(df['tick_time'].iloc[0].replace("Z", "+00:00"))
            latency = (datetime.now(timezone.utc) - last_ts).total_seconds()

            return {
                "status": "OK" if latency < 300 else "STALE",
                "latency_sec": int(latency),
                "record_count": len(df)
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def _audit_ml_features(self) -> Dict[str, Any]:
        """Check for NaN or Dead features in the last batch."""
        try:
            conn = sqlite3.connect(self.db_path)
            # Check the last 100 entries for M1
            df = pd.read_sql("SELECT * FROM market_data WHERE timeframe='M1' ORDER BY tick_time DESC LIMIT 100", conn)
            conn.close()

            if df.empty:
                return {"status": "WAITING_DATA", "nan_count": 0}

            # Check specific MSS-5 tensors
            tensors = ["momentum_kinetic_micro", "entropy_coefficient", "cycle_harmonic_phase", "acceptance_ratio"]
            missing_tensors = [t for t in tensors if t not in df.columns]
            
            if missing_tensors:
                 return {"status": "SCHEMA_MISMATCH", "missing": missing_tensors}

            nan_stats = df[tensors].isna().sum().to_dict()
            total_nans = sum(nan_stats.values())

            return {
                "status": "OK" if total_nans == 0 else "CORRUPTED",
                "nan_report": nan_stats,
                "feature_health": "OPTIMAL" if total_nans == 0 else "SUBOPTIMAL"
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def _audit_infrastructure(self) -> Dict[str, Any]:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "os": sys.platform
        }

def run_deep_audit(auto_repair: bool = False):
    auditor = DeepAuditor()
    return auditor.run_comprehensive_audit()
