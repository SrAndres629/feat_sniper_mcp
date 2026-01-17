"""
SYSTEM GUARD - Unified Guardian Module (Vibranium Grade)
=========================================================
Consolidated integrity, validation, and health monitoring for the FEAT Sniper system.

[PROJECT ATLAS] Consolidation of:
- integrity_check.py (ModelGuardian)
- validation.py (OrderValidator)
- exceptions.py (Custom Exceptions)
+ NEW: ResourcePredictor (RAM Slope Detection)

Features:
- Model artifact validation (PyTorch/Joblib deep check)
- Order pre-validation (margin, stops, risk)
- RAM slope prediction (OOM prevention)
- Unified exception hierarchy
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Deque, Any
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger("feat.core.system_guard")


# =============================================================================
# UNIFIED EXCEPTION HIERARCHY
# =============================================================================

class SystemGuardError(Exception):
    """Base exception for all system guard errors."""
    ...


class ArtifactIntegrityError(SystemGuardError):
    """
    Raised when a model artifact fails validation.
    
    Attributes:
        path: Path to the failed file
        error_type: Type of error (NOT_FOUND, EMPTY, CORRUPTED, VERSION_MISMATCH)
        details: Diagnostic message
    """
    def __init__(self, path: str, error_type: str, details: str):
        self.path = path
        self.error_type = error_type
        self.details = details
        super().__init__(f"[{error_type}] {path}: {details}")


class RiskViolationError(SystemGuardError):
    """
    Raised when an operation violates a hard risk rule.
    Must immediately abort the trading sequence.
    """
    ...


class CircuitBreakerTrip(SystemGuardError):
    """
    Raised when the system detects a systemic anomaly (Latency/Disconnect).
    """
    ...


class ResourceExhaustionError(SystemGuardError):
    """
    Raised when the system predicts or detects resource exhaustion (OOM).
    """
    ...


# =============================================================================
# ARTIFACT TYPES
# =============================================================================

class ArtifactType(Enum):
    """Supported artifact types."""
    PYTORCH = "pytorch"
    JOBLIB = "joblib"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Result of artifact validation."""
    path: str
    artifact_type: ArtifactType
    is_valid: bool
    file_size_bytes: int = 0
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# MODEL GUARDIAN - Artifact Integrity Validation
# =============================================================================

class ModelGuardian:
    """
    Guardian of ML Artifact Integrity.
    
    Singleton pattern ensures consistent validation across the system.
    
    Why integrity validation vs simple path validation?
    ---------------------------------------------------
    Path validation only confirms file existence. However:
    1. File may be empty (0 bytes) - interrupted write
    2. File may be corrupted - partial download, disk failure
    3. File may be incompatible - different PyTorch/Joblib version
    
    Integrity validation LOADS the file to confirm usability,
    detecting issues BEFORE the system attempts inference during
    a high-volatility tick.
    """
    
    _instance: Optional['ModelGuardian'] = None
    
    def __new__(cls) -> 'ModelGuardian':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._validation_cache: Dict[str, ValidationResult] = {}
        logger.info("[GUARDIAN] ModelGuardian initialized - Artifact Integrity Protection ACTIVE")
    
    def _detect_type(self, path: str) -> ArtifactType:
        """Detect artifact type by extension."""
        ext = Path(path).suffix.lower()
        if ext in (".pt", ".pth"):
            return ArtifactType.PYTORCH
        elif ext in (".joblib", ".pkl"):
            return ArtifactType.JOBLIB
        return ArtifactType.UNKNOWN

    def calculate_checksum(self, path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def validate_existence(self, path: str) -> Tuple[bool, Optional[str]]:
        """Existence validation: file exists and is not empty."""
        if not os.path.exists(path):
            return False, f"File not found: {path}"
        
        file_size = os.path.getsize(path)
        if file_size == 0:
            return False, f"Empty file (0 bytes): {path}"
        
        return True, None
    
    def validate_pytorch(self, path: str, weights_only: bool = True) -> ValidationResult:
        """Deep Check for PyTorch files (.pt, .pth)."""
        result = ValidationResult(
            path=path,
            artifact_type=ArtifactType.PYTORCH,
            is_valid=False
        )
        
        exists, error = self.validate_existence(path)
        if not exists:
            result.error_type = "NOT_FOUND" if "not found" in error else "EMPTY"
            result.error_details = error
            return result
        
        result.file_size_bytes = os.path.getsize(path)
        
        try:
            import torch
            
            try:
                data = torch.load(path, map_location="cpu", weights_only=weights_only)
            except TypeError:
                data = torch.load(path, map_location="cpu")
            
            if isinstance(data, dict):
                result.metadata["keys"] = list(data.keys())[:10]
                if "model_config" in data:
                    result.metadata["config"] = data["model_config"]
            
            result.is_valid = True
            result.metadata["sha256"] = self.calculate_checksum(path)
            logger.debug(f"[GUARDIAN] ✅ PyTorch artifact valid: {path}")
            
        except Exception as e:
            result.error_type = "CORRUPTED"
            result.error_details = f"Failed to load PyTorch file: {str(e)}"
            logger.error(f"[GUARDIAN] ❌ PyTorch artifact corrupted: {path} - {e}")
        
        return result
    
    def validate_joblib(self, path: str) -> ValidationResult:
        """Deep Check for Joblib files (.joblib, .pkl)."""
        result = ValidationResult(
            path=path,
            artifact_type=ArtifactType.JOBLIB,
            is_valid=False
        )
        
        exists, error = self.validate_existence(path)
        if not exists:
            result.error_type = "NOT_FOUND" if "not found" in error else "EMPTY"
            result.error_details = error
            return result
        
        result.file_size_bytes = os.path.getsize(path)
        
        try:
            import joblib
            
            data = joblib.load(path)
            result.metadata["type"] = type(data).__name__
            
            if hasattr(data, "get_params"):
                result.metadata["sklearn_params"] = True
            if isinstance(data, dict):
                result.metadata["keys"] = list(data.keys())[:10]
            
            result.is_valid = True
            result.metadata["sha256"] = self.calculate_checksum(path)
            logger.debug(f"[GUARDIAN] ✅ Joblib artifact valid: {path}")
            
        except ModuleNotFoundError as e:
            result.error_type = "VERSION_MISMATCH"
            result.error_details = f"Missing module during deserialization: {str(e)}"
            logger.error(f"[GUARDIAN] ❌ Joblib version mismatch: {path} - {e}")
            
        except Exception as e:
            result.error_type = "CORRUPTED"
            result.error_details = f"Failed to load Joblib file: {str(e)}"
            logger.error(f"[GUARDIAN] ❌ Joblib artifact corrupted: {path} - {e}")
        
        return result
    
    def validate(self, path: str, use_cache: bool = True) -> ValidationResult:
        """Validate an artifact, auto-detecting its type."""
        if use_cache and path in self._validation_cache:
            return self._validation_cache[path]
        
        artifact_type = self._detect_type(path)
        
        if artifact_type == ArtifactType.PYTORCH:
            result = self.validate_pytorch(path)
        elif artifact_type == ArtifactType.JOBLIB:
            result = self.validate_joblib(path)
        else:
            result = ValidationResult(
                path=path,
                artifact_type=ArtifactType.UNKNOWN,
                is_valid=False,
                error_type="UNKNOWN_TYPE",
                error_details=f"Unsupported artifact type: {Path(path).suffix}"
            )
        
        self._validation_cache[path] = result
        return result
    
    def validate_all(self, paths: List[str], raise_on_failure: bool = True) -> Dict[str, ValidationResult]:
        """Validate multiple artifacts."""
        results = {}
        
        for path in paths:
            result = self.validate(path)
            results[path] = result
            
            if not result.is_valid and raise_on_failure:
                raise ArtifactIntegrityError(
                    path=result.path,
                    error_type=result.error_type or "UNKNOWN",
                    details=result.error_details or "Validation failed"
                )
        
        return results
    
    def get_status(self) -> Dict:
        """Return guardian status for diagnostics."""
        valid_count = sum(1 for r in self._validation_cache.values() if r.is_valid)
        invalid_count = sum(1 for r in self._validation_cache.values() if not r.is_valid)
        
        return {
            "cached_validations": len(self._validation_cache),
            "valid_artifacts": valid_count,
            "invalid_artifacts": invalid_count,
            "all_valid": invalid_count == 0 if self._validation_cache else None
        }
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()
        logger.info("[GUARDIAN] Validation cache cleared")


# =============================================================================
# RESOURCE PREDICTOR - RAM Slope Detection (NEW)
# =============================================================================

class ResourcePredictor:
    """
    Predictive resource monitor using linear regression on RAM usage.
    
    Detects memory leaks and predicts OOM conditions BEFORE they happen,
    allowing graceful degradation or restart.
    
    Algorithm:
    1. Sample RAM usage at regular intervals (e.g., every 1s)
    2. Maintain rolling window of samples (e.g., last 60 samples = 1 min)
    3. Calculate linear regression slope
    4. If slope > threshold, predict OOM and alert
    """
    
    def __init__(self, window_size: int = 60, sample_interval_sec: float = 1.0):
        self.window_size = window_size
        self.sample_interval = sample_interval_sec
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self._last_sample_time: float = 0.0
        logger.info(f"[PREDICTOR] ResourcePredictor initialized (window={window_size}s)")
    
    def sample(self) -> Optional[float]:
        """
        Take a RAM sample if enough time has passed.
        
        Returns:
            Current RAM usage in MB, or None if skipped
        """
        now = time.time()
        if now - self._last_sample_time < self.sample_interval:
            return None
        
        self._last_sample_time = now
        
        try:
            import psutil
            process = psutil.Process()
            ram_mb = process.memory_info().rss / (1024 * 1024)
            self._samples.append((now, ram_mb))
            return ram_mb
        except ImportError:
            logger.warning("[PREDICTOR] psutil not available - ResourcePredictor disabled")
            return None
        except Exception as e:
            logger.error(f"[PREDICTOR] Failed to sample RAM: {e}")
            return None
    
    def get_slope(self) -> Optional[float]:
        """
        Calculate RAM usage slope (MB/second).
        
        Positive slope = memory increasing (potential leak)
        Negative slope = memory decreasing (normal GC)
        
        Returns:
            Slope in MB/s, or None if insufficient samples
        """
        if len(self._samples) < 10:
            return None
        
        # Simple linear regression: y = mx + b
        # where x = time, y = RAM usage
        n = len(self._samples)
        sum_x = sum(s[0] for s in self._samples)
        sum_y = sum(s[1] for s in self._samples)
        sum_xy = sum(s[0] * s[1] for s in self._samples)
        sum_xx = sum(s[0] ** 2 for s in self._samples)
        
        denominator = n * sum_xx - sum_x ** 2
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def predict_oom(self, threshold_mb: float = 1000.0, critical_slope_mb_s: float = 1.0) -> Dict:
        """
        Predict if OOM is imminent based on current RAM and slope.
        
        Args:
            threshold_mb: RAM threshold to consider dangerous
            critical_slope_mb_s: Slope threshold to consider a memory leak
            
        Returns:
            Dict with prediction status and details
        """
        if not self._samples:
            return {"status": "UNKNOWN", "message": "No samples yet"}
        
        current_ram = self._samples[-1][1]
        slope = self.get_slope()
        
        result = {
            "status": "OK",
            "current_ram_mb": round(current_ram, 2),
            "slope_mb_s": round(slope, 6) if slope is not None else None,
            "samples": len(self._samples)
        }
        
        if slope is None:
            result["status"] = "INSUFFICIENT_DATA"
            return result
        
        # Check for dangerous conditions
        if current_ram > threshold_mb:
            result["status"] = "CRITICAL"
            result["message"] = f"RAM usage {current_ram:.0f}MB exceeds threshold {threshold_mb:.0f}MB"
        elif slope > critical_slope_mb_s:
            # Extrapolate time to OOM
            remaining_mb = threshold_mb - current_ram
            time_to_oom_s = remaining_mb / slope if slope > 0 else float('inf')
            result["status"] = "WARNING"
            result["message"] = f"Memory leak detected. Slope={slope:.3f}MB/s. OOM in ~{time_to_oom_s:.0f}s"
            result["time_to_oom_s"] = round(time_to_oom_s, 1)
        
        return result
    
    def get_status(self) -> Dict:
        """Return predictor status for diagnostics."""
        return {
            "samples_collected": len(self._samples),
            "window_size": self.window_size,
            "current_slope_mb_s": self.get_slope()
        }


# =============================================================================
# ORDER VALIDATOR - Pre-Trade Validation
# =============================================================================

class OrderValidator:
    """
    Intelligent validation engine for trading orders and risk management.
    """
    
    @staticmethod
    async def validate_order(
        symbol: str, 
        volume: float, 
        action: str, 
        price: Optional[float] = None, 
        sl: Optional[float] = None, 
        tp: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that an order meets broker requirements and risk rules."""
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.warning("[VALIDATOR] MT5 not available - skipping order validation")
            return True, None
        
        from app.core.mt5_conn import mt5_conn
        from app.core.config import settings
        
        # 1. Get symbol info
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        if not symbol_info:
            return False, f"Symbol {symbol} not found."

        if not symbol_info.visible:
            await mt5_conn.execute(mt5.symbol_select, symbol, True)

        # 2. Validate volume
        if volume < symbol_info.volume_min:
            return False, f"Volume {volume} below minimum ({symbol_info.volume_min})."
        
        if volume > symbol_info.volume_max:
            return False, f"Volume {volume} above maximum ({symbol_info.volume_max})."
        
        # Validate step
        steps = round(volume / symbol_info.volume_step)
        expected_vol = steps * symbol_info.volume_step
        
        if abs(volume - expected_vol) > 0.00001:
            return False, f"Volume {volume} doesn't match step ({symbol_info.volume_step})."

        # 3. Validate stops
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        if not tick:
            return False, "Could not get price for stop validation."

        ref_price = price if price else (tick.ask if "BUY" in action else tick.bid)
        
        stop_level_points = symbol_info.stops_level
        point_size = symbol_info.point
        min_distance = stop_level_points * point_size

        if sl:
            distance_sl = abs(ref_price - sl)
            if distance_sl < min_distance:
                return False, f"Stop Loss too close. Min distance: {stop_level_points} points."

        if tp:
            distance_tp = abs(ref_price - tp)
            if distance_tp < min_distance:
                return False, f"Take Profit too close. Min distance: {stop_level_points} points."

        # 4. Risk validation (Drawdown)
        account_info = await mt5_conn.execute(mt5.account_info)
        if not account_info:
            return False, "Could not get account info for risk validation."

        daily_loss_percent = ((account_info.balance - account_info.equity) / account_info.balance) * 100 if account_info.balance > 0 else 0
        if daily_loss_percent > settings.MAX_DAILY_DRAWDOWN_PERCENT:
            return False, f"Daily drawdown limit exceeded ({daily_loss_percent:.2f}% > {settings.MAX_DAILY_DRAWDOWN_PERCENT}%)."

        # 5. Position limits
        positions_total = await mt5_conn.execute(mt5.positions_total)
        if positions_total >= settings.MAX_OPEN_POSITIONS:
            return False, f"Position limit reached ({positions_total}/{settings.MAX_OPEN_POSITIONS})."

        return True, None

    @staticmethod
    async def validate_margin(symbol: str, volume: float, action: str) -> Tuple[bool, Optional[str]]:
        """Verify sufficient margin before sending order."""
        try:
            import MetaTrader5 as mt5
        except ImportError:
            return True, None
        
        from app.core.mt5_conn import mt5_conn
        
        order_type = mt5.ORDER_TYPE_BUY if "BUY" in action else mt5.ORDER_TYPE_SELL
        margin_required = await mt5_conn.execute(mt5.order_calc_margin, order_type, symbol, volume, 0.0)
        
        account_info = await mt5_conn.execute(mt5.account_info)
        if account_info.margin_free < margin_required:
            return False, f"Insufficient margin. Required: {margin_required}, Available: {account_info.margin_free}"
        
        return True, None


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

model_guardian = ModelGuardian()
resource_predictor = ResourcePredictor()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_model_artifacts(models_dir: str = "models") -> bool:
    """
    Convenience function to validate all artifacts in a directory.
    
    Returns:
        True if all artifacts are valid
    
    Raises:
        ArtifactIntegrityError if any artifact fails
    """
    if not os.path.exists(models_dir):
        logger.warning(f"[GUARDIAN] Models directory not found: {models_dir}")
        return True
    
    artifact_paths = []
    for ext in ["*.pt", "*.pth", "*.joblib", "*.pkl"]:
        artifact_paths.extend(Path(models_dir).glob(ext))
    
    if not artifact_paths:
        logger.info(f"[GUARDIAN] No artifacts found in {models_dir}")
        return True
    
    logger.info(f"[GUARDIAN] Validating {len(artifact_paths)} artifacts...")
    
    results = model_guardian.validate_all(
        [str(p) for p in artifact_paths],
        raise_on_failure=True
    )
    
    valid_count = sum(1 for r in results.values() if r.is_valid)
    logger.info(f"[GUARDIAN] ✅ All {valid_count} artifacts validated successfully")
    
    return True

# =============================================================================
# SYSTEM SENTINEL - The Immune System Core
# =============================================================================

class SystemSentinel:
    """
    [LEVEL 44] Central Defense Hub.
    Manages DEFCON state, Kill Switch, and Health Heartbeat.
    """
    def __init__(self):
        self.kill_switch_active = False
        self.kill_reason = None
        self.last_heartbeat = time.time()
        self.defcon = 5 # 5=Normal, 1=Nuclear
        
    def trigger_kill_switch(self, reason: str):
        """IMMEDIATE SYSTEM SHUTDOWN PROTOCOL."""
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.kill_reason = reason
            self.defcon = 1
            logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
            # Potential: Async notification to admin (Telegram/Discord)
            
    def reset_kill_switch(self, auth_token: str = "ADMIN_RESET"):
        """Manual reset of the defense system."""
        self.kill_switch_active = False
        self.kill_reason = None
        self.defcon = 5
        logger.info("🟢 Kill Switch RESET. System Nominal.")
        
    def is_safe(self) -> bool:
        """Master Gate for Trading Execution."""
        return not self.kill_switch_active

    def check_health(self) -> Dict[str, Any]:
        """Real-time Health Aggregation."""
        ram_status = resource_predictor.predict_oom()
        
        # Auto-trigger if RAM is critical
        if ram_status["status"] == "CRITICAL" and not self.kill_switch_active:
            self.trigger_kill_switch(f"RAM CRITICAL: {ram_status.get('message')}")
            
        return {
            "safe": self.is_safe(),
            "kill_switch": self.kill_switch_active,
            "reason": self.kill_reason,
            "ram": ram_status,
            "defcon": self.defcon
        }

# Singleton
system_sentinel = SystemSentinel()
