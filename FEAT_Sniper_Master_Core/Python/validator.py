"""
validator.py - Validation Module for Unified Model
Tests for data integrity, coherence, and determinism.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from stats_engine import StatsEngine, load_csv_data, validate_data
from brute_force import ThresholdSet, StateSimulator
import json


@dataclass
class ValidationResult:
    """Result from a validation test."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


class Validator:
    """
    Comprehensive validation suite for Unified Model.
    Tests data integrity, state coherence, and determinism.
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def add_result(self, name: str, passed: bool, message: str, 
                   details: Optional[Dict] = None) -> None:
        """Add validation result."""
        self.results.append(ValidationResult(name, passed, message, details))
    
    def clear(self) -> None:
        """Clear all results."""
        self.results = []
    
    # ==================== DATA VALIDATION ====================
    
    def validate_no_nan(self, data: np.ndarray, name: str) -> bool:
        """Check for NaN values."""
        has_nan = np.any(np.isnan(data))
        self.add_result(
            f"no_nan_{name}",
            not has_nan,
            f"No NaN values in {name}" if not has_nan else f"Found NaN in {name}"
        )
        return not has_nan
    
    def validate_no_inf(self, data: np.ndarray, name: str) -> bool:
        """Check for infinite values."""
        has_inf = np.any(np.isinf(data))
        self.add_result(
            f"no_inf_{name}",
            not has_inf,
            f"No Inf values in {name}" if not has_inf else f"Found Inf in {name}"
        )
        return not has_inf
    
    def validate_positive(self, data: np.ndarray, name: str) -> bool:
        """Check all values are positive."""
        all_positive = np.all(data >= 0)
        self.add_result(
            f"positive_{name}",
            all_positive,
            f"All {name} values positive" if all_positive else f"Negative values in {name}"
        )
        return all_positive
    
    def validate_sufficient_data(self, data: np.ndarray, name: str, 
                                  min_count: int = 100) -> bool:
        """Check sufficient data points."""
        sufficient = len(data) >= min_count
        self.add_result(
            f"sufficient_{name}",
            sufficient,
            f"{name} has {len(data)} points (min {min_count})",
            {'count': len(data), 'min_required': min_count}
        )
        return sufficient
    
    def validate_variance(self, data: np.ndarray, name: str, 
                          min_std: float = 1e-10) -> bool:
        """Check data has sufficient variance."""
        std = np.std(data)
        has_variance = std > min_std
        self.add_result(
            f"variance_{name}",
            has_variance,
            f"{name} std={std:.6f}" if has_variance else f"{name} has no variance",
            {'std': std, 'min_required': min_std}
        )
        return has_variance
    
    def validate_data_integrity(self, effort: np.ndarray, 
                                result: np.ndarray) -> bool:
        """Run all data integrity checks."""
        checks = [
            self.validate_no_nan(effort, 'effort'),
            self.validate_no_nan(result, 'result'),
            self.validate_no_inf(effort, 'effort'),
            self.validate_no_inf(result, 'result'),
            self.validate_positive(effort, 'effort'),
            self.validate_positive(result, 'result'),
            self.validate_sufficient_data(effort, 'effort'),
            self.validate_sufficient_data(result, 'result'),
            self.validate_variance(effort, 'effort'),
            self.validate_variance(result, 'result')
        ]
        return all(checks)
    
    # ==================== EMA VALIDATION ====================
    
    def validate_ema_order(self, ema_values: List[float], 
                           is_bullish: bool) -> bool:
        """
        Validate EMA ordering for trend.
        Bullish: Fast > Medium > Slow
        Bearish: Fast < Medium < Slow
        """
        if len(ema_values) < 3:
            self.add_result("ema_order", False, "Insufficient EMAs for order check")
            return False
        
        ordered = True
        for i in range(len(ema_values) - 1):
            if is_bullish:
                if ema_values[i] <= ema_values[i + 1]:
                    ordered = False
                    break
            else:
                if ema_values[i] >= ema_values[i + 1]:
                    ordered = False
                    break
        
        self.add_result(
            "ema_order",
            ordered,
            f"EMAs {'correctly' if ordered else 'not'} ordered for {'bullish' if is_bullish else 'bearish'} trend"
        )
        return ordered
    
    def validate_compression_range(self, compression: float) -> bool:
        """Validate compression is in expected range [0, 1]."""
        valid = 0 <= compression <= 1
        self.add_result(
            "compression_range",
            valid,
            f"Compression {compression:.3f} in valid range" if valid else f"Compression {compression:.3f} out of range"
        )
        return valid
    
    # ==================== FSM VALIDATION ====================
    
    def validate_state_determinism(self, thresholds: ThresholdSet,
                                   effort: np.ndarray, result: np.ndarray,
                                   compression: np.ndarray, slope: np.ndarray,
                                   speed: np.ndarray) -> bool:
        """
        Verify FSM produces identical results on repeated runs.
        Critical for no-repaint guarantee.
        """
        effort_pcts = np.array([np.sum(effort < e) / len(effort) for e in effort])
        result_pcts = np.array([np.sum(result < r) / len(result) for r in result])
        
        # Run twice
        states1 = []
        states2 = []
        
        sim1 = StateSimulator(thresholds)
        sim2 = StateSimulator(thresholds)
        
        for i in range(len(effort)):
            s1, _ = sim1.update(effort_pcts[i], result_pcts[i], 
                                compression[i], slope[i], speed[i], i)
            s2, _ = sim2.update(effort_pcts[i], result_pcts[i],
                                compression[i], slope[i], speed[i], i)
            states1.append(s1)
            states2.append(s2)
        
        identical = states1 == states2
        self.add_result(
            "fsm_determinism",
            identical,
            "FSM produces identical states on repeated runs" if identical else "FSM is non-deterministic!"
        )
        return identical
    
    def validate_state_coverage(self, thresholds: ThresholdSet,
                                effort: np.ndarray, result: np.ndarray,
                                compression: np.ndarray, slope: np.ndarray,
                                speed: np.ndarray,
                                min_states: int = 2) -> bool:
        """Verify FSM visits multiple states (not stuck)."""
        effort_pcts = np.array([np.sum(effort < e) / len(effort) for e in effort])
        result_pcts = np.array([np.sum(result < r) / len(result) for r in result])
        
        sim = StateSimulator(thresholds)
        states = set()
        
        for i in range(len(effort)):
            state, _ = sim.update(effort_pcts[i], result_pcts[i],
                                  compression[i], slope[i], speed[i], i)
            states.add(state)
        
        has_coverage = len(states) >= min_states
        self.add_result(
            "state_coverage",
            has_coverage,
            f"FSM visited {len(states)} states (min {min_states})",
            {'states_visited': list(states)}
        )
        return has_coverage
    
    def validate_no_rapid_flipflop(self, thresholds: ThresholdSet,
                                    effort: np.ndarray, result: np.ndarray,
                                    compression: np.ndarray, slope: np.ndarray,
                                    speed: np.ndarray,
                                    max_rate: float = 0.1) -> bool:
        """Verify FSM doesn't flip-flop rapidly (instability check)."""
        effort_pcts = np.array([np.sum(effort < e) / len(effort) for e in effort])
        result_pcts = np.array([np.sum(result < r) / len(result) for r in result])
        
        sim = StateSimulator(thresholds)
        
        for i in range(len(effort)):
            sim.update(effort_pcts[i], result_pcts[i],
                      compression[i], slope[i], speed[i], i)
        
        metrics = sim.get_metrics()
        transition_rate = metrics['transitions'] / len(effort)
        stable = transition_rate <= max_rate
        
        self.add_result(
            "no_flipflop",
            stable,
            f"Transition rate {transition_rate:.3f} (max {max_rate})",
            {'transitions': metrics['transitions'], 'rate': transition_rate}
        )
        return stable
    
    # ==================== CALIBRATION VALIDATION ====================
    
    def validate_calibration_file(self, filepath: str) -> bool:
        """Validate calibration file format and content."""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            required_keys = ['effortP80', 'resultP80', 'accumulationCompression',
                            'expansionSlope', 'hysteresisMargin']
            found_keys = set()
            
            for line in lines:
                if '=' in line:
                    key = line.split('=')[0].strip()
                    found_keys.add(key)
            
            missing = set(required_keys) - found_keys
            valid = len(missing) == 0
            
            self.add_result(
                "calibration_format",
                valid,
                f"Calibration file valid" if valid else f"Missing keys: {missing}",
                {'found_keys': list(found_keys), 'missing': list(missing)}
            )
            return valid
            
        except Exception as e:
            self.add_result(
                "calibration_format",
                False,
                f"Failed to read calibration: {e}"
            )
            return False
    
    # ==================== SUMMARY ====================
    
    def run_full_validation(self, effort: np.ndarray, result: np.ndarray,
                           compression: np.ndarray, slope: np.ndarray,
                           speed: np.ndarray,
                           thresholds: ThresholdSet) -> Dict:
        """Run complete validation suite."""
        self.clear()
        
        # Data integrity
        self.validate_data_integrity(effort, result)
        
        # FSM validation
        self.validate_state_determinism(thresholds, effort, result, 
                                        compression, slope, speed)
        self.validate_state_coverage(thresholds, effort, result,
                                     compression, slope, speed)
        self.validate_no_rapid_flipflop(thresholds, effort, result,
                                        compression, slope, speed)
        
        return self.get_summary()
    
    def get_summary(self) -> Dict:
        """Get validation summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        return {
            'total': len(self.results),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(self.results) if self.results else 0,
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.results
            ]
        }
    
    def print_report(self) -> None:
        """Print validation report."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        print(f"Total: {summary['total']} | Passed: {summary['passed']} | Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']*100:.1f}%")
        print("-"*60)
        
        for r in self.results:
            status = "✓" if r.passed else "✗"
            print(f"  [{status}] {r.name}: {r.message}")
        
        print("="*60)


def validate_cross_asset(data_files: List[str], 
                         thresholds: ThresholdSet) -> Dict[str, Dict]:
    """
    Validate thresholds work across multiple assets.
    
    Args:
        data_files: List of CSV file paths
        thresholds: Threshold configuration to test
    
    Returns:
        Dictionary mapping filename to validation summary
    """
    results = {}
    validator = Validator()
    
    for filepath in data_files:
        print(f"\n[CrossAsset] Validating {filepath}")
        
        try:
            effort, result = load_csv_data(filepath)
            
            n = len(effort)
            compression = np.random.uniform(0.3, 0.9, n)
            slope = np.random.normal(0, 0.3, n)
            speed = np.diff(result, prepend=result[0])
            
            summary = validator.run_full_validation(
                effort, result, compression, slope, speed, thresholds
            )
            results[filepath] = summary
            
        except Exception as e:
            results[filepath] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    print("Validator - Unified Model")
    
    # Demo with synthetic data
    np.random.seed(42)
    n = 500
    
    effort = np.random.lognormal(0, 0.5, n)
    result = np.random.exponential(0.5, n)
    compression = np.random.uniform(0.3, 0.9, n)
    slope = np.random.normal(0, 0.3, n)
    speed = np.diff(result, prepend=result[0])
    
    thresholds = ThresholdSet(
        accumulation_compression=0.7,
        expansion_slope=0.3,
        distribution_momentum=-0.2,
        reset_speed=2.0,
        hysteresis_margin=0.1,
        min_bars_in_state=3
    )
    
    validator = Validator()
    summary = validator.run_full_validation(
        effort, result, compression, slope, speed, thresholds
    )
    
    validator.print_report()
