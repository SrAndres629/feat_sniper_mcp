"""
ml_engine.py - Machine Learning Engine for FSM Classification
Replaces heuristic rules with supervised learning using scikit-learn.

Enhancement over brute_force.py:
- RandomForest/XGBoost for state classification
- Feature importance analysis
- Cross-validation for robustness
- Export capabilities for MQL5 interop
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


@dataclass
class MLConfig:
    """Configuration for ML models."""
    model_type: str = 'rf'  # 'rf' or 'gb'
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


class FSMClassifier:
    """
    Supervised FSM state classifier.
    Replaces heuristic DetermineState() with ML model.
    
    States:
        0: ACCUMULATION
        1: EXPANSION
        2: DISTRIBUTION
        3: RESET
    """
    
    STATE_NAMES = ['ACCUMULATION', 'EXPANSION', 'DISTRIBUTION', 'RESET']
    FEATURE_NAMES = [
        'effort_pct', 'result_pct', 'compression', 
        'slope', 'speed', 'momentum',
        'effort_ma', 'result_ma'  # Additional features
    ]
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.STATE_NAMES)
        
        if self.config.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=self.config.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.config.model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=min(5, self.config.max_depth),
                learning_rate=0.1,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.is_fitted = False
        self.training_metrics: Dict[str, Any] = {}
    
    @staticmethod
    def prepare_features(effort_pct: np.ndarray,
                         result_pct: np.ndarray,
                         compression: np.ndarray,
                         slope: np.ndarray,
                         speed: np.ndarray,
                         window: int = 5) -> np.ndarray:
        """
        Prepare feature matrix from raw inputs.
        
        Adds derived features:
        - momentum: rate of change of result
        - effort_ma: moving average of effort
        - result_ma: moving average of result
        """
        n = len(effort_pct)
        
        # Compute momentum as gradient
        momentum = np.gradient(result_pct)
        
        # Moving averages
        effort_ma = np.convolve(effort_pct, np.ones(window)/window, mode='same')
        result_ma = np.convolve(result_pct, np.ones(window)/window, mode='same')
        
        X = np.column_stack([
            effort_pct,
            result_pct,
            compression,
            slope,
            speed,
            momentum,
            effort_ma,
            result_ma
        ])
        
        return X
    
    def generate_labels_from_heuristic(self,
                                       effort_pct: np.ndarray,
                                       result_pct: np.ndarray,
                                       compression: np.ndarray,
                                       slope: np.ndarray,
                                       speed: np.ndarray,
                                       thresholds: Optional[Dict] = None) -> np.ndarray:
        """
        Generate training labels using heuristic rules.
        This bootstraps the ML model from existing logic.
        """
        if thresholds is None:
            thresholds = {
                'effort_p80': 0.8,
                'result_p20': 0.2,
                'accumulation_compression': 0.7,
                'expansion_slope': 0.3,
                'distribution_momentum': -0.2,
                'reset_speed': 2.0
            }
        
        n = len(effort_pct)
        labels = np.zeros(n, dtype=int)
        
        for i in range(n):
            # RESET: High speed
            if abs(speed[i]) > thresholds['reset_speed'] and compression[i] > 0.5:
                labels[i] = 3  # RESET
            # ACCUMULATION: High effort, low result, compressed
            elif (effort_pct[i] > thresholds['effort_p80'] and 
                  result_pct[i] < thresholds['result_p20'] and
                  compression[i] > thresholds['accumulation_compression']):
                labels[i] = 0  # ACCUMULATION
            # EXPANSION: High result, active slope
            elif (result_pct[i] > 0.8 and 
                  effort_pct[i] < 0.5 and
                  abs(slope[i]) > thresholds['expansion_slope']):
                labels[i] = 1  # EXPANSION
            # DISTRIBUTION: Effort up, momentum down
            elif (effort_pct[i] > 0.5 and 
                  slope[i] < thresholds['distribution_momentum'] and
                  compression[i] < 0.5):
                labels[i] = 2  # DISTRIBUTION
            else:
                labels[i] = 0  # Default ACCUMULATION
        
        return labels
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - integers 0-3 or string state names
        
        Returns:
            Dictionary with training metrics
        """
        # Handle string labels
        if isinstance(y[0], str):
            y = self.label_encoder.transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Train
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, 
                            random_state=self.config.random_state)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv)
        
        self.training_metrics = {
            'accuracy': float(accuracy_score(y_val, y_pred)),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist(),
            'classification_report': classification_report(
                y_val, y_pred,
                labels=[0, 1, 2, 3],
                target_names=self.STATE_NAMES,
                output_dict=True,
                zero_division=0
            ),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'feature_importances': dict(zip(
                self.FEATURE_NAMES[:X.shape[1]],
                self.model.feature_importances_.tolist()
            ))
        }
        
        return self.training_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state for new samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predicted states as integers
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities for confidence scoring.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability matrix (n_samples, 4)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict state with confidence scores.
        
        Returns:
            (predictions, confidences) tuple
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1) * 100  # Convert to 0-100
        return predictions, confidences
    
    def save(self, filepath: str) -> None:
        """Save model for later use."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'feature_names': self.FEATURE_NAMES
        }, filepath)
        print(f"[MLEngine] Saved model to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load pre-trained model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.config = data.get('config', MLConfig())
        self.training_metrics = data.get('training_metrics', {})
        self.is_fitted = True
        print(f"[MLEngine] Loaded model from {filepath}")
    
    def export_for_mql5(self, filepath: str) -> None:
        """
        Export model insights for MQL5.
        
        Since MQL5 can't run sklearn directly, we export:
        1. Feature importance weights for weighted heuristics
        2. Decision thresholds derived from tree structure
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        importances = self.model.feature_importances_
        
        # Extract approximate thresholds from tree structure
        thresholds = self._extract_thresholds()
        
        output = {
            'model_type': self.config.model_type,
            'accuracy': self.training_metrics.get('accuracy', 0),
            'cv_mean': self.training_metrics.get('cv_mean', 0),
            'feature_weights': dict(zip(
                self.FEATURE_NAMES[:len(importances)],
                importances.tolist()
            )),
            'derived_thresholds': thresholds
        }
        
        # Save as key=value for MQL5
        with open(filepath, 'w') as f:
            f.write("# ML-derived parameters for MQL5\n")
            f.write(f"ml_accuracy={output['accuracy']:.4f}\n")
            f.write(f"cv_mean={output['cv_mean']:.4f}\n")
            
            for feat, weight in output['feature_weights'].items():
                f.write(f"weight_{feat}={weight:.6f}\n")
            
            for key, value in output['derived_thresholds'].items():
                f.write(f"{key}={value:.4f}\n")
        
        # Also save as JSON for Python use
        json_path = filepath.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[MLEngine] Exported to {filepath} and {json_path}")
    
    def _extract_thresholds(self) -> Dict[str, float]:
        """Extract approximate decision thresholds from tree structure."""
        if self.config.model_type != 'rf':
            return {}
        
        # Analyze first few trees to find common split points
        thresholds = {
            'ml_effort_threshold': 0.5,
            'ml_result_threshold': 0.5,
            'ml_compression_threshold': 0.6,
            'ml_slope_threshold': 0.3,
            'ml_speed_threshold': 2.0
        }
        
        # Get feature split values from trees
        for i, tree in enumerate(self.model.estimators_[:10]):
            tree_struct = tree.tree_
            for node_id in range(tree_struct.node_count):
                if tree_struct.feature[node_id] >= 0:  # Not a leaf
                    feat_idx = tree_struct.feature[node_id]
                    threshold = tree_struct.threshold[node_id]
                    
                    # Update thresholds based on common splits
                    if feat_idx == 0:  # effort_pct
                        thresholds['ml_effort_threshold'] = (
                            thresholds['ml_effort_threshold'] + threshold
                        ) / 2
                    elif feat_idx == 1:  # result_pct
                        thresholds['ml_result_threshold'] = (
                            thresholds['ml_result_threshold'] + threshold
                        ) / 2
        
        return thresholds


def train_fsm_classifier(effort: np.ndarray,
                         result: np.ndarray,
                         compression: np.ndarray,
                         slope: np.ndarray,
                         speed: np.ndarray,
                         model_type: str = 'rf') -> FSMClassifier:
    """
    Convenience function to train FSM classifier from raw data.
    
    Uses heuristic labels for bootstrapping.
    """
    # Prepare features
    n = len(effort)
    effort_pcts = np.array([np.sum(effort < e) / n for e in effort])
    result_pcts = np.array([np.sum(result < r) / n for r in result])
    
    X = FSMClassifier.prepare_features(
        effort_pcts, result_pcts, compression, slope, speed
    )
    
    # Generate labels from heuristic
    classifier = FSMClassifier(MLConfig(model_type=model_type))
    y = classifier.generate_labels_from_heuristic(
        effort_pcts, result_pcts, compression, slope, speed
    )
    
    # Train
    metrics = classifier.train(X, y)
    
    print(f"\n[MLEngine] Training Results:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  CV Score: {metrics['cv_mean']:.2%} ± {metrics['cv_std']:.2%}")
    print(f"\n  Feature Importances:")
    for feat, imp in sorted(metrics['feature_importances'].items(), 
                           key=lambda x: x[1], reverse=True):
        print(f"    {feat}: {imp:.3f}")
    
    return classifier


if __name__ == "__main__":
    print("ML Engine - FSM Classifier")
    print("="*60)
    
    # Demo with synthetic data
    np.random.seed(42)
    n = 2000
    
    effort = np.random.lognormal(0, 0.5, n)
    result = np.random.exponential(0.5, n)
    compression = np.random.uniform(0.3, 0.9, n)
    slope = np.random.normal(0, 0.3, n)
    speed = np.diff(result, prepend=result[0])
    
    # Train classifier
    classifier = train_fsm_classifier(
        effort, result, compression, slope, speed,
        model_type='rf'
    )
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "fsm_model.joblib")
    classifier.save(model_path)
    
    # Export for MQL5
    mql5_path = os.path.join(os.path.dirname(__file__), "ml_thresholds.txt")
    classifier.export_for_mql5(mql5_path)
    
    # Test prediction
    print("\n[MLEngine] Test Prediction:")
    
    # Need at least window size samples for proper feature computation
    # Create test data first, then take slices
    test_effort_pcts = np.array([0.8, 0.3, 0.6, 0.5, 0.7, 0.4, 0.6])
    test_result_pcts = np.array([0.2, 0.9, 0.5, 0.6, 0.3, 0.8, 0.4])
    test_compression = np.array([0.8, 0.3, 0.5, 0.6, 0.7, 0.4, 0.5])
    test_slope = np.array([0.1, 0.5, -0.3, 0.2, 0.1, 0.4, -0.1])
    test_speed = np.array([0.1, 0.8, 2.5, 0.2, 0.3, 0.5, 0.2])
    
    X_test = classifier.prepare_features(
        test_effort_pcts,
        test_result_pcts,
        test_compression,
        test_slope,
        test_speed
    )
    
    predictions, confidences = classifier.predict_with_confidence(X_test)
    for i, (pred, conf) in enumerate(zip(predictions[:3], confidences[:3])):
        print(f"  Sample {i+1}: {classifier.STATE_NAMES[pred]} ({conf:.1f}%)")

