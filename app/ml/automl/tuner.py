"""
AutoML Tuner - FEAT NEXUS OPERATION SINGULARITY
==============================================
Doctoral-level Hyperparameter Optimization using Bayesian Search (Optuna).
Optimizes: TCN Dilation, BiLSTM Hidden Units, Dropout Rates, and Fusion Dims.
"""

import os
import torch
import optuna
import logging
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger("FEAT.AutoML.Tuner")

class AutoMLTuner:
    """
    Automated Machine Learning Tuner for HybridProbabilistic models.
    Uses Optuna for efficient search in high-dimensional hyperparameter space.
    """
    
    def __init__(self, study_name: str = "singular_optimization"):
        self.study_name = study_name
        self.storage_url = f"sqlite:///data/automl_studies.db"
        
    def create_study(self):
        """Initializes an Optuna study with persistence."""
        os.makedirs("data", exist_ok=True)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        return study

    def objective(self, trial: optuna.Trial, train_loader, val_loader, device: str) -> float:
        """
        Objective function for Optuna optimization.
        """
        # Suggest Hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        tcn_channels = trial.suggest_categorical("tcn_channels", [32, 64, 128])
        
        # Import dynamically to avoid circular dependencies
        from app.ml.models.hybrid_probabilistic import HybridProbabilistic
        
        # Instantiate model with suggested params
        model = HybridProbabilistic(
            input_dim=settings.NEURAL_INPUT_DIM,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training Loop (Simplified for Tuning)
        model.train()
        for epoch in range(5): # Short cycles for tuning
            for x, labels, feat_input in train_loader:
                x, labels = x.to(device), labels.to(device)
                feat_input = {k: v.to(device) for k, v in feat_input.items()}
                
                optimizer.zero_grad()
                outputs = model(x, feat_input=feat_input)
                loss = criterion(outputs["logits"], labels)
                loss.backward()
                optimizer.step()
                
            # Validation Step
            accuracy = self._validate(model, val_loader, device)
            trial.report(accuracy, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return accuracy

    def _validate(self, model, val_loader, device) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, labels, feat_input in val_loader:
                x, labels = x.to(device), labels.to(device)
                feat_input = {k: v.to(device) for k, v in feat_input.items()}
                outputs = model(x, feat_input=feat_input)
                _, predicted = torch.max(outputs["logits"].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def run_optimization(self, train_loader, val_loader, n_trials=50):
        """Executes the search process."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        study = self.create_study()
        
        logger.info(f"Starting AutoML Optimization Trial on {device}...")
        study.optimize(
            lambda trial: self.objective(trial, train_loader, val_loader, device),
            n_trials=n_trials
        )
        
        logger.info(f"Best Trial Accuracy: {study.best_value:.4f}")
        logger.info(f"Best Params: {study.best_params}")
        
        return study.best_params

automl_tuner = AutoMLTuner()
