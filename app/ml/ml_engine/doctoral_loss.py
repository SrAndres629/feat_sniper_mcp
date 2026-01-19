import torch
import torch.nn as nn
import torch.nn.functional as F

class DoctoralLoss(nn.Module):
    """
    [MATH SENIOR FULLSTACK]
    Composite Loss Function for Probabilistic Hybrid Models.
    
    Components:
    1. Aleatoric CrossEntropy: Handles label noise via Variance Sampling.
    2. Quantile Loss: For 'p_win' regression (Confidence Interval Learning).
    3. Stability: Explicit Log-Sum-Exp implementation.
    """
    def __init__(self, num_classes=3, monte_carlo_samples=100, quantile=0.5):
        super(DoctoralLoss, self).__init__()
        self.num_classes = num_classes
        self.T = monte_carlo_samples
        self.quantile = quantile
        
    def log_sum_exp_stability(self, logits):
        """
        [SUBSKILL: COMPUTATIONAL]
        Stable LogSoftmax implementation.
        """
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # x - max(x) prevents overflow
        return logits - max_logits - torch.log(torch.sum(torch.exp(logits - max_logits), dim=1, keepdim=True))

    def quantile_loss(self, preds, target):
        """
        [SUBSKILL: NEURAL]
        Pinball Loss for Uncertainty Quantification.
        L_q = max(q * (y - y_p), (q - 1) * (y - y_p))
        """
        errors = target - preds
        loss = torch.max(
            (self.quantile * errors),
            ((self.quantile - 1) * errors)
        )
        return torch.mean(loss)

    def aleatoric_classification_loss(self, logits, log_var, targets):
        """
        [SUBSKILL: PROBABILISTIC]
        Heteroscedastic Loss for Classification.
        We assume logits are corrupted by Gaussian noise with variance = exp(log_var).
        We sample T distortions to approximate the expected loss under uncertainty.
        
        Args:
            logits: (B, C)
            log_var: (B, 1) - Aleatoric Uncertainty
            targets: (B)
        """
        std = torch.exp(0.5 * log_var) # (B, 1)
        
        # Monte Carlo Sampling
        loss_sum = 0
        for _ in range(self.T):
            # Reparameterization Trick: z = mu + sigma * epsilon
            # We add noise to the logits to simulate uncertainty
            epsilon = torch.randn_like(logits) * std
            distorted_logits = logits + epsilon
            
            # Standard CrossEntropy on distorted logits
            loss_sum += F.cross_entropy(distorted_logits, targets)
            
        return loss_sum / self.T

    def forward(self, model_outputs, targets):
        """
        Args:
            model_outputs: Dict containing "logits", "p_win", "log_var"
            targets: Dict containing "class", "profit" (optional)
        """
        logits = model_outputs["logits"]
        log_var = model_outputs["log_var"]
        p_win = model_outputs["p_win"]
        
        # 1. Classification Loss (with Aleatoric Uncertainty)
        # Regularization: Uncertainty (log_var) essentially regularizes the logits.
        class_loss = self.aleatoric_classification_loss(logits, log_var, targets["class"])
        
        # 2. Quantile Loss on Win Probability
        # If we have a 'profit' target or validation, we can treat it as binary 1.0 (Win) or 0.0 (Loss)
        # Or checking if class is correct.
        # Here we approximate: If target class is Bullish(0)/Bearish(2) -> 1.0, Neutral(1) -> 0.0?
        # Better: The user didn't specify the 'p_win' target. We will self-supervise or ignore for now
        # BUT user asked to USE Quantile Loss.
        # We will apply Quantile Loss to the 'alpha' or 'p_win' if available. 
        # Let's assume 'p_win' targets are binary 1.0 (Profitable) or 0.0 (Loss) derived from class.
        
        # Derived Target: 1.0 if not Neutral, 0.0 if Neutral? 
        # Or just use the class directly? 
        # Let's trust the 'targets' passed in. If simple 'y' from train_lstm, it's just class.
        # We will construct a dummy quantile target to satisfy the requirement if actual profit data isn't there.
        # Assuming y=0 (Sell), y=1 (Hold), y=2 (Buy).
        # We want to predict confidence.
        # Let's apply simple Quantile Loss as a Regularizer on 'log_var' directly? No.
        
        # Let's perform Quantile Loss regression on the logits themselves? No.
        
        # Implementation Decision:
        # We will add an auxiliary Quantile Loss on 'p_win' to match the 'correctness' of prediction.
        # Target = 1.0 if prediction matches label, 0.0 otherwise.
        # This trains the network to estimate its own accuracy (Calibration).
        
        with torch.no_grad():
            pred_class = torch.argmax(logits, dim=1)
            correctness = (pred_class == targets["class"]).float().unsqueeze(1)
            
        confidence_loss = self.quantile_loss(p_win, correctness)
        
        # Total Loss
        # We add a regularization term for variance to prevent it from growing infinite to minimize loss
        # The aleatoric sampling naturally handles this (high variance = high noise = bad CE).
        # But explicit regularization helps.
        
        total_loss = class_loss + 0.5 * confidence_loss + 0.1 * torch.mean(torch.exp(log_var))
        
        return total_loss
