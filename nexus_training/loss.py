
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsAwareLoss(nn.Module):
    """
    [ARCH-HYBRID-V1] Physics-Regularized Loss Function.
    Penalizes the model if it predicts strong moves (BUY/SELL) against Physics Laws (Acceleration).
    """
    def __init__(self, physics_lambda=0.5):
        super(PhysicsAwareLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.physics_lambda = physics_lambda
        
    def forward(self, pred, target, physics_inputs):
        """
        pred: Logits (Batch, 3) [SELL, HOLD, BUY]
        target: Labels (Batch) [0, 1, 2]
        physics_inputs: (Batch, Features) where feature 'acceleration' is normalized 0-1
        """
        # 1. Classification Error (CrossEntropy expects logits)
        ce = self.ce_loss(pred, target)
        
        # 2. Physics Violation
        # Convert logits to probabilities
        probs = F.softmax(pred, dim=1)
        
        # Extract Probabilities
        prob_sell = probs[:, 0]
        prob_buy = probs[:, 2]
        
        # Extract Acceleration (Assuming it's the last feature or passed specifically)
        # For this implementation, we assume physics_inputs contains 'acceleration' at index -1
        # In practice, the DataLoader should yield this specifically.
        # If physics_inputs is just the features tensor, we need to know the index.
        # Assuming 'acceleration' is passed as a separate tensor of shape (Batch,)
        accel = physics_inputs
        
        # Violation 1: Strong BUY prediction but Low Acceleration
        buy_violation = torch.relu(prob_buy - accel)
        
        # Violation 2: Strong SELL prediction but Low Acceleration (Accel is magnitude)
        sell_violation = torch.relu(prob_sell - accel) 
        
        # Total Penalty
        physics_penalty = torch.mean(buy_violation + sell_violation) * self.physics_lambda
        
        return ce + physics_penalty
