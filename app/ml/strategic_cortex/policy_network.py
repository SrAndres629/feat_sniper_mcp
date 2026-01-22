"""
FEAT SNIPER: STRATEGIC POLICY NETWORK (PPO Actor-Critic)
=========================================================
Neural network that learns optimal trading strategy selection.

Uses Proximal Policy Optimization (PPO) for stable learning.
Implements:
- Actor: Policy π(a|s) - probability distribution over actions
- Critic: Value V(s) - expected return from state s
"""

import os
import logging
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from .action_space import StrategicAction, get_num_actions
from .state_encoder import StateVector

logger = logging.getLogger("FEAT.PolicyNetwork")


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for strategic policy optimization.
    
    Architecture:
    - Shared backbone with 2 hidden layers
    - Actor head: outputs action probabilities
    - Critic head: outputs state value estimate
    """
    
    def __init__(self, 
                 state_dim: int = 45,
                 hidden_dim: int = 128,
                 num_actions: int = 4):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with orthogonal initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Small initialization for action output
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and value estimate.
        
        Args:
            state: Batch of state vectors [B, state_dim].
            
        Returns:
            action_logits: [B, num_actions]
            value: [B, 1]
        """
        features = self.backbone(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Returns softmax action probabilities."""
        logits, _ = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Returns state value estimate."""
        _, value = self.forward(state)
        return value


class StrategicPolicyAgent:
    """
    High-level agent that uses the PolicyNetwork for decision making.
    
    Handles:
    - Action selection (with exploration via sampling)
    - Experience storage for training
    - Model persistence (save/load)
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "cpu",
                 exploration_mode: bool = True):
        """
        Args:
            model_path: Path to load pre-trained weights.
            device: "cpu" or "cuda".
            exploration_mode: If True, sample actions. If False, take argmax.
        """
        self.device = torch.device(device)
        self.exploration_mode = exploration_mode
        
        # Initialize network
        self.network = PolicyNetwork(
            state_dim=StateVector.get_state_dim(),
            hidden_dim=128,
            num_actions=get_num_actions(),
        ).to(self.device)
        
        # Shadow mode flag (suggest but don't execute)
        self.shadow_mode = True
        
        # Experience buffer for training
        self.experience_buffer = []
        self.experience_log_path = "data/experiences.jsonl"
        os.makedirs("data", exist_ok=True)
        
        # Load pretrained if available
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
            logger.info(f"✅ Loaded policy weights from {model_path}")
        else:
            logger.info("🧠 Initialized PolicyNetwork with random weights (training mode)")
    
    def select_action(self, state: StateVector, deterministic: bool = False) -> Tuple[StrategicAction, float, float]:
        """
        Selects an action given the current state.
        
        Args:
            state: Current environment state.
            deterministic: If True, always pick highest probability action.
            
        Returns:
            Tuple of (action, action_probability, state_value).
        """
        self.network.eval()
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(self.device)
            logits, value = self.network(state_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            
            if deterministic or not self.exploration_mode:
                action_idx = torch.argmax(probs).item()
            else:
                # Sample from distribution for exploration
                action_idx = torch.multinomial(probs, 1).item()
            
            action_prob = probs[action_idx].item()
            state_value = value.squeeze().item()
        
        return StrategicAction(action_idx), action_prob, state_value
    
    def get_action_distribution(self, state: StateVector) -> dict:
        """
        Returns full probability distribution over actions.
        
        Useful for logging and debugging.
        """
        self.network.eval()
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(self.device)
            probs = self.network.get_action_probs(state_tensor).squeeze(0)
        
        return {
            StrategicAction(i).name: round(probs[i].item(), 4)
            for i in range(get_num_actions())
        }
    
    def record_experience(self, 
                          state: StateVector,
                          action: StrategicAction,
                          reward: float,
                          next_state: StateVector,
                          done: bool):
        """
        Stores transition for offline training and persists to disk.
        """
        exp = {
            "state": state.to_tensor().tolist(),
            "action": action.value,
            "reward": float(reward),
            "next_state": next_state.to_tensor().tolist(),
            "done": done,
            "timestamp": datetime.now().isoformat()
        }
        self.experience_buffer.append(exp)
        
        # Persistent Logging (Append to JSONL)
        try:
            with open(self.experience_log_path, 'a') as f:
                f.write(json.dumps(exp) + "\n")
        except Exception as e:
            logger.error(f"Failed to log experience to disk: {e}")
        
        # Limit RAM buffer size
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
    
    def save_weights(self, path: str):
        """Saves model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)
        logger.info(f"💾 Policy weights saved to {path}")
    
    def load_weights(self, path: str):
        """Loads model weights from disk."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()
    
    def pretrain(self, states: List[StateVector], target_actions: List[StrategicAction], epochs: int = 5):
        """
        Performs Supervised Pre-training (Imitation Learning).
        Train the network to mimic a 'Teacher' (e.g., the legacy if/else logic).
        
        Args:
            states: List of states captured during teacher execution.
            target_actions: Actions chosen by the teacher for those states.
            epochs: Number of training passes.
        """
        if not states:
            return
            
        self.network.train()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        state_tensors = torch.stack([torch.from_numpy(s.to_tensor()) for s in states]).to(self.device)
        action_tensors = torch.tensor([a.value for a in target_actions]).to(self.device)
        
        logger.info(f"🎓 PRE-TRAINING: Mimicking Teacher on {len(states)} samples...")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits, _ = self.network(state_tensors)
            loss = criterion(logits, action_tensors)
            loss.backward()
            optimizer.step()
            
            if epoch % 1 == 0:
                logger.info(f"   Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
        
        self.network.eval()
        logger.info("✅ Pre-training complete. Agent is now aligned with Legacy Rules.")
        
        # Record training session for dashboard
        from nexus_core.neural_health import neural_health
        neural_health.record_training_session(
            session_type="PRETRAIN",
            samples=len(states),
            epochs=epochs
        )

    def get_decision_report(self, state: StateVector) -> dict:
        """
        Comprehensive decision report for logging/debugging.
        """
        action, prob, value = self.select_action(state, deterministic=True)
        distribution = self.get_action_distribution(state)
        
        return {
            "recommended_action": action.name,
            "confidence": round(prob, 4),
            "state_value": round(value, 4),
            "distribution": distribution,
            "shadow_mode": self.shadow_mode,
            "exploration": self.exploration_mode,
        }


# Singleton for global access
policy_agent = StrategicPolicyAgent(
    model_path="models/strategic_policy.pth",
    device="cpu",
    exploration_mode=True,
)
