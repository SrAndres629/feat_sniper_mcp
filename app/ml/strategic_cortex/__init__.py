"""
FEAT SNIPER: STRATEGIC CORTEX
=============================
Reinforcement Learning module for adaptive strategy selection.

Replaces deterministic rules with a learned Policy Network (PPO)
that decides optimal trading mode based on account state,
market microstructure, and neural predictions.

Components:
- ActionSpace: Discrete actions (TWIN_SNIPER, STANDARD, DEFENSIVE, HOLD)
- StateEncoder: Transforms context into 16-dim normalized vector
- PolicyNetwork: PPO Actor-Critic that learns optimal policy
"""

from .action_space import (
    StrategicAction,
    ActionConfig,
    ACTION_CONFIGS,
    get_action_config,
    get_num_actions,
    action_to_name,
    action_from_name,
)
from .state_encoder import (
    StateVector,
    StateEncoder,
    state_encoder,
)
from .policy_network import (
    PolicyNetwork,
    StrategicPolicyAgent,
    policy_agent,
)

__all__ = [
    # Action Space
    "StrategicAction",
    "ActionConfig",
    "ACTION_CONFIGS",
    "get_action_config",
    "get_num_actions",
    "action_to_name",
    "action_from_name",
    # State Encoder
    "StateVector",
    "StateEncoder",
    "state_encoder",
    # Policy Network
    "PolicyNetwork",
    "StrategicPolicyAgent",
    "policy_agent",
]
