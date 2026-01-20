"""
FEAT SNIPER: STRATEGIC ACTION SPACE
====================================
Defines the discrete action space for the Strategic Policy Agent.

Actions represent trading strategy modes, not individual trades.
Each action maps to specific risk/reward parameters in StrategyEngine.
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, Any


class StrategicAction(IntEnum):
    """
    Discrete actions the Policy Agent can select.
    
    Action semantics:
    - TWIN_SNIPER: Aggressive dual-position (Scalp + Runner)
    - STANDARD: Single conservative position
    - DEFENSIVE: Minimal exposure, tight stops
    - HOLD: Wait, do not trade
    """
    TWIN_SNIPER = 0
    STANDARD = 1
    DEFENSIVE = 2
    HOLD = 3


@dataclass
class ActionConfig:
    """Configuration parameters for each action type."""
    risk_multiplier: float      # Multiplier on base risk % (e.g., 1.5 = 150% of normal)
    use_twin_trading: bool      # Whether to split into two positions
    tp_multiplier: float        # Take profit distance multiplier
    sl_tightness: float         # Stop loss tightness (0.5 = 50% of normal SL)
    runner_enabled: bool        # Whether to allow a swing/runner position
    description: str


# Action configuration mapping
ACTION_CONFIGS: Dict[StrategicAction, ActionConfig] = {
    StrategicAction.TWIN_SNIPER: ActionConfig(
        risk_multiplier=1.5,
        use_twin_trading=True,
        tp_multiplier=1.0,
        sl_tightness=1.0,
        runner_enabled=True,
        description="Aggressive mode: Twin positions for cash + wealth accumulation"
    ),
    StrategicAction.STANDARD: ActionConfig(
        risk_multiplier=1.0,
        use_twin_trading=False,
        tp_multiplier=1.0,
        sl_tightness=1.0,
        runner_enabled=False,
        description="Standard mode: Single position with normal parameters"
    ),
    StrategicAction.DEFENSIVE: ActionConfig(
        risk_multiplier=0.5,
        use_twin_trading=False,
        tp_multiplier=0.7,
        sl_tightness=0.6,
        runner_enabled=False,
        description="Defensive mode: Reduced size, tight stops, quick exits"
    ),
    StrategicAction.HOLD: ActionConfig(
        risk_multiplier=0.0,
        use_twin_trading=False,
        tp_multiplier=0.0,
        sl_tightness=0.0,
        runner_enabled=False,
        description="Hold mode: No trading, wait for better opportunity"
    ),
}


def get_action_config(action: StrategicAction) -> ActionConfig:
    """Returns the configuration for a given action."""
    return ACTION_CONFIGS[action]


def get_num_actions() -> int:
    """Returns the size of the action space."""
    return len(StrategicAction)


def action_to_name(action: int) -> str:
    """Converts action index to human-readable name."""
    return StrategicAction(action).name


def action_from_name(name: str) -> StrategicAction:
    """Converts name string to action enum."""
    return StrategicAction[name.upper()]
