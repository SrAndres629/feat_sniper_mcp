from enum import Enum

class ExecutionMode(str, Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    BACKTEST = "BACKTEST"
