from .utils import MT5_AVAILABLE, resilient, mt5
from .terminator import TerminalTerminator
from .manager import MT5Connection

# Global instance for project-wide use
mt5_conn = MT5Connection()

__all__ = ["mt5_conn", "MT5Connection", "TerminalTerminator", "MT5_AVAILABLE", "resilient", "mt5"]
