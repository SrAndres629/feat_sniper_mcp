from .models import ValidationResult, FEATDecision, MicroStructure
from .liquidity import LiquidityChannel
from .kinetics import KineticsChannel
from .volatility import VolatilityChannel
from .main_chain import FEATChain

# Singleton for institutional access (Backward Compatibility)
feat_full_chain_institucional = FEATChain()
