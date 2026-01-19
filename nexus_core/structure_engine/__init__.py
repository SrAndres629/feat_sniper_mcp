from .fractals import identify_fractals
from .zones import detect_zones
from .transitions import detect_structural_shifts
from .imbalances import detect_imbalances
from .liquidity import detect_liquidity_pools
from .order_blocks import detect_order_blocks
from .trap_detector import calculate_trap_score, get_trap_report
from .engine import StructureEngine

# Singleton for backward compatibility
structure_engine = StructureEngine()
