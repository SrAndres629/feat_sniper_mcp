from .fractals import identify_fractals
from .transitions import detect_structural_shifts
from .imbalances import detect_imbalances
from .liquidity import detect_liquidity_pools
from .order_blocks import detect_order_blocks
from .trap_detector import calculate_trap_score, get_trap_report
from .critical_points import detect_critical_points
from .shadow_zones import detect_shadow_zones
from .consolidation_zones import detect_consolidation_zones
from .engine import StructureEngine

# Singleton for backward compatibility
structure_engine = StructureEngine()
