"""
[MICROSTRUCTURE ENGINE]
The Physics Layer of the FEAT System.
Handles:
- Price Impact Modeling (Liquidity Estimation)
- Fractal Persistence (Hurst Exponent)
- Order Flow Imbalance (OFI)
- Shannon Entropy (Market Noise Detection)
- Poisson Arrival Rates
"""
from .impact_model import PriceImpactModel
from .hurst import HurstExponent
from .ofi import OrderFlowImbalance
from .shannon_entropy import ShannonEntropyAnalyzer, entropy_analyzer
from .scanner import MicrostructureScanner, micro_scanner


