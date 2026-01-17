from .models import EMALayer, LayerMetrics, StructuralConfidence
from .ema_layers import FourLayerEMA, four_layer_ema
from .patterns import MAE_Pattern_Recognizer
from .fractals import identify_fractals
from .zones import detect_zones
from .transitions import detect_structural_shifts
from .engine import StructureEngine

# Singleton for backward compatibility
structure_engine = StructureEngine()
