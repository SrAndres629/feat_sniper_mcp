from .models import ValidationResult, FEATDecision, MicroStructure
from .rules.form import FormRule
from .rules.space import SpaceRule
from .rules.acceleration import AccelerationRule
from .rules.time import TimeRule
from .chain import FEATChain

# Singleton for institutional access (Backward Compatibility)
feat_full_chain_institucional = FEATChain()
