"""
COGNITIVE LAYER: FUZZY LOGIC ENGINE
===================================
Handles market ambiguity using Soft Computing.
Translates rigid numbers into human concepts (Low, Medium, High).
Dependency: scikit-fuzzy
"""
import numpy as np
import logging

logger = logging.getLogger("feat.fuzzy")

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("scikit-fuzzy not installed. Fuzzy Logic disabled.")

class FuzzyBrain:
    def __init__(self):
        if not FUZZY_AVAILABLE:
            return

        # Antecedents (Inputs)
        # RSI: 0-100
        self.rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
        # Volatility (ATR relative to price %): 0 to 5%
        self.volatility = ctrl.Antecedent(np.arange(0, 5.0, 0.1), 'volatility') 
        # Spread (Pips): 0 to 50
        self.spread = ctrl.Antecedent(np.arange(0, 50, 1), 'spread')
        
        # Consequents (Outputs)
        # Quality Score: 0-10
        self.quality = ctrl.Consequent(np.arange(0, 11, 1), 'quality')
        
        # ---------------------------------------------------------
        # MEMBERSHIP FUNCTIONS
        # ---------------------------------------------------------
        
        # RSI
        self.rsi['oversold'] = fuzz.trapmf(self.rsi.universe, [0, 0, 30, 40])
        self.rsi['neutral'] = fuzz.trimf(self.rsi.universe, [30, 50, 70])
        self.rsi['overbought'] = fuzz.trapmf(self.rsi.universe, [60, 70, 100, 100])
        
        # Volatility
        self.volatility['dead'] = fuzz.trapmf(self.volatility.universe, [0, 0, 0.5, 1.0])
        self.volatility['healthy'] = fuzz.trimf(self.volatility.universe, [0.5, 2.0, 3.5])
        self.volatility['extreme'] = fuzz.trapmf(self.volatility.universe, [2.5, 4.0, 5.0, 5.0])
        
        # Spread
        self.spread['tight'] = fuzz.trapmf(self.spread.universe, [0, 0, 5, 10])
        self.spread['normal'] = fuzz.trimf(self.spread.universe, [5, 15, 25])
        self.spread['wide'] = fuzz.trapmf(self.spread.universe, [20, 30, 50, 50])
        
        # Output Quality
        self.quality.automf(3, names=['poor', 'decent', 'prime'])
        
        # ---------------------------------------------------------
        # FUZZY RULES (The Expert System Knowledge Base)
        # ---------------------------------------------------------
        self.rules = [
            # HIGH QUALTITY SCENARIOS
            ctrl.Rule(self.spread['tight'] & self.volatility['healthy'], self.quality['prime']),
            ctrl.Rule(self.rsi['oversold'] & self.volatility['healthy'], self.quality['decent']), # Reversal
            ctrl.Rule(self.rsi['overbought'] & self.volatility['healthy'], self.quality['decent']),
            
            # LOW QUALITY SCENARIOS
            ctrl.Rule(self.spread['wide'], self.quality['poor']),
            ctrl.Rule(self.volatility['dead'], self.quality['poor']),
            ctrl.Rule(self.volatility['extreme'], self.quality['poor']), # Too risky
            
            # NEUTRAL
            ctrl.Rule(self.spread['normal'] & self.volatility['healthy'], self.quality['decent'])
        ]
        
        self.system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.system)
        
    def evaluate(self, rsi_val: float, vol_val: float, spread_val: float) -> float:
        """
        Returns a Market Quality Score (0-10).
        """
        if not FUZZY_AVAILABLE:
            return 5.0 # Fallback
            
        try:
            self.simulation.input['rsi'] = float(rsi_val)
            self.simulation.input['volatility'] = float(vol_val)
            self.simulation.input['spread'] = float(spread_val)
            
            self.simulation.compute()
            return self.simulation.output['quality']
        except Exception as e:
            # logger.error(f"Fuzzy Compute Error: {e}")
            return 5.0

# Singleton
fuzzy_brain = FuzzyBrain()
