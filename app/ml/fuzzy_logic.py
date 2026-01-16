"""
COGNITIVE LAYER: FUZZY LOGIC ENGINE (Level 18)
==============================================
Handles market ambiguity using Soft Computing.
Translates rigid numbers into human concepts.
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

class FuzzyLogic:
    def __init__(self):
        if not FUZZY_AVAILABLE:
            return

        # ---------------------------------------------------------
        # ANTECEDENTS (Inputs)
        # ---------------------------------------------------------
        # RSI: 0-100
        self.rsi = ctrl.Antecedent(np.arange(0, 101, 1), 'rsi')
        
        # Acceleration (Price Velocity Change): 0.0 to 5.0 (Normalized)
        self.acceleration = ctrl.Antecedent(np.arange(0, 5.0, 0.1), 'acceleration') 
        
        # ---------------------------------------------------------
        # CONSEQUENT (Output)
        # ---------------------------------------------------------
        # Signal Strength: -10 (Strong Sell) to 10 (Strong Buy)
        self.signal = ctrl.Consequent(np.arange(-10, 11, 1), 'signal')

        # ---------------------------------------------------------
        # MEMBERSHIP FUNCTIONS
        # ---------------------------------------------------------
        
        # RSI (User: Low, Medium, High)
        self.rsi['low'] = fuzz.trapmf(self.rsi.universe, [0, 0, 30, 40])
        self.rsi['medium'] = fuzz.trimf(self.rsi.universe, [30, 50, 70])
        self.rsi['high'] = fuzz.trapmf(self.rsi.universe, [60, 70, 100, 100])
        
        # Acceleration (User: Slow, Normal, Explosive)
        self.acceleration['slow'] = fuzz.trapmf(self.acceleration.universe, [0, 0, 0.5, 0.8])
        self.acceleration['normal'] = fuzz.trimf(self.acceleration.universe, [0.5, 1.0, 1.5])
        self.acceleration['explosive'] = fuzz.trapmf(self.acceleration.universe, [1.2, 2.0, 5.0, 5.0])
        
        # Output Zones
        self.signal['sell'] = fuzz.trimf(self.signal.universe, [-10, -10, -5])
        self.signal['neutral'] = fuzz.trimf(self.signal.universe, [-5, 0, 5])
        self.signal['buy'] = fuzz.trimf(self.signal.universe, [5, 10, 10])

        # ---------------------------------------------------------
        # RULES (The Logic)
        # ---------------------------------------------------------
        self.rules = [
            # STRONG BUY: RSI Low (Oversold) + Explosive Accel (Reversal/Breakout)
            ctrl.Rule(self.rsi['low'] & self.acceleration['explosive'], self.signal['buy']),
            
            # NORMAL BUY: RSI Medium + Explosive (Momentum)
            ctrl.Rule(self.rsi['medium'] & self.acceleration['explosive'], self.signal['buy']),
            
            # STRONG SELL: RSI High (Overbought) + Explosive (Climax) - Wait, usually accel is directionless here
            # Assuming 'Acceleration' is magnitude. If Price is going UP and Accel is Explosive -> Climax?
            # Or if Price is going DOWN?
            # User example: IF RSI Low AND Accel Explosive THEN BUY.
            # I will assume 'Acceleration' supports the reversal logic for Oversold.
            
            # SELL Rule (Inverse) -> RSI High + Explosive (or just High)
            ctrl.Rule(self.rsi['high'], self.signal['sell']),
            
            # NEUTRAL
            ctrl.Rule(self.acceleration['slow'], self.signal['neutral']),
            ctrl.Rule(self.rsi['medium'] & self.acceleration['normal'], self.signal['neutral'])
        ]
        
        self.system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.system)
        
    def evaluate(self, rsi_val: float, accel_val: float) -> float:
        """
        Returns Signal Strength (-10 to 10).
        > 5: BUY
        < -5: SELL
        """
        if not FUZZY_AVAILABLE:
            return 0.0 
            
        try:
            self.simulation.input['rsi'] = float(rsi_val)
            self.simulation.input['acceleration'] = float(accel_val)
            
            self.simulation.compute()
            return self.simulation.output['signal']
        except Exception:
            return 0.0

# Singleton
fuzzy_logic = FuzzyLogic()
