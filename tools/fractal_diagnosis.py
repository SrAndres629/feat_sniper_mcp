"""
FEAT SNIPER: MULTI-TIMEFRAME FRACTAL COHERENCE ENGINE
=====================================================
Measures the "Resonance" between different timeframes to determine
if the market is in a COHERENT (aligned) or CHAOTIC (conflicting) state.

Core Concept: If M1, M5, M15, H1, H4, D1, W1 all agree on direction,
the probability of a successful trade is exponentially higher.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class TimeframeBias(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1

@dataclass
class TimeframeAnalysis:
    timeframe: str
    bias: TimeframeBias
    strength: float  # 0.0 to 1.0
    hurst_exponent: float
    wavelet_energy: float

class FractalCoherenceEngine:
    """
    Measures alignment across multiple timeframes using:
    1. Hurst Exponent (Trend Persistence)
    2. Wavelet Energy (Momentum)
    3. SGI (Spectral Gravity Index)
    """
    
    # Timeframe hierarchy (from micro to macro)
    TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
    
    # Weights: Macro timeframes have more weight in coherence calculation
    TF_WEIGHTS = {
        'M1': 0.05,
        'M5': 0.10,
        'M15': 0.15,
        'M30': 0.10,
        'H1': 0.15,
        'H4': 0.20,
        'D1': 0.15,
        'W1': 0.10
    }
    
    def __init__(self):
        self.last_analysis: Dict[str, TimeframeAnalysis] = {}
        
    def analyze_timeframe(self, tf_data: dict) -> TimeframeAnalysis:
        """
        Analyzes a single timeframe and returns its bias and strength.
        
        tf_data should contain: {
            'timeframe': 'H4',
            'sgi': 0.65,       # Spectral Gravity Index (-1 to 1)
            'hurst': 0.72,     # Hurst Exponent (0 to 1)
            'wavelet_energy': 0.8,  # Energy burst
            'ema_slope': 0.002 # Price momentum
        }
        """
        sgi = tf_data.get('sgi', 0.0)
        hurst = tf_data.get('hurst', 0.5)
        energy = tf_data.get('wavelet_energy', 0.5)
        slope = tf_data.get('ema_slope', 0.0)
        
        # Determine Bias from SGI and Slope
        if sgi > 0.3 and slope > 0:
            bias = TimeframeBias.BULLISH
        elif sgi < -0.3 and slope < 0:
            bias = TimeframeBias.BEARISH
        else:
            bias = TimeframeBias.NEUTRAL
            
        # Strength is combination of Hurst (persistence) and Energy (momentum)
        strength = (hurst * 0.6) + (energy * 0.4)
        
        return TimeframeAnalysis(
            timeframe=tf_data['timeframe'],
            bias=bias,
            strength=min(1.0, strength),
            hurst_exponent=hurst,
            wavelet_energy=energy
        )
    
    def calculate_coherence(self, all_tf_data: List[dict]) -> dict:
        """
        Calculates the overall Fractal Coherence Score.
        
        Returns: {
            'coherence_score': 0.0 - 1.0,
            'dominant_bias': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
            'alignment_map': {'M1': 'BULLISH', 'H4': 'BULLISH', ...},
            'recommendation': 'TWIN_AGGRESSIVE' | 'SINGLE_STANDARD' | 'NO_TRADE'
        }
        """
        analyses = []
        for tf_data in all_tf_data:
            analysis = self.analyze_timeframe(tf_data)
            self.last_analysis[tf_data['timeframe']] = analysis
            analyses.append(analysis)
        
        # Count bias votes (weighted)
        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0
        
        alignment_map = {}
        
        for a in analyses:
            weight = self.TF_WEIGHTS.get(a.timeframe, 0.1) * a.strength
            alignment_map[a.timeframe] = a.bias.name
            
            if a.bias == TimeframeBias.BULLISH:
                bullish_weight += weight
            elif a.bias == TimeframeBias.BEARISH:
                bearish_weight += weight
            else:
                neutral_weight += weight
        
        total_weight = bullish_weight + bearish_weight + neutral_weight
        if total_weight == 0:
            total_weight = 1.0
            
        # Dominant Bias
        if bullish_weight > bearish_weight and bullish_weight > neutral_weight:
            dominant = 'BULLISH'
            coherence = bullish_weight / total_weight
        elif bearish_weight > bullish_weight and bearish_weight > neutral_weight:
            dominant = 'BEARISH'
            coherence = bearish_weight / total_weight
        else:
            dominant = 'NEUTRAL'
            coherence = neutral_weight / total_weight
            
        # Recommendation based on Coherence Level
        if coherence >= 0.75:
            recommendation = 'TWIN_AGGRESSIVE'
        elif coherence >= 0.50:
            recommendation = 'SINGLE_STANDARD'
        else:
            recommendation = 'NO_TRADE'
            
        return {
            'coherence_score': round(coherence, 3),
            'dominant_bias': dominant,
            'alignment_map': alignment_map,
            'recommendation': recommendation,
            'bullish_weight': round(bullish_weight, 3),
            'bearish_weight': round(bearish_weight, 3)
        }

# Singleton for global access
fractal_engine = FractalCoherenceEngine()

def diagnose_market_fractals(mock_mode=True) -> dict:
    """
    Main entry point for fractal diagnosis.
    In production, this would fetch real data from MT5.
    """
    if mock_mode:
        # Simulate aligned market (High Coherence BUY scenario)
        mock_data = [
            {'timeframe': 'M1', 'sgi': 0.4, 'hurst': 0.65, 'wavelet_energy': 0.7, 'ema_slope': 0.001},
            {'timeframe': 'M5', 'sgi': 0.5, 'hurst': 0.68, 'wavelet_energy': 0.75, 'ema_slope': 0.002},
            {'timeframe': 'M15', 'sgi': 0.6, 'hurst': 0.70, 'wavelet_energy': 0.8, 'ema_slope': 0.003},
            {'timeframe': 'M30', 'sgi': 0.55, 'hurst': 0.72, 'wavelet_energy': 0.78, 'ema_slope': 0.002},
            {'timeframe': 'H1', 'sgi': 0.65, 'hurst': 0.75, 'wavelet_energy': 0.85, 'ema_slope': 0.004},
            {'timeframe': 'H4', 'sgi': 0.70, 'hurst': 0.78, 'wavelet_energy': 0.9, 'ema_slope': 0.005},
            {'timeframe': 'D1', 'sgi': 0.60, 'hurst': 0.80, 'wavelet_energy': 0.88, 'ema_slope': 0.003},
            {'timeframe': 'W1', 'sgi': 0.50, 'hurst': 0.82, 'wavelet_energy': 0.85, 'ema_slope': 0.002},
        ]
        return fractal_engine.calculate_coherence(mock_data)
    else:
        # TODO: Fetch real data from MT5 bridge
        raise NotImplementedError("Production mode requires MT5 bridge")

if __name__ == "__main__":
    print("=== FRACTAL COHERENCE DIAGNOSIS ===")
    result = diagnose_market_fractals(mock_mode=True)
    
    print(f"\n🌀 Coherence Score: {result['coherence_score']*100:.1f}%")
    print(f"📊 Dominant Bias: {result['dominant_bias']}")
    print(f"🎯 Recommendation: {result['recommendation']}")
    print(f"\nAlignment Map:")
    for tf, bias in result['alignment_map'].items():
        print(f"   {tf}: {bias}")
