"""
MODULO 9: Feature Importance & Contribution Analysis
Phase 13: The Profit Pulse

Objetivo del Visionario:
- Analizar la contribucion de cada feature al rendimiento del modelo
- Identificar features que aportan valor vs ruido
- Optimizar el vector de entrada para maxima precision

Features del Vector 5D:
1. L1_Mean - Precio micro promedio
2. L1_Width - Volatilidad micro
3. L4_Slope - Tendencia gravitacional
4. Div_L1_L2 - Divergencia entre capas
5. Vol_ZScore - Anomalias de volumen
"""
import logging
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("feat.feature_importance")

class FeatureImportanceAnalyzer:
    """
    Analizador de importancia de features.
    Rastrea la correlacion entre cada feature y el resultado del trade.
    """
    
    FEATURE_NAMES = ["L1_Mean", "L1_Width", "L4_Slope", "Div_L1_L2", "Vol_ZScore"]
    
    def __init__(self):
        self.feature_records = defaultdict(list)  # {feature_name: [(value, outcome)]}
        self.trade_count = 0
        
    def record_trade(self, features: List[float], outcome: str):
        """
        Registra un trade con sus features y resultado.
        
        Args:
            features: Vector 5D [L1_Mean, L1_Width, L4_Slope, Div_L1_L2, Vol_ZScore]
            outcome: "WIN" or "LOSS"
        """
        if len(features) != 5:
            logger.warning(f"Expected 5 features, got {len(features)}")
            return
            
        outcome_val = 1 if outcome == "WIN" else 0
        
        for i, name in enumerate(self.FEATURE_NAMES):
            self.feature_records[name].append((features[i], outcome_val))
        
        self.trade_count += 1
    
    def calculate_importance(self) -> Dict[str, Any]:
        """
        Calcula la importancia de cada feature usando correlacion simple.
        """
        if self.trade_count < 20:
            return {"status": "INSUFFICIENT_DATA", "min_required": 20, "current": self.trade_count}
        
        importance = {}
        
        for name in self.FEATURE_NAMES:
            records = self.feature_records[name]
            values = np.array([r[0] for r in records])
            outcomes = np.array([r[1] for r in records])
            
            # Simple correlation coefficient
            if np.std(values) > 0 and np.std(outcomes) > 0:
                correlation = np.corrcoef(values, outcomes)[0, 1]
            else:
                correlation = 0.0
            
            # Win rate when feature is above/below median
            median_val = np.median(values)
            above_median = outcomes[values > median_val]
            below_median = outcomes[values <= median_val]
            
            wr_above = np.mean(above_median) if len(above_median) > 0 else 0.5
            wr_below = np.mean(below_median) if len(below_median) > 0 else 0.5
            
            importance[name] = {
                "correlation": round(correlation, 3),
                "wr_above_median": round(wr_above, 3),
                "wr_below_median": round(wr_below, 3),
                "signal_strength": round(abs(wr_above - wr_below), 3),
                "predictive": abs(correlation) > 0.1
            }
        
        # Rank features by signal strength
        ranked = sorted(importance.items(), key=lambda x: x[1]["signal_strength"], reverse=True)
        
        return {
            "status": "ANALYZED",
            "trade_count": self.trade_count,
            "features": importance,
            "ranking": [f[0] for f in ranked],
            "top_predictor": ranked[0][0] if ranked else None,
            "recommendation": self._get_recommendation(importance)
        }
    
    def _get_recommendation(self, importance: Dict) -> str:
        """Genera recomendaciones basadas en el analisis."""
        weak_features = [name for name, data in importance.items() if not data["predictive"]]
        
        if not weak_features:
            return "All features contributing. System optimal."
        elif len(weak_features) <= 2:
            return f"Consider removing or replacing: {', '.join(weak_features)}"
        else:
            return "Multiple weak features. Model may need retraining."
    
    def generate_report(self) -> str:
        """Genera un reporte de importancia de features."""
        analysis = self.calculate_importance()
        
        if analysis["status"] == "INSUFFICIENT_DATA":
            return f"# Feature Importance Report\n⚠️ Insufficient data ({analysis['current']}/{analysis['min_required']} trades)"
        
        report = f"""# 📊 Feature Importance Report
**Timestamp:** {datetime.now().isoformat()}
**Trades Analyzed:** {analysis['trade_count']}

## Feature Ranking (by Signal Strength)

| Rank | Feature | Correlation | WR Above Med | WR Below Med | Signal | Predictive |
|------|---------|-------------|--------------|--------------|--------|------------|
"""
        for i, name in enumerate(analysis['ranking']):
            data = analysis['features'][name]
            report += f"| {i+1} | {name} | {data['correlation']:.3f} | {data['wr_above_median']:.1%} | {data['wr_below_median']:.1%} | {data['signal_strength']:.3f} | {'✅' if data['predictive'] else '❌'} |\n"
        
        report += f"""
## Top Predictor: **{analysis['top_predictor']}**

## Recommendation
{analysis['recommendation']}

---
*Generated by FEAT Feature Importance Module 9 - Phase 13*
"""
        return report

# Singleton global
feature_analyzer = FeatureImportanceAnalyzer()

def test_feature_importance():
    """Test the feature importance analyzer."""
    import random
    print("=" * 60)
    print("📊 FEAT SYSTEM - MODULE 9: FEATURE IMPORTANCE TEST")
    print("=" * 60)
    
    fa = FeatureImportanceAnalyzer()
    
    # Simulate 30 trades with correlated features
    for i in range(30):
        # L4_Slope strongly correlates with wins
        l4_slope = random.uniform(-5, 5)
        outcome = "WIN" if l4_slope > 0 and random.random() > 0.3 else "LOSS"
        
        features = [
            random.uniform(1800, 1850),  # L1_Mean
            random.uniform(0.5, 2.0),     # L1_Width
            l4_slope,                     # L4_Slope (correlated)
            random.uniform(0.95, 1.05),   # Div_L1_L2
            random.uniform(-2, 2)         # Vol_ZScore
        ]
        fa.record_trade(features, outcome)
    
    analysis = fa.calculate_importance()
    print(f"\n📊 Feature Analysis:")
    print(f"   Top Predictor: {analysis['top_predictor']}")
    print(f"   Ranking: {' > '.join(analysis['ranking'])}")
    print(f"   Recommendation: {analysis['recommendation']}")

if __name__ == "__main__":
    test_feature_importance()
