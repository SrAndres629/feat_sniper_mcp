"""
ml_engine.py - Multifractal Layer Intelligence
Upgraded to understand:
1. Layer relationship (Compression/Expansion)
2. Regime classification (Absorption, Manipulation, Continuity)
3. Cognitive alerts for layer state changes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class MultifractalAI:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Professional Labels for IA perception
        self.regimes = ['ABSORPTION', 'MANIPULATION', 'CONTINUITY', 'CONSOLIDATION', 'EXHAUSTION']
        
    def extract_layer_features(self, df: pd.DataFrame):
        """
        Transforms raw data into multifractal questions:
        1. Compresión de Micro Nube
        2. Separación entre Operativa y Macro
        3. Inclinación de Capa 2 (Estructura)
        4. Relación de Capas (Acercamiento vs Alejamiento)
        """
        X = pd.DataFrame()
        X['micro_comp'] = df['microComp']
        X['oper_slope'] = df['operSlope']
        X['layer_sep'] = df['layerSep']
        X['effort_result'] = df['effort'] / (df['result'] + 1e-6)
        
        # Derived: Acceleration of layers
        X['sep_velocity'] = X['layer_sep'].diff()
        X['slope_accel'] = X['oper_slope'].diff()
        
        return X.fillna(0)

    def train_on_layers(self, csv_path: str):
        if not os.path.exists(csv_path):
            return "No data for training"
            
        df = pd.read_csv(csv_path)
        X = self.extract_layer_features(df)
        
        # Bootstrap labels based on professional layer rules
        y = []
        for i, row in X.iterrows():
            if row['micro_comp'] > 0.8 and row['effort_result'] > 2.0:
                y.append(0) # ABSORPTION
            elif abs(row['sep_velocity']) > 1.0 and abs(row['oper_slope']) < 0.1:
                y.append(1) # MANIPULATION
            elif abs(row['oper_slope']) > 0.3 and row['sep_velocity'] > 0:
                y.append(2) # CONTINUITY
            elif row['micro_comp'] > 0.7 and abs(row['oper_slope']) < 0.1:
                y.append(3) # CONSOLIDATION
            else:
                y.append(4) # EXHAUSTION
                
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Save model
        joblib.dump(self.model, 'fsm_multifractal.joblib')
        joblib.dump(self.scaler, 'scaler_multifractal.joblib')
        return "Model trained on Multifractal Layer Patterns"

    def predict_regime(self, current_metrics: dict):
        """Cognitive prediction based on real-time layer data."""
        if not self.is_trained:
            return "UNTRAINED", 0.0
            
        # Convert dict to array
        feat = np.array([[
            current_metrics['microComp'],
            current_metrics['operSlope'],
            current_metrics['layerSep'],
            current_metrics['effort'] / (current_metrics['result'] + 1e-6),
            current_metrics.get('sep_velocity', 0),
            current_metrics.get('slope_accel', 0)
        ]])
        
        feat_scaled = self.scaler.transform(feat)
        pred = self.model.predict(feat_scaled)[0]
        proba = self.model.predict_proba(feat_scaled)[0][pred]
        
        return self.regimes[pred], proba

if __name__ == "__main__":
    ai = MultifractalAI()
    print("Multifractal AI Engine initialized.")
    # In a real scenario, this would load the CSV exported by MT5
    # ai.train_on_layers('UnifiedModel_Export_XAUUSD.csv')
