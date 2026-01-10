"""
Data Collector - Quantum Leap Phase 1
======================================
Captura de ticks/velas, construcción de vector de features X,
y etiquetado diferido por "Oracle" tras N velas.

Persistence: CSV y/o SQLite para training dataset.
"""

import os
import csv
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque

# Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DATA_DIR = os.getenv("DATA_DIR", "data")
CSV_PATH = os.path.join(DATA_DIR, "training_dataset.csv")
N_LOOKAHEAD = int(os.getenv("N_LOOKAHEAD", "10"))  # Velas hacia adelante para etiquetar
PROFIT_THRESHOLD = float(os.getenv("PROFIT_THRESHOLD", "0.002"))  # 0.2% para WIN
MAX_BUFFER = int(os.getenv("MAX_BUFFER", "10000"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QuantumLeap.DataCollector")

# Feature names for consistency
FEATURE_NAMES = [
    "close", "open", "high", "low", "volume",
    "rsi", "ema_fast", "ema_slow", "ema_spread",
    "feat_score", "fsm_state", "atr", "compression",
    "liquidity_above", "liquidity_below"
]


class OracleLabeler:
    """
    Oracle que etiqueta datos mirando N velas hacia el futuro.
    Implementa el concepto de "Ground Truth Diferida".
    """
    
    def __init__(self, lookahead: int = N_LOOKAHEAD, threshold: float = PROFIT_THRESHOLD):
        self.lookahead = lookahead
        self.threshold = threshold
        self.buffer: List[Dict] = []
        
    def add_sample(self, features: Dict[str, float], close_price: float, timestamp: str):
        """Añade muestra al buffer para etiquetado futuro."""
        self.buffer.append({
            "timestamp": timestamp,
            "features": features,
            "close": close_price,
            "label": None  # Pending
        })
        
    def process_labels(self) -> List[Dict]:
        """
        Procesa etiquetas para muestras que ya tienen suficiente historia futura.
        
        Returns:
            Lista de muestras completamente etiquetadas listas para guardar.
        """
        labeled = []
        
        # Solo podemos etiquetar hasta buffer_size - lookahead
        max_index = len(self.buffer) - self.lookahead
        
        for i in range(max_index):
            if self.buffer[i]["label"] is not None:
                continue  # Ya etiquetado
                
            entry_price = self.buffer[i]["close"]
            future_price = self.buffer[i + self.lookahead]["close"]
            
            # Calcular PnL porcentual
            pnl = (future_price - entry_price) / entry_price
            
            # Etiquetar: 1 = WIN (profit > threshold), 0 = LOSS
            label = 1 if pnl > self.threshold else 0
            
            self.buffer[i]["label"] = label
            labeled.append(self.buffer[i])
            
        # Limpiar muestras ya procesadas
        self.buffer = [s for s in self.buffer if s["label"] is None]
        
        return labeled


class DataCollector:
    """
    Colector principal de datos para ML training.
    
    Funciones:
    - Captura features de cada vela
    - Etiqueta con Oracle (lookahead)
    - Persiste en CSV
    """
    
    def __init__(self, csv_path: str = CSV_PATH):
        self.csv_path = csv_path
        self.oracle = OracleLabeler()
        self.samples_collected = 0
        self._ensure_csv_header()
        
    def _ensure_csv_header(self):
        """Crea archivo CSV con header si no existe."""
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        
        if not os.path.exists(self.csv_path):
            fieldnames = ["timestamp", "symbol"] + FEATURE_NAMES + ["label"]
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            logger.info(f"Created training dataset: {self.csv_path}")
            
    def compute_features(self, candle: Dict, indicators: Dict) -> Dict[str, float]:
        """
        Construye vector de features X desde vela e indicadores.
        
        Args:
            candle: Dict con open, high, low, close, volume
            indicators: Dict con RSI, EMAs, FEAT score, FSM state, etc.
            
        Returns:
            Dict con features normalizadas
        """
        features = {
            # Price Action
            "close": float(candle.get("close", 0)),
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "volume": float(candle.get("volume", 0)),
            
            # Technical Indicators
            "rsi": float(indicators.get("rsi", 50.0)),
            "ema_fast": float(indicators.get("ema_fast", candle.get("close", 0))),
            "ema_slow": float(indicators.get("ema_slow", candle.get("close", 0))),
            "ema_spread": float(
                indicators.get("ema_fast", 0) - indicators.get("ema_slow", 0)
            ),
            
            # FEAT System
            "feat_score": float(indicators.get("feat_score", 0.0)),
            "fsm_state": float(indicators.get("fsm_state", 0)),  # Encoded as int
            "atr": float(indicators.get("atr", 0.001)),
            "compression": float(indicators.get("compression", 0.5)),
            
            # Liquidity
            "liquidity_above": float(indicators.get("liquidity_above", 0)),
            "liquidity_below": float(indicators.get("liquidity_below", 0))
        }
        
        return features
        
    def collect(self, symbol: str, candle: Dict, indicators: Dict):
        """
        Añade nueva muestra al sistema de recolección.
        
        Args:
            symbol: Símbolo del activo
            candle: Datos OHLCV
            indicators: Indicadores técnicos y FEAT
        """
        features = self.compute_features(candle, indicators)
        timestamp = datetime.utcnow().isoformat()
        
        # Añadir al Oracle para etiquetado diferido
        self.oracle.add_sample(
            features=features,
            close_price=features["close"],
            timestamp=timestamp
        )
        
        # Procesar etiquetas maduras
        labeled_samples = self.oracle.process_labels()
        
        # Persistir muestras etiquetadas
        for sample in labeled_samples:
            row = {
                "timestamp": sample["timestamp"],
                "symbol": symbol,
                **sample["features"],
                "label": sample["label"]
            }
            self._append_row(row)
            self.samples_collected += 1
            
        if self.samples_collected > 0 and self.samples_collected % 100 == 0:
            logger.info(f"Collected {self.samples_collected} labeled samples")
            
    def _append_row(self, row: Dict):
        """Añade fila al CSV."""
        fieldnames = ["timestamp", "symbol"] + FEATURE_NAMES + ["label"]
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            
    def get_stats(self) -> Dict:
        """Retorna estadísticas de recolección."""
        return {
            "samples_collected": self.samples_collected,
            "buffer_size": len(self.oracle.buffer),
            "csv_path": self.csv_path,
            "lookahead": self.oracle.lookahead,
            "threshold": self.oracle.threshold
        }


# Singleton instance
data_collector = DataCollector()


# Async wrapper for MCP integration
async def collect_sample(symbol: str, candle: Dict, indicators: Dict) -> Dict:
    """MCP-compatible async wrapper."""
    data_collector.collect(symbol, candle, indicators)
    return data_collector.get_stats()
