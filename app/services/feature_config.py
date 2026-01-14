"""
MODULO 1 FASE 14: Feature Vector Configurator
Permite activar/desactivar features basado en el análisis de importancia.

Configuración óptima basada en Feature Importance:
- L1_Mean: ACTIVE (contribuye)
- L1_Width: INACTIVE (weak - remove)
- L4_Slope: ACTIVE (top predictor)
- Div_L1_L2: INACTIVE (weak - remove) 
- Vol_ZScore: ACTIVE (volumen anomaly detection)

Vector optimizado: 3D [L1_Mean, L4_Slope, Vol_ZScore]
"""
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger("feat.feature_config")

class FeatureVectorConfig:
    """
    Configurador centralizado del vector de features.
    Permite optimización dinámica basada en análisis de importancia.
    """
    
    # Default configuration (all features active)
    DEFAULT_CONFIG = {
        "L1_Mean": {"active": True, "index": 0, "description": "Micro price mean"},
        "L1_Width": {"active": True, "index": 1, "description": "Micro volatility"},
        "L4_Slope": {"active": True, "index": 2, "description": "Gravitational slope"},
        "Div_L1_L2": {"active": True, "index": 3, "description": "Layer divergence"},
        "Vol_ZScore": {"active": True, "index": 4, "description": "Volume anomaly"}
    }
    
    # Optimized configuration based on Feature Importance analysis
    OPTIMIZED_CONFIG = {
        "L1_Mean": {"active": True, "index": 0, "description": "Micro price mean"},
        "L1_Width": {"active": False, "index": 1, "description": "Micro volatility (WEAK)"},
        "L4_Slope": {"active": True, "index": 2, "description": "Gravitational slope (TOP)"},
        "Div_L1_L2": {"active": False, "index": 3, "description": "Layer divergence (WEAK)"},
        "Vol_ZScore": {"active": True, "index": 4, "description": "Volume anomaly"}
    }
    
    def __init__(self, use_optimized: bool = True):
        """
        Initialize with default or optimized configuration.
        Can be overridden via environment variable FEAT_VECTOR_CONFIG.
        """
        env_config = os.getenv("FEAT_VECTOR_CONFIG", "OPTIMIZED" if use_optimized else "DEFAULT")
        
        if env_config == "OPTIMIZED":
            self.config = self.OPTIMIZED_CONFIG.copy()
        else:
            self.config = self.DEFAULT_CONFIG.copy()
            
        self._update_indices()
        logger.info(f"📊 Feature Vector Config: {self.get_active_names()}")
    
    def _update_indices(self):
        """Recalculate indices for active features only."""
        active_idx = 0
        for name, cfg in self.config.items():
            if cfg["active"]:
                cfg["optimized_index"] = active_idx
                active_idx += 1
            else:
                cfg["optimized_index"] = -1
    
    def is_active(self, feature_name: str) -> bool:
        """Check if a feature is active."""
        return self.config.get(feature_name, {}).get("active", False)
    
    def get_active_names(self) -> List[str]:
        """Get list of active feature names."""
        return [name for name, cfg in self.config.items() if cfg["active"]]
    
    def get_active_count(self) -> int:
        """Get number of active features."""
        return len(self.get_active_names())
    
    def filter_vector(self, full_vector: List[float]) -> List[float]:
        """
        Filter a full 5D vector to only include active features.
        
        Args:
            full_vector: [L1_Mean, L1_Width, L4_Slope, Div_L1_L2, Vol_ZScore]
            
        Returns:
            Filtered vector with only active features.
        """
        if len(full_vector) != 5:
            logger.warning(f"Expected 5D vector, got {len(full_vector)}D")
            return full_vector
            
        filtered = []
        for i, (name, cfg) in enumerate(self.config.items()):
            if cfg["active"]:
                filtered.append(full_vector[i])
        
        return filtered
    
    def set_active(self, feature_name: str, active: bool):
        """Dynamically enable/disable a feature."""
        if feature_name in self.config:
            self.config[feature_name]["active"] = active
            self._update_indices()
            logger.info(f"Feature '{feature_name}' set to {'ACTIVE' if active else 'INACTIVE'}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "total_features": len(self.config),
            "active_features": self.get_active_count(),
            "active_names": self.get_active_names(),
            "details": {name: {"active": cfg["active"], "desc": cfg["description"]} 
                       for name, cfg in self.config.items()}
        }

# Global singleton - uses optimized config by default
feature_config = FeatureVectorConfig(use_optimized=True)

def test_feature_config():
    """Test the feature configuration."""
    print("=" * 60)
    print("📊 FEAT SYSTEM - MODULE 1 PHASE 14: FEATURE CONFIG TEST")
    print("=" * 60)
    
    fc = FeatureVectorConfig(use_optimized=True)
    
    print(f"\n📋 Configuration Summary:")
    summary = fc.get_config_summary()
    print(f"   Total Features: {summary['total_features']}")
    print(f"   Active Features: {summary['active_features']}")
    print(f"   Active Names: {summary['active_names']}")
    
    print(f"\n📊 Feature Details:")
    for name, details in summary['details'].items():
        status = "✅" if details["active"] else "❌"
        print(f"   {status} {name}: {details['desc']}")
    
    # Test filtering
    full_vector = [1850.5, 0.75, 0.025, 1.002, 1.5]
    filtered = fc.filter_vector(full_vector)
    print(f"\n🔄 Vector Filtering:")
    print(f"   Input (5D): {full_vector}")
    print(f"   Output ({len(filtered)}D): {filtered}")

if __name__ == "__main__":
    test_feature_config()
