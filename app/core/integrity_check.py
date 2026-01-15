"""
INTEGRITY CHECK - ModelGuardian
===============================
Validación de artefactos de modelo con grado institucional.

[SENIOR ML OPS] Este módulo es el primer filtro de seguridad antes de iniciar
la inferencia. Previene crashes en caliente por modelos corruptos.

Features:
- Validación de existencia (no archivos vacíos)
- Deep Check: Carga real de .pt y .joblib para detectar corrupción
- Excepciones explícitas con diagnóstico detallado
- Optimización con weights_only=True para seguridad
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("feat.core.integrity_check")


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ArtifactIntegrityError(Exception):
    """
    Excepción lanzada cuando un artefacto de modelo falla la validación.
    
    Atributos:
        path: Ruta del archivo que falló
        error_type: Tipo de error (NOT_FOUND, EMPTY, CORRUPTED, VERSION_MISMATCH)
        details: Mensaje detallado del diagnóstico
    """
    def __init__(self, path: str, error_type: str, details: str):
        self.path = path
        self.error_type = error_type
        self.details = details
        super().__init__(f"[{error_type}] {path}: {details}")


class ArtifactType(Enum):
    """Tipos de artefactos soportados."""
    PYTORCH = "pytorch"
    JOBLIB = "joblib"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Resultado de validación de un artefacto."""
    path: str
    artifact_type: ArtifactType
    is_valid: bool
    file_size_bytes: int = 0
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# MODEL GUARDIAN - Singleton Pattern
# =============================================================================

class ModelGuardian:
    """
    Guardian de Integridad de Artefactos ML.
    
    Patron Singleton para garantizar validación consistente en todo el sistema.
    
    ¿Por qué validación de integridad vs simple validación de ruta?
    ---------------------------------------------------------------
    Una validación de ruta solo confirma que el archivo existe. Sin embargo:
    1. El archivo puede estar vacío (0 bytes) - fallo de escritura interrumpida
    2. El archivo puede estar corrupto - descarga parcial, disco dañado
    3. El archivo puede ser incompatible - versión diferente de PyTorch/Joblib
    
    La validación de integridad CARGA el archivo para confirmar que es usable,
    detectando estos problemas ANTES de que el sistema intente hacer inferencia
    durante un tick de alta volatilidad.
    """
    
    _instance: Optional['ModelGuardian'] = None
    
    def __new__(cls) -> 'ModelGuardian':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._validation_cache: Dict[str, ValidationResult] = {}
        logger.info("[GUARDIAN] ModelGuardian initialized - Artifact Integrity Protection ACTIVE")
    
    def _detect_type(self, path: str) -> ArtifactType:
        """Detecta el tipo de artefacto por extensión."""
        ext = Path(path).suffix.lower()
        if ext == ".pt" or ext == ".pth":
            return ArtifactType.PYTORCH
        elif ext == ".joblib" or ext == ".pkl":
            return ArtifactType.JOBLIB
        return ArtifactType.UNKNOWN
    
    def validate_existence(self, path: str) -> Tuple[bool, Optional[str]]:
        """
        Validación de existencia: archivo existe y no está vacío.
        
        Returns:
            (is_valid, error_message)
        """
        if not os.path.exists(path):
            return False, f"File not found: {path}"
        
        file_size = os.path.getsize(path)
        if file_size == 0:
            return False, f"Empty file (0 bytes): {path}"
        
        return True, None
    
    def validate_pytorch(self, path: str, weights_only: bool = True) -> ValidationResult:
        """
        Deep Check para archivos PyTorch (.pt, .pth).
        
        Args:
            path: Ruta al archivo de modelo
            weights_only: Si True, usa torch.load con weights_only=True para seguridad
                         Esto previene ejecución de código arbitrario en archivos maliciosos
        
        Returns:
            ValidationResult con estado de validación
        """
        result = ValidationResult(
            path=path,
            artifact_type=ArtifactType.PYTORCH,
            is_valid=False
        )
        
        # 1. Validación de existencia
        exists, error = self.validate_existence(path)
        if not exists:
            result.error_type = "NOT_FOUND" if "not found" in error else "EMPTY"
            result.error_details = error
            return result
        
        result.file_size_bytes = os.path.getsize(path)
        
        # 2. Deep Check - intentar cargar el modelo
        try:
            import torch
            
            # weights_only=True es más seguro pero puede fallar con modelos antiguos
            try:
                data = torch.load(path, map_location="cpu", weights_only=weights_only)
            except TypeError:
                # PyTorch < 1.13 no soporta weights_only
                data = torch.load(path, map_location="cpu")
            
            # Extraer metadata si es un state_dict con config
            if isinstance(data, dict):
                result.metadata["keys"] = list(data.keys())[:10]  # Primeras 10 keys
                if "model_config" in data:
                    result.metadata["config"] = data["model_config"]
                if "model_state" in data:
                    result.metadata["state_dict_keys"] = len(data["model_state"])
            
            result.is_valid = True
            logger.debug(f"[GUARDIAN] ✅ PyTorch artifact valid: {path}")
            
        except Exception as e:
            result.error_type = "CORRUPTED"
            result.error_details = f"Failed to load PyTorch file: {str(e)}"
            logger.error(f"[GUARDIAN] ❌ PyTorch artifact corrupted: {path} - {e}")
        
        return result
    
    def validate_joblib(self, path: str) -> ValidationResult:
        """
        Deep Check para archivos Joblib (.joblib, .pkl).
        
        Valida que el archivo pueda ser deserializado correctamente.
        Detecta incompatibilidades de versión de scikit-learn.
        
        Returns:
            ValidationResult con estado de validación
        """
        result = ValidationResult(
            path=path,
            artifact_type=ArtifactType.JOBLIB,
            is_valid=False
        )
        
        # 1. Validación de existencia
        exists, error = self.validate_existence(path)
        if not exists:
            result.error_type = "NOT_FOUND" if "not found" in error else "EMPTY"
            result.error_details = error
            return result
        
        result.file_size_bytes = os.path.getsize(path)
        
        # 2. Deep Check - intentar cargar el objeto
        try:
            import joblib
            
            data = joblib.load(path)
            
            # Extraer metadata
            result.metadata["type"] = type(data).__name__
            
            # Si es un modelo sklearn, extraer info
            if hasattr(data, "get_params"):
                result.metadata["sklearn_params"] = True
            if hasattr(data, "n_features_in_"):
                result.metadata["n_features"] = data.n_features_in_
            
            # Si es un dict (como scaler_stats), extraer keys
            if isinstance(data, dict):
                result.metadata["keys"] = list(data.keys())[:10]
            
            result.is_valid = True
            logger.debug(f"[GUARDIAN] ✅ Joblib artifact valid: {path}")
            
        except ModuleNotFoundError as e:
            result.error_type = "VERSION_MISMATCH"
            result.error_details = f"Missing module during deserialization: {str(e)}"
            logger.error(f"[GUARDIAN] ❌ Joblib version mismatch: {path} - {e}")
            
        except Exception as e:
            result.error_type = "CORRUPTED"
            result.error_details = f"Failed to load Joblib file: {str(e)}"
            logger.error(f"[GUARDIAN] ❌ Joblib artifact corrupted: {path} - {e}")
        
        return result
    
    def validate(self, path: str, use_cache: bool = True) -> ValidationResult:
        """
        Valida un artefacto detectando automáticamente su tipo.
        
        Args:
            path: Ruta al artefacto
            use_cache: Si True, retorna resultado cacheado si existe
        
        Returns:
            ValidationResult
        
        Raises:
            ArtifactIntegrityError si el artefacto es inválido
        """
        # Check cache
        if use_cache and path in self._validation_cache:
            return self._validation_cache[path]
        
        artifact_type = self._detect_type(path)
        
        if artifact_type == ArtifactType.PYTORCH:
            result = self.validate_pytorch(path)
        elif artifact_type == ArtifactType.JOBLIB:
            result = self.validate_joblib(path)
        else:
            result = ValidationResult(
                path=path,
                artifact_type=ArtifactType.UNKNOWN,
                is_valid=False,
                error_type="UNKNOWN_TYPE",
                error_details=f"Unsupported artifact type: {Path(path).suffix}"
            )
        
        # Cache result
        self._validation_cache[path] = result
        
        return result
    
    def validate_all(self, paths: List[str], raise_on_failure: bool = True) -> Dict[str, ValidationResult]:
        """
        Valida múltiples artefactos y opcionalmente lanza excepción si alguno falla.
        
        Args:
            paths: Lista de rutas a validar
            raise_on_failure: Si True, lanza ArtifactIntegrityError en el primer fallo
        
        Returns:
            Dict[path, ValidationResult]
        """
        results = {}
        
        for path in paths:
            result = self.validate(path)
            results[path] = result
            
            if not result.is_valid and raise_on_failure:
                raise ArtifactIntegrityError(
                    path=result.path,
                    error_type=result.error_type or "UNKNOWN",
                    details=result.error_details or "Validation failed"
                )
        
        return results
    
    def get_status(self) -> Dict:
        """Retorna estado del guardian para diagnóstico."""
        valid_count = sum(1 for r in self._validation_cache.values() if r.is_valid)
        invalid_count = sum(1 for r in self._validation_cache.values() if not r.is_valid)
        
        return {
            "cached_validations": len(self._validation_cache),
            "valid_artifacts": valid_count,
            "invalid_artifacts": invalid_count,
            "all_valid": invalid_count == 0 if self._validation_cache else None
        }
    
    def clear_cache(self) -> None:
        """Limpia el cache de validaciones."""
        self._validation_cache.clear()
        logger.info("[GUARDIAN] Validation cache cleared")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

model_guardian = ModelGuardian()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_model_artifacts(models_dir: str = "models") -> bool:
    """
    Función de conveniencia para validar todos los artefactos en un directorio.
    
    Uso típico en nexus.bat:
        python -c "from app.core.integrity_check import validate_model_artifacts; validate_model_artifacts()"
    
    Returns:
        True si todos los artefactos son válidos
    
    Raises:
        ArtifactIntegrityError si algún artefacto falla
    """
    if not os.path.exists(models_dir):
        logger.warning(f"[GUARDIAN] Models directory not found: {models_dir}")
        return True  # No hay artefactos que validar
    
    # Buscar todos los .pt y .joblib
    artifact_paths = []
    for ext in ["*.pt", "*.pth", "*.joblib", "*.pkl"]:
        artifact_paths.extend(Path(models_dir).glob(ext))
    
    if not artifact_paths:
        logger.info(f"[GUARDIAN] No artifacts found in {models_dir}")
        return True
    
    logger.info(f"[GUARDIAN] Validating {len(artifact_paths)} artifacts...")
    
    results = model_guardian.validate_all(
        [str(p) for p in artifact_paths],
        raise_on_failure=True
    )
    
    valid_count = sum(1 for r in results.values() if r.is_valid)
    logger.info(f"[GUARDIAN] ✅ All {valid_count} artifacts validated successfully")
    
    return True
