import os
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .models import ArtifactType, ValidationResult
from .exceptions import ArtifactIntegrityError

logger = logging.getLogger("SystemGuard.Artifacts")

class ModelGuardian:
    _instance: Optional['ModelGuardian'] = None
    
    def __new__(cls) -> 'ModelGuardian':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
        self._initialized = True
        self._validation_cache: Dict[str, ValidationResult] = {}

    def _detect_type(self, path: str) -> ArtifactType:
        ext = Path(path).suffix.lower()
        if ext in (".pt", ".pth"): return ArtifactType.PYTORCH
        if ext in (".joblib", ".pkl"): return ArtifactType.JOBLIB
        return ArtifactType.UNKNOWN

    def calculate_checksum(self, path: str) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): sha.update(chunk)
        return sha.hexdigest()
    
    def validate_existence(self, path: str) -> Tuple[bool, Optional[str]]:
        if not os.path.exists(path): return False, "File not found"
        if os.path.getsize(path) == 0: return False, "Empty file"
        return True, None

    def validate_pytorch(self, path: str, weights_only: bool = True) -> ValidationResult:
        res = ValidationResult(path=path, artifact_type=ArtifactType.PYTORCH, is_valid=False)
        ex, err = self.validate_existence(path)
        if not ex:
            res.error_type, res.error_details = ("NOT_FOUND" if "found" in err else "EMPTY"), err
            return res
        res.file_size_bytes = os.path.getsize(path)
        try:
            import torch
            try: data = torch.load(path, map_location="cpu", weights_only=weights_only)
            except: data = torch.load(path, map_location="cpu")
            if isinstance(data, dict): res.metadata["keys"] = list(data.keys())[:10]
            res.is_valid, res.metadata["sha256"] = True, self.calculate_checksum(path)
        except Exception as e:
            res.error_type, res.error_details = "CORRUPTED", str(e)
        return res

    def validate_joblib(self, path: str) -> ValidationResult:
        res = ValidationResult(path, ArtifactType.JOBLIB, False)
        ex, err = self.validate_existence(path)
        if not ex:
            res.error_type, res.error_details = ("NOT_FOUND" if "found" in err else "EMPTY"), err
            return res
        res.file_size_bytes = os.path.getsize(path)
        try:
            import joblib
            data = joblib.load(path)
            res.is_valid, res.metadata["sha256"] = True, self.calculate_checksum(path)
        except Exception as e:
            res.error_type, res.error_details = "CORRUPTED", str(e)
        return res

    def validate(self, path: str, use_cache: bool = True) -> ValidationResult:
        if use_cache and path in self._validation_cache: return self._validation_cache[path]
        t = self._detect_type(path)
        if t == ArtifactType.PYTORCH: res = self.validate_pytorch(path)
        elif t == ArtifactType.JOBLIB: res = self.validate_joblib(path)
        else: res = ValidationResult(path, ArtifactType.UNKNOWN, False, error_type="UNKNOWN")
        self._validation_cache[path] = res
        return res

    def validate_all(self, paths: List[str], raise_on_failure: bool = True) -> Dict[str, ValidationResult]:
        results = {}
        for p in paths:
            r = self.validate(p)
            results[p] = r
            if not r.is_valid and raise_on_failure:
                raise ArtifactIntegrityError(r.path, r.error_type or "ERR", r.error_details or "Failed")
        return results

def validate_model_artifacts(models_dir: str = "models") -> bool:
    if not os.path.exists(models_dir): return True
    paths = []
    for ext in ["*.pt", "*.pth", "*.joblib", "*.pkl"]: paths.extend(Path(models_dir).glob(ext))
    if not paths: return True
    ModelGuardian().validate_all([str(p) for p in paths], raise_on_failure=True)
    return True
