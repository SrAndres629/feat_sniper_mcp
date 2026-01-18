from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

class ArtifactType(Enum):
    PYTORCH = "pytorch"
    JOBLIB = "joblib"
    UNKNOWN = "unknown"

@dataclass
class ValidationResult:
    path: str
    artifact_type: ArtifactType
    is_valid: bool
    file_size_bytes: int = 0
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
