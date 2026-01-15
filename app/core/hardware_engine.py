"""
HARDWARE ENGINE - Compute Environment Diagnostics
=================================================
Diagnóstico de hardware y configuración óptima del entorno de cómputo.

[SENIOR SYSTEMS ARCHITECT] Este módulo detecta la presencia de aceleración GPU
y configura el entorno de ejecución óptimo para las redes neuronales híbridas.

Features:
- Detección CUDA con extracción de metadatos GPU
- Fallback inteligente a CPU con advertencias
- Configuración automática de DEVICE global
- Optimización de cuDNN para CNN/LSTM
- Ficha técnica de hardware al arrancar
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("feat.core.hardware_engine")


# =============================================================================
# COMPUTE MODES
# =============================================================================

class ComputeMode(Enum):
    """Modos de cómputo disponibles."""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon


@dataclass
class GPUInfo:
    """Información detallada de la GPU."""
    name: str
    compute_capability: Tuple[int, int]
    total_memory_mb: int
    available_memory_mb: int
    cuda_version: str
    cudnn_version: Optional[str]
    driver_version: str
    

@dataclass
class HardwareProfile:
    """Perfil completo de hardware del sistema."""
    compute_mode: ComputeMode
    device_name: str
    gpu_info: Optional[GPUInfo]
    cpu_cores: int
    ram_total_mb: int
    ram_available_mb: int
    is_optimized: bool
    warnings: list


# =============================================================================
# HARDWARE ENGINE - Singleton
# =============================================================================

class HardwareEngine:
    """
    Motor de Diagnóstico de Hardware para FEAT Sniper.
    
    ¿Por qué es crítico el diagnóstico de hardware en HFT?
    ------------------------------------------------------
    1. Latencia de Inferencia: GPU puede reducir latencia de 10ms a <1ms
    2. VRAM Leakage: Sin monitoreo, el RAG puede llenar la memoria
    3. Determinismo: El sistema debe saber exactamente con qué recursos cuenta
    4. Fallback Seguro: Si GPU falla, el sistema debe ajustar thresholds
    """
    
    _instance: Optional['HardwareEngine'] = None
    _profile: Optional[HardwareProfile] = None
    
    # Constantes de ajuste
    CPU_CONFIDENCE_PENALTY: float = 0.10  # +10% threshold en CPU
    
    def __new__(cls) -> 'HardwareEngine':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._device = None
        self._torch_available = False
        
        # Intentar import de torch
        try:
            import torch
            self._torch_available = True
        except ImportError:
            logger.warning("[HARDWARE] PyTorch not available - running in degraded mode")
        
        logger.info("[HARDWARE] HardwareEngine initialized - starting diagnostics...")
    
    def detect_cuda(self) -> Tuple[bool, Optional[GPUInfo]]:
        """
        Detecta disponibilidad de CUDA y extrae metadatos de GPU.
        
        Returns:
            (cuda_available, gpu_info)
        """
        if not self._torch_available:
            return False, None
        
        import torch
        
        if not torch.cuda.is_available():
            return False, None
        
        try:
            device_idx = 0
            props = torch.cuda.get_device_properties(device_idx)
            
            # Memoria
            total_mem = props.total_memory // (1024 * 1024)  # MB
            # Memoria disponible (aproximada)
            torch.cuda.empty_cache()
            available_mem = (props.total_memory - torch.cuda.memory_allocated(device_idx)) // (1024 * 1024)
            
            # Versiones
            cuda_version = torch.version.cuda or "unknown"
            cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            
            # Driver version (si disponible)
            try:
                driver_version = torch.cuda.get_device_capability(device_idx)
                driver_str = f"CC {driver_version[0]}.{driver_version[1]}"
            except:
                driver_str = "unknown"
            
            gpu_info = GPUInfo(
                name=props.name,
                compute_capability=(props.major, props.minor),
                total_memory_mb=total_mem,
                available_memory_mb=available_mem,
                cuda_version=cuda_version,
                cudnn_version=cudnn_version,
                driver_version=driver_str
            )
            
            return True, gpu_info
            
        except Exception as e:
            logger.error(f"[HARDWARE] Failed to get GPU info: {e}")
            return True, None  # CUDA disponible pero sin info detallada
    
    def detect_mps(self) -> bool:
        """Detecta MPS (Apple Silicon)."""
        if not self._torch_available:
            return False
        
        import torch
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    def get_system_info(self) -> Tuple[int, int, int]:
        """
        Obtiene información básica del sistema.
        
        Returns:
            (cpu_cores, ram_total_mb, ram_available_mb)
        """
        import multiprocessing
        
        cpu_cores = multiprocessing.cpu_count()
        
        # RAM info
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram_total = mem.total // (1024 * 1024)
            ram_available = mem.available // (1024 * 1024)
        except ImportError:
            # Fallback sin psutil
            ram_total = 0
            ram_available = 0
        
        return cpu_cores, ram_total, ram_available
    
    def configure_device(self) -> HardwareProfile:
        """
        Configura el DEVICE global y retorna el perfil de hardware.
        
        Esta función debe llamarse UNA VEZ al inicio del sistema.
        """
        import torch
        
        warnings = []
        
        # 1. Detectar hardware
        cuda_available, gpu_info = self.detect_cuda()
        mps_available = self.detect_mps()
        cpu_cores, ram_total, ram_available = self.get_system_info()
        
        # 2. Seleccionar compute mode
        if cuda_available:
            compute_mode = ComputeMode.CUDA
            device_name = gpu_info.name if gpu_info else "CUDA GPU"
            
            # Optimizaciones CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("[HARDWARE] 🎮 CUDA detected - cuDNN benchmark ENABLED")
            
        elif mps_available:
            compute_mode = ComputeMode.MPS
            device_name = "Apple Silicon (MPS)"
            logger.info("[HARDWARE] 🍎 MPS detected - Apple Silicon acceleration")
            
        else:
            compute_mode = ComputeMode.CPU
            device_name = f"CPU ({cpu_cores} cores)"
            warnings.append(
                "⚠️ Running in CPU MODE. Inference latency will increase by ~5-10ms. "
                "Consider adjusting confidence threshold by +10% to compensate for slippage."
            )
            logger.warning("[HARDWARE] ⚠️ No GPU detected - Running in CPU MODE")
        
        # 3. Configurar DEVICE global
        self._device = torch.device(compute_mode.value)
        
        # 4. Configurar default tensor type
        if compute_mode == ComputeMode.CUDA:
            torch.set_default_dtype(torch.float32)
        
        # 5. Verificar VRAM suficiente (mínimo 2GB recomendado)
        if gpu_info and gpu_info.available_memory_mb < 2048:
            warnings.append(
                f"⚠️ Low VRAM: {gpu_info.available_memory_mb}MB available. "
                "Consider closing other GPU applications to avoid OOM."
            )
        
        # 6. Crear perfil
        self._profile = HardwareProfile(
            compute_mode=compute_mode,
            device_name=device_name,
            gpu_info=gpu_info,
            cpu_cores=cpu_cores,
            ram_total_mb=ram_total,
            ram_available_mb=ram_available,
            is_optimized=compute_mode == ComputeMode.CUDA,
            warnings=warnings
        )
        
        return self._profile
    
    def get_device(self) -> 'torch.device':
        """Retorna el DEVICE configurado."""
        if self._device is None:
            self.configure_device()
        return self._device
    
    def get_profile(self) -> Optional[HardwareProfile]:
        """Retorna el perfil de hardware."""
        return self._profile
    
    def get_confidence_adjustment(self) -> float:
        """
        Retorna el ajuste de confianza basado en el modo de cómputo.
        
        En CPU, se recomienda aumentar el threshold de confianza
        para compensar la mayor latencia de inferencia.
        
        Returns:
            Valor a SUMAR al threshold de confianza (0.0 para GPU, 0.10 para CPU)
        """
        if self._profile is None:
            return 0.0
        
        if self._profile.compute_mode == ComputeMode.CPU:
            return self.CPU_CONFIDENCE_PENALTY
        
        return 0.0
    
    def print_hardware_card(self) -> None:
        """
        Imprime una ficha técnica del hardware al arrancar.
        """
        if self._profile is None:
            self.configure_device()
        
        p = self._profile
        
        print("\n" + "=" * 60)
        print("  FEAT SNIPER - Hardware Profile")
        print("=" * 60)
        print(f"  Compute Mode : {p.compute_mode.value.upper()}")
        print(f"  Device       : {p.device_name}")
        
        if p.gpu_info:
            g = p.gpu_info
            print(f"  CUDA Version : {g.cuda_version}")
            print(f"  Compute Cap  : {g.compute_capability[0]}.{g.compute_capability[1]}")
            print(f"  VRAM Total   : {g.total_memory_mb:,} MB")
            print(f"  VRAM Free    : {g.available_memory_mb:,} MB")
            if g.cudnn_version:
                print(f"  cuDNN        : {g.cudnn_version}")
        
        print(f"  CPU Cores    : {p.cpu_cores}")
        print(f"  RAM Total    : {p.ram_total_mb:,} MB")
        print(f"  RAM Free     : {p.ram_available_mb:,} MB")
        print(f"  Optimized    : {'YES ✅' if p.is_optimized else 'NO ⚠️'}")
        
        if p.warnings:
            print("-" * 60)
            for w in p.warnings:
                print(f"  {w}")
        
        print("=" * 60 + "\n")
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado para diagnóstico."""
        if self._profile is None:
            return {"initialized": False}
        
        p = self._profile
        return {
            "initialized": True,
            "compute_mode": p.compute_mode.value,
            "device_name": p.device_name,
            "is_optimized": p.is_optimized,
            "confidence_adjustment": self.get_confidence_adjustment(),
            "gpu_info": {
                "name": p.gpu_info.name,
                "vram_mb": p.gpu_info.total_memory_mb,
                "cuda_version": p.gpu_info.cuda_version
            } if p.gpu_info else None,
            "warnings_count": len(p.warnings)
        }


# =============================================================================
# SINGLETON INSTANCE & GLOBAL DEVICE
# =============================================================================

hardware_engine = HardwareEngine()

# Conveniencia: DEVICE global
def get_device() -> 'torch.device':
    """Retorna el DEVICE global configurado."""
    return hardware_engine.get_device()

def get_confidence_adjustment() -> float:
    """Retorna el ajuste de confianza para compensar latencia CPU."""
    return hardware_engine.get_confidence_adjustment()


# =============================================================================
# STARTUP FUNCTION
# =============================================================================

def initialize_hardware(print_card: bool = True) -> HardwareProfile:
    """
    Función de inicialización de hardware.
    
    Uso típico en mcp_server.py:
        from app.core.hardware_engine import initialize_hardware
        profile = initialize_hardware()
    
    Returns:
        HardwareProfile con información del sistema
    """
    profile = hardware_engine.configure_device()
    
    if print_card:
        hardware_engine.print_hardware_card()
    
    return profile
