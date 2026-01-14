No tengo permisos de escritura en este entorno para crear archivos directamente. Sin embargo, he generado el código completo para `audit_alpha_causality.py`.

Este script realiza lo siguiente:
1.  **Carga el modelo**: Intenta cargar `models/lstm_BTCUSD_v2.pt`.
2.  **Procesa los datos**: Carga `data/training_dataset.csv`, calcula las "Capas FEAT" (`L1_Mean`, etc.) usando `app.skills.indicators` y extrae un lote reciente.
3.  **Verifica integridad**: Confirma que no haya `NaN` ni `Inf` y que las dimensiones sean correctas (4 características).
4.  **Detecta Drift**: Busca `models/scaler_stats.json` (ubicación asumida) para comparar la media del lote actual con las estadísticas originales. Si la desviación (Z-Score) es > 3, reporta "State Drift".

Por favor, crea un archivo llamado `audit_alpha_causality.py` en la raíz del proyecto con el siguiente contenido:

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
import logging

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlphaCausality")

# Asegurar que la raíz del proyecto está en el path
sys.path.append(os.getcwd())

try:
    from app.skills.indicators import calculate_feat_layers
except ImportError:
    logger.error("No se pudo importar calculate_feat_layers. Ejecuta el script desde la raíz del proyecto.")
    sys.exit(1)

def audit_alpha_causality():
    model_path = "models/lstm_BTCUSD_v2.pt"
    data_path = "data/training_dataset.csv"
    scaler_path = "models/scaler_stats.json" # Asumimos esta ruta para los stats

    logger.info("=== INICIANDO AUDITORÍA ALPHA CAUSALITY ===")

    # 1. Verificación del Modelo
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ Archivo del modelo no encontrado: {model_path}")
        # Continuamos para auditar los datos
    else:
        try:
            model = torch.load(model_path)
            logger.info(f"✅ Modelo cargado: {model_path}")
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {e}")

    # 2. Carga y Procesamiento de Datos
    if not os.path.exists(data_path):
        logger.error(f"❌ Dataset no encontrado: {data_path}")
        return

    logger.info(f"Cargando datos de {data_path}...")
    df = pd.read_csv(data_path)
    
    if df.empty:
        logger.error("❌ El dataset está vacío.")
        return

    # Calcular Features (L1_Mean, L1_Width, L4_Slope, Div_L1_L2)
    logger.info("Calculando capas FEAT...")
    physics_df = calculate_feat_layers(df)
    
    # Unir con el dataframe original para tener contexto si es necesario, 
    # pero aquí solo necesitamos las features calculadas.
    # Tomamos un lote reciente (ej. últimos 64 ticks)
    batch_size = 64
    if len(physics_df) < batch_size:
        logger.warning(f"⚠️ Datos insuficientes para lote completo. Usando {len(physics_df)} muestras.")
    
    batch = physics_df.iloc[-batch_size:].copy()
    
    # 3. Verificación de Dimensiones y Columnas
    required_cols = ['L1_Mean', 'L1_Width', 'L4_Slope', 'Div_L1_L2']
    missing_cols = [c for c in required_cols if c not in batch.columns]
    
    if missing_cols:
        logger.error(f"❌ Faltan columnas requeridas: {missing_cols}")
        return

    # Verificar Dimensión (1, 4) por muestra (implícito al tener 4 columnas)
    logger.info(f"✅ Dimensión de entrada verificada: (N, {len(required_cols)}) -> Cumple requisito (1, 4) por vector.")

    # 4. Verificación de Integridad (NaN / Inf)
    if batch[required_cols].isna().any().any():
        logger.error("❌ CRITICAL: Valores NaN detectados en el vector de entrada.")
        return
    
    if np.isinf(batch[required_cols]).any().any():
        logger.error("❌ CRITICAL: Valores Infinitos detectados en el vector de entrada.")
        return
        
    logger.info("✅ Integridad de Datos: PASSED (Sin NaN/Inf)")

    # 5. Verificación contra Scaler Stats (State Drift)
    if not os.path.exists(scaler_path):
        logger.warning(f"⚠️ No se encontró {scaler_path}. Omitiendo verificación de Drift.")
        logger.info("NOTA: Asegúrate de generar 'scaler_stats.json' con {'feature': {'mean': x, 'std': y}}.")
        return

    try:
        with open(scaler_path, 'r') as f:
            scaler_stats = json.load(f)
        
        logger.info(f"Verificando Drift contra {scaler_path}...")
        drift_detected = False
        
        for col in required_cols:
            if col not in scaler_stats:
                continue
                
            stats = scaler_stats[col]
            ref_mean = stats.get('mean', 0.0)
            ref_std = stats.get('std', 1.0)
            
            # Media del lote actual
            current_mean = batch[col].mean()
            
            # Cálculo de Z-Score para detectar desviación
            # Evitamos división por cero
            std_safe = ref_std if ref_std != 0 else 1e-6
            z_score = abs(current_mean - ref_mean) / std_safe
            
            if z_score > 3.0:
                logger.warning(f"⚠️ STATE DRIFT DETECTED en {col}: Z-Score {z_score:.2f} (Ref: {ref_mean:.3f}, Actual: {current_mean:.3f})")
                drift_detected = True
            else:
                logger.info(f"   {col}: Estable (Z={z_score:.2f})")
        
        if drift_detected:
            logger.warning("🚨 AUDIT RESULT: STATE DRIFT DETECTED")
        else:
            logger.info("✅ AUDIT RESULT: SISTEMA ESTABLE (No Drift)")
            
    except Exception as e:
        logger.error(f"❌ Error leyendo scaler_stats: {e}")

if __name__ == "__main__":
    audit_alpha_causality()