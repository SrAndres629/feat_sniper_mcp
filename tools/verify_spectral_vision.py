
import logging
import os
import sys
import argparse
import pandas as pd
import numpy as np

# Adjust path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from nexus_core.kinetic_engine import kinetic_engine, SpectralMechanics

def create_spectral_scenario(
    scenario_name: str, 
    n_candles: int = 100, 
    base_price: float = 2000.0,
    regime: str = "BULL_TREND"
):
    """
    Simulates price action to create specific spectral signatures.
    """
    print(f"\nGenerando Escenario: {scenario_name} ({regime})")
    
    dates = pd.date_range(start="2025-01-01", periods=n_candles, freq="1min")
    close = []
    high = []
    low = []
    
    curr = base_price
    
    # 1. Base Trend Logic
    trend_factor = 0.0
    noise_factor = 1.0
    
    if regime == "BULL_TREND":
        trend_factor = 1.5
    elif regime == "PULLBACK_IN_BULL":
        trend_factor = 1.5
    elif regime == "BEAR_COLLAPSE":
        trend_factor = -2.0
        
    for i in range(n_candles):
        # Default movement
        move = np.random.normal(0, 1.0) * noise_factor + trend_factor
        
        # Pullback Logic: Last 20 candles drop against trend
        if regime == "PULLBACK_IN_BULL" and i > n_candles - 20:
             move = np.random.normal(0, 1.0) - 2.0 # Strong drop
             
        # Collapse Logic: Acceleration
        if regime == "BEAR_COLLAPSE" and i > n_candles - 30:
             move -= (i - (n_candles - 30)) * 0.1 # Accelerating drop
             
        curr += move
        close.append(curr)
        high.append(curr + abs(np.random.normal(0, 2.0)))
        low.append(curr - abs(np.random.normal(0, 2.0)))
        
    df = pd.DataFrame({
        "open": [close[0]] + close[:-1], # Simple shift for open
        "close": close,
        "high": high,
        "low": low,
        "volume": np.random.randint(100, 1000, n_candles)
    }, index=dates)
    
    return df

def audit_spectral_vision(df: pd.DataFrame, scenario_name: str):
    """
    Runs the Spectral Mechanics engine on the DF and reporting the "Vision".
    """
    metrics = kinetic_engine.compute_kinetic_state(df)
    
    print(f"--- REPORTE ESPECTRAL: {scenario_name} ---")
    
    # 1. Integrity Report (The Color Palette)
    # Scale -1.0 (Bear) to 1.0 (Bull)
    
    def get_color(score):
        if score > 0.8: return "🟢 BRILLANTE (Full Bull)"
        if score > 0.3: return "🟡 VERDOSO (Weak Bull)"
        if score > -0.3: return "⚪ GRIS (Caos/Rango)"
        if score > -0.8: return "🟠 NARANJA (Weak Bear)"
        return "🔴 ROJO (Full Bear)"
        
    m = metrics.get('integrity_micro', 0)
    s = metrics.get('integrity_structure', 0)
    M = metrics.get('integrity_macro', 0)
    
    print(f"1. MICRO LAYER (Gas):      {m:.2f} -> {get_color(m)}")
    print(f"2. OPERATIVE LAYER (Liq):  {s:.2f} -> {get_color(s)}")
    print(f"3. MACRO LAYER (Solid):    {M:.2f} -> {get_color(M)}")
    
    # 2. Spectral State Interpretation
    print("\n--- INTERPRETACIÓN NEURONAL (Lo que ve el Bot) ---")
    
    if m < 0 and s > 0.5:
        print(">> OJO: PULLBACK DETECTADO (Micro Rojo en Estructura Verde)")
        print(">> ACCIÓN: Buscar Compras en Soporte Dinámico.")
        
    elif m > 0 and s < -0.5:
        print(">> OJO: RALLY BAJISTA (Micro Verde en Estructura Roja)")
        print(">> ACCIÓN: Buscar Ventas en Resistencia.")
        
    elif m > 0.8 and s > 0.8 and M > 0.8:
        print(">> ESTADO: IMPULSO FRACTAL (Alineación Total)")
        print(">> ACCIÓN: Mantener / Agregar. No operar contra tendencia.")
        
    elif abs(s) < 0.2:
        print(">> ESTADO: RANGO / ACUMULACIÓN (Estructura Gris)")
        print(">> ACCIÓN: Esperar ruptura de volatilidad.")
        
    # 3. Expansion Metrics
    print(f"\n--- FUERZA ESPECTRAL (Expansion) ---")
    print(f"Micro Spectrum (Thrust): {metrics.get('micro_spectrum', 0):.2f} units")
    print(f"Operative Spectrum (Flow): {metrics.get('operative_spectrum', 0):.2f} units")

def main():
    print("=== INICIANDO AUDITORÍA DE VISIÓN ESPECTRAL ===")
    
    # Scenario 1: Healthy Bull Trend
    df_bull = create_spectral_scenario("Tendencia Sana", n_candles=150, regime="BULL_TREND")
    audit_spectral_vision(df_bull, "BULL RUN")
    
    # Scenario 2: Pullback (The "Kiss")
    df_pull = create_spectral_scenario("Pullback Táctico", n_candles=150, regime="PULLBACK_IN_BULL")
    audit_spectral_vision(df_pull, "PULLBACK")
    
    # Scenario 3: Collapse
    df_bear = create_spectral_scenario("Colapso Total", n_candles=150, regime="BEAR_COLLAPSE")
    audit_spectral_vision(df_bear, "CRASH")
    
if __name__ == "__main__":
    main()
