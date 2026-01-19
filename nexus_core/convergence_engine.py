"""
FEAT NEXUS: CONVERGENCE ENGINE (Operation Gravity Binding)
==========================================================
The 'Bridge' between Spectral Physics and Auction Theory.
Fuses Price Energy (Waves) with Liquidity Mass (Volume).
"""

import numpy as np
from typing import Dict, Any

class ConvergenceEngine:
    """
    Calculates synergetic metrics between the Physics and Volume sectors.
    Determines if a trend is 'Solid' (Volume-Backed) or 'Hollow' (Vacuum-Driven).
    """

    def calculate_sgi(self, pvp: float, spectral_layers: Dict[str, float]) -> float:
        """
        Spectral Gravity Index (SGI).
        Measures the distance between the center of gravity (PVP) and 
        the operative spectral ribbon (SC4 Sniper to SC6 Base).
        
        SGI > 0: PVP is above the ribbon (Bullish Mass).
        SGI < 0: PVP is below the ribbon (Bearish Mass).
        Overlap -> Equilibrium.
        """
        sc4 = spectral_layers.get("SC_4_SNIPER", 0.0)
        sc6 = spectral_layers.get("SC_6_BASE", 0.0)
        
        ribbon_center = (sc4 + sc6) / 2.0
        # Normalization by the ribbon width to get a sensitivity score
        ribbon_width = abs(sc4 - sc6) + 1e-9
        
        sgi = (pvp - ribbon_center) / ribbon_width
        return np.clip(sgi, -3.0, 3.0)

    def calculate_vam(self, delta_price: float, volume_density: float, delta_time: float = 1.0) -> float:
        """
        Volume-Adjusted Momentum (VAM).
        Momentum = (dP * Vol_Density) / dT.
        
        High VAM: Explosive move with real accumulation.
        Low VAM: Fast price move but 'hollow' (no volume density support).
        """
        # We use volume_density (kurtosis or relative concentration) as the multiplier
        vam = (delta_price * (1.0 + volume_density)) / (delta_time + 1e-9)
        return vam

    def detect_hollow_rally(self, sgi: float, elastic_gap: float) -> bool:
        """
        Detects a 'Hollow Rally' (Vacuum).
        Price is stretched (High Elastic Gap) but PVP is anclado (Low/Divergent SGI).
        """
        # If price is far above (Elastic Gap > 1) but PVP remains far below (SGI < -1)
        if elastic_gap > 0.8 and sgi < -0.5:
            return True # Bullish Vacuum
        if elastic_gap < -0.8 and sgi > 0.5:
            return True # Bearish Vacuum
        return False

    def calculate_titanium_floor(self, poc: float, ema_operative: float, threshold_pct: float = 0.002) -> str:
        """
        Titanium Floor Detection (EMA-POC Confluence).
        When a Dynamic Support (EMA) perfectly overlaps a Static Support (POC),
        the probability of bounce exceeds 90%.
        
        Args:
            poc: Point of Control (Volume Mass Center).
            ema_operative: Operative EMA Layer (e.g., SC4 Sniper).
            threshold_pct: Confluence threshold (default 0.2% of price).
        
        Returns:
            'TITANIUM_SUPPORT' | 'TITANIUM_RESISTANCE' | 'NEUTRAL'
        """
        distance = abs(poc - ema_operative)
        relative_dist = distance / (poc + 1e-9)
        
        if relative_dist < threshold_pct:
            if poc < ema_operative:
                return "TITANIUM_RESISTANCE"
            else:
                return "TITANIUM_SUPPORT"
        return "NEUTRAL"

    def calculate_hurst_gate(self, hurst_exponent: float, ema_signal_valid: bool) -> bool:
        """
        Hurst-Gated EMA Validity.
        
        Rule: If Hurst < 0.5 (Mean Reversion regime), EMAs are unreliable.
        We gate the EMA signal to prevent false crossovers in ranging markets.
        
        Args:
            hurst_exponent: The H value from Hurst analysis (0 to 1).
            ema_signal_valid: Whether the EMA crossover/signal is currently true.
        
        Returns:
            True if the EMA signal should be trusted, False otherwise.
        """
        if hurst_exponent < 0.5:
            # Mean Reversion: Do NOT trust EMA signals.
            return False
        if hurst_exponent > 0.6:
            # Trending: Trust the EMA signal fully.
            return ema_signal_valid
        # Gray Zone (0.5 - 0.6): Reduce confidence but don't fully gate.
        return ema_signal_valid # Pass through, let other filters handle it.

    def quantum_semaphore(self, time_valid: bool, structure_valid: bool, 
                          space_valid: bool, acceleration_valid: bool) -> dict:
        """
        The 'Quantum Semaphore' (Serial Signal Validation Chain).
        
        Implements the FEAT Signal Custody Chain:
        1. TIME (Chronos): Are we in a valid session/killzone?
        2. FORM + PvP (Structure): Is price on a valid pattern with volume?
        3. SPACE + EMAs (Spectral): Is the route clear (aligned layers)?
        4. ACCELERATION (Physics): Is there kinetic ignition?
        
        Each stage MUST pass before the next is evaluated.
        
        Returns:
            dict with 'execute', 'stage_reached', and 'reason'.
        """
        if not time_valid:
            return {"execute": False, "stage_reached": 1, "reason": "TIME: Not in Killzone. System idle."}
        
        if not structure_valid:
            return {"execute": False, "stage_reached": 2, "reason": "FORM: No valid structure or volume confluence."}
        
        if not space_valid:
            return {"execute": False, "stage_reached": 3, "reason": "SPACE: Spectral layers not aligned or route blocked."}
        
        if not acceleration_valid:
            return {"execute": False, "stage_reached": 4, "reason": "ACCELERATION: No kinetic ignition detected."}
        
        return {"execute": True, "stage_reached": 5, "reason": "ALL CLEAR: Quantum Confirmation achieved."}

# Singleton Export
convergence_engine = ConvergenceEngine()
