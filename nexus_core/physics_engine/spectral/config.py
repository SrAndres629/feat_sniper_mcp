
"""
[LEVEL 52] DECA-CORE SPECTRAL CONFIGURATION
Centralized mapping for frequency bands and sub-layers.
"""

DECA_LAYERS = {
    # GRUPO 1: MICRO-INTENCIÓN (High Frequency)
    "GROUP_1_MICRO": {
        "SC_1_NOISE": [1, 2, 3],
        "SC_2_FAST":  [6, 7, 8, 9],
        "SC_3_VALID": [12, 13, 14]
    },

    # GRUPO 2: OPERATIVA / AGUA (Medium Frequency)
    "GROUP_2_OPERATIVE": {
        "SC_4_SNIPER": [16, 24, 32],
        "SC_5_FAIR":   [48, 64, 96],
        "SC_6_BASE":   [128, 160, 192, 224]
    },

    # GRUPO 3: MACRO / MURO (Low Frequency)
    "GROUP_3_MACRO": {
        "SC_7_SESSION": [256, 320, 384],
        "SC_8_DAY":     [448, 512, 640],
        "SC_9_WEEK":    [768, 896, 1024, 1280]
    },

    # GRUPO 4: SESGO (Ultra-Low Frequency)
    "GROUP_4_BIAS": {
        "SC_10_AXIS": [2048]
    }
}

ORDERED_SUBLAYERS = [
    "SC_1_NOISE", "SC_2_FAST", "SC_3_VALID",
    "SC_4_SNIPER", "SC_5_FAIR", "SC_6_BASE",
    "SC_7_SESSION", "SC_8_DAY", "SC_9_WEEK",
    "SC_10_AXIS"
]
