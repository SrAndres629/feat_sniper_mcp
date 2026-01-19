
"""
[LEVEL 51] SPECTRAL CONFIGURATION
Centralized constants for the 10-Layer Spectral Architecture.
"""

SPECTRAL_CONFIG = {
    "LAYER_1_MICRO": {
        "ignition": [1, 2, 3],
        "combustion": [6, 7, 8, 9],
        "thrust": [12, 13, 14],
        "colors": ["#FF0000", "#FF4500", "#FF8C00"] # Red -> Orange
    },
    "LAYER_2_OPERATIVE": {
        "transition": [16, 24, 32],
        "flow": [48, 64, 96, 128],
        "bank": [160, 192, 224],
        "colors": ["#FFFF00", "#ADFF2F", "#008000"] # Yellow -> Green
    },
    "LAYER_3_MACRO": {
        "deep": [256, 320, 384],
        "abyss": [448, 512, 640, 768],
        "core": [896, 1024, 1280],
        "colors": ["#0000FF", "#4B0082", "#8B008B"] # Blue -> Violet
    },
    "LAYER_4_BIAS": {
        "horizon": [2048],
        "colors": ["#808080"] # Grey
    }
}
