"""
FEAT KILLZONE MICRO-TEMPORAL INTELLIGENCE
==========================================
Transforma el conocimiento de dominio (killzone 09:00-13:00) en tensores ML.

16 bloques de 15 minutos con:
- session_heat_score (0.0-1.0)
- expansion_probability (prior bayesiano)
- liquidity_state (FSM)
- action_recommendation

TIMEZONE: Bolivia (UTC-4)
PEAK BLOCK: 09:30-09:44 ⭐ (máxima convergencia institucional)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple
from enum import Enum
import pytz

logger = logging.getLogger("FEAT.KillzoneIntelligence")


# =============================================================================
# TIMEZONE
# =============================================================================

try:
    BOLIVIA_TZ = pytz.timezone('America/La_Paz')
except:
    BOLIVIA_TZ = None


# =============================================================================
# ENUMS: LIQUIDITY STATES (FSM)
# =============================================================================

class LiquidityState(Enum):
    ACCUMULATION = "ACCUMULATION"      # 09:00-09:14 - Building liquidity
    SWEEP_HUNT = "SWEEP_HUNT"          # 09:15-09:29 - HFT hunting stops
    EXPANSION_PEAK = "EXPANSION_PEAK"  # 09:30-09:44 - ⭐ Max execution
    CONTINUATION = "CONTINUATION"       # 09:45-10:29 - Follow-through
    CONSOLIDATION = "CONSOLIDATION"     # 10:30-10:59 - Testing structure
    FIX_EVENT = "FIX_EVENT"            # 11:00-11:14 - LBMA PM fix
    POST_FIX = "POST_FIX"              # 11:15-11:44 - Post-fix reaction
    DECELERATION = "DECELERATION"      # 11:45-12:29 - Winding down
    CLOSE_PHASE = "CLOSE_PHASE"        # 12:30-12:59 - Closing killzone


class ActionRecommendation(Enum):
    WAIT_CONFIRM = "WAIT_CONFIRM"      # Wait for confirmation
    PREPARE = "PREPARE"                 # Prepare entry, mark levels
    EXECUTE = "EXECUTE"                 # Entry with full size
    EXECUTE_HALF = "EXECUTE_HALF"      # Entry with 50% size
    TRAILING = "TRAILING"               # Manage existing position
    SCALE_OUT = "SCALE_OUT"            # Reduce exposure
    NO_NEW = "NO_NEW"                  # No new positions
    AVOID = "AVOID"                    # Avoid trading (fix event)


# =============================================================================
# KILLZONE BLOCKS (16 x 15-min blocks)
# =============================================================================

KILLZONE_BLOCKS = {
    # Block 1: 09:00-09:14 - Opening accumulation
    "09:00": {
        "end": "09:14",
        "session_heat": 0.70,
        "expansion_prob": 0.65,
        "state": LiquidityState.ACCUMULATION,
        "action": ActionRecommendation.WAIT_CONFIRM,
        "volume_threshold": 1.3,
        "description": "Apertura NY. Liquidez sube. Esperar confirmación H1.",
        "notes": "No entrar en primera vela. Exigir cierre H1 + tick-vol ≥1.3x."
    },
    
    # Block 2: 09:15-09:29 - Sweep hunting
    "09:15": {
        "end": "09:29",
        "session_heat": 0.80,
        "expansion_prob": 0.75,
        "state": LiquidityState.SWEEP_HUNT,
        "action": ActionRecommendation.PREPARE,
        "volume_threshold": 1.3,
        "description": "HFT busca stops. Posibles sweeps en niveles clave.",
        "notes": "Preparar entrada en rechazo de sweep. Stop bajo mínimo del sweep."
    },
    
    # Block 3: 09:30-09:44 ⭐ PEAK - Maximum probability
    "09:30": {
        "end": "09:44",
        "session_heat": 1.00,  # ⭐ PEAK
        "expansion_prob": 0.85,
        "state": LiquidityState.EXPANSION_PEAK,
        "action": ActionRecommendation.EXECUTE,
        "volume_threshold": 1.5,
        "description": "⭐ MÁXIMA PROBABILIDAD. Convergencia institucional.",
        "notes": "Break con cierre H1 + vol ≥1.5x → entrada en retest. RR mínimo 1:1.5.",
        "is_peak": True
    },
    
    # Block 4: 09:45-09:59 - Continuation
    "09:45": {
        "end": "09:59",
        "session_heat": 0.85,
        "expansion_prob": 0.75,
        "state": LiquidityState.CONTINUATION,
        "action": ActionRecommendation.TRAILING,
        "volume_threshold": 1.3,
        "description": "Continuación o agotamiento. Trailing si dentro.",
        "notes": "Si ya entraste, trailing. Si no, solo retest limpio con volumen."
    },
    
    # Block 5: 10:00-10:14 - Follow-through
    "10:00": {
        "end": "10:14",
        "session_heat": 0.80,
        "expansion_prob": 0.70,
        "state": LiquidityState.CONTINUATION,
        "action": ActionRecommendation.EXECUTE_HALF,
        "volume_threshold": 1.3,
        "description": "Segunda ola de ejecución. Pullbacks válidos.",
        "notes": "Entrada en pullback con confirmación 5m/15m si H4/D1 alineados."
    },
    
    # Block 6: 10:15-10:29 - Bank hedging
    "10:15": {
        "end": "10:29",
        "session_heat": 0.75,
        "expansion_prob": 0.65,
        "state": LiquidityState.CONTINUATION,
        "action": ActionRecommendation.EXECUTE_HALF,
        "volume_threshold": 1.2,
        "description": "Cobertura de bancos. Micro-breaks y retests.",
        "notes": "Preparar segunda entrada (scaling in). Stops ajustados."
    },
    
    # Block 7: 10:30-10:44 - Consolidation
    "10:30": {
        "end": "10:44",
        "session_heat": 0.70,
        "expansion_prob": 0.60,
        "state": LiquidityState.CONSOLIDATION,
        "action": ActionRecommendation.EXECUTE_HALF,
        "volume_threshold": 1.2,
        "description": "Consolidación. Confirmar tendencia real.",
        "notes": "Entrar si 2 H1 seguidas confirman. Si divergencia en vol, salir."
    },
    
    # Block 8: 10:45-10:59 - Structure testing
    "10:45": {
        "end": "10:59",
        "session_heat": 0.65,
        "expansion_prob": 0.55,
        "state": LiquidityState.CONSOLIDATION,
        "action": ActionRecommendation.SCALE_OUT,
        "volume_threshold": 1.1,
        "description": "Testing estructura. Reducir exposición.",
        "notes": "No abrir nuevas grandes posiciones. Reducir si ya dentro."
    },
    
    # Block 9: 11:00-11:14 - LBMA PM FIX ⚠️
    "11:00": {
        "end": "11:14",
        "session_heat": 0.50,
        "expansion_prob": 0.45,
        "state": LiquidityState.FIX_EVENT,
        "action": ActionRecommendation.AVOID,
        "volume_threshold": 1.0,
        "description": "⚠️ LBMA PM Fix. Posibles falsos breakouts.",
        "notes": "EVITAR ±10-20 min. Usar nivel del fix como S/R intradía.",
        "fix_event": True,
        "fix_name": "LBMA_PM"
    },
    
    # Block 10: 11:15-11:29 - Post-fix reaction
    "11:15": {
        "end": "11:29",
        "session_heat": 0.60,
        "expansion_prob": 0.55,
        "state": LiquidityState.POST_FIX,
        "action": ActionRecommendation.EXECUTE_HALF,
        "volume_threshold": 1.2,
        "description": "Reacción post-fix. Decisión de continuar o revertir.",
        "notes": "Si continúa con vol → retest. Si revierte → sweep+rej para reversal."
    },
    
    # Block 11: 11:30-11:44 - Second wave or decay
    "11:30": {
        "end": "11:44",
        "session_heat": 0.55,
        "expansion_prob": 0.50,
        "state": LiquidityState.POST_FIX,
        "action": ActionRecommendation.TRAILING,
        "volume_threshold": 1.1,
        "description": "Segunda fase overlap o desgaste.",
        "notes": "Scale out parcial si +1xATR de beneficio."
    },
    
    # Block 12: 11:45-11:59 - Pre-close transition
    "11:45": {
        "end": "11:59",
        "session_heat": 0.45,
        "expansion_prob": 0.40,
        "state": LiquidityState.DECELERATION,
        "action": ActionRecommendation.SCALE_OUT,
        "volume_threshold": 1.0,
        "description": "Transición a última hora. Liquidez baja.",
        "notes": "Evitar añadir. Preparar cierre parcial."
    },
    
    # Block 13: 12:00-12:14 - Late execution attempts
    "12:00": {
        "end": "12:14",
        "session_heat": 0.40,
        "expansion_prob": 0.35,
        "state": LiquidityState.DECELERATION,
        "action": ActionRecommendation.NO_NEW,
        "volume_threshold": 1.2,
        "description": "Intentos tardíos de ejecución.",
        "notes": "Solo entradas con vol confirmado y retest."
    },
    
    # Block 14: 12:15-12:29 - Continuation or finish
    "12:15": {
        "end": "12:29",
        "session_heat": 0.35,
        "expansion_prob": 0.30,
        "state": LiquidityState.DECELERATION,
        "action": ActionRecommendation.TRAILING,
        "volume_threshold": 1.1,
        "description": "Continuación o finalización del día.",
        "notes": "Trailing y protección. Solo scalps con RR ceñido."
    },
    
    # Block 15: 12:30-12:44 - Low probability
    "12:30": {
        "end": "12:44",
        "session_heat": 0.25,
        "expansion_prob": 0.20,
        "state": LiquidityState.CLOSE_PHASE,
        "action": ActionRecommendation.NO_NEW,
        "volume_threshold": 1.0,
        "description": "Baja probabilidad de nuevas expansiones.",
        "notes": "Pequeñas entradas con stops tight. No escalar."
    },
    
    # Block 16: 12:45-12:59 - Killzone close
    "12:45": {
        "end": "12:59",
        "session_heat": 0.20,
        "expansion_prob": 0.15,
        "state": LiquidityState.CLOSE_PHASE,
        "action": ActionRecommendation.SCALE_OUT,
        "volume_threshold": 1.0,
        "description": "Cierre de killzone. Toma de ganancias.",
        "notes": "Cerrar mayoría de posiciones intradía."
    }
}


# =============================================================================
# CONFIRMATION RULES
# =============================================================================

class ConfirmationLevel(Enum):
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    CONFIRMED = 4


# H1 confirmation windows
H1_CONFIRMATION = {
    "retest_window_min": 5,    # 5-30 min for retest
    "retest_window_max": 30,
    "volume_threshold": 1.3,
    "killzone_boost": 0.15,    # Boost if in killzone
}

# H4 confirmation rules
H4_CONFIRMATION = {
    "follow_h4_count": 1,       # Need 1 H4 confirmation
    "follow_h1_count": 4,       # Or 4 H1 confirmations
    "killzone_weight_boost": 0.15,  # Boost from 0.25 to 0.40
}

# D1 confirmation rules
D1_CONFIRMATION = {
    "next_killzone_confirm": True,  # Wait for confirmation in next 09:00-13:00
    "fix_confirmation": True,        # LBMA fix alignment adds confidence
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_bolivia_time(utc_time: datetime = None) -> datetime:
    """Convert UTC to Bolivia time (UTC-4)."""
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    
    if BOLIVIA_TZ:
        try:
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            return utc_time.astimezone(BOLIVIA_TZ)
        except:
            pass
    return utc_time - timedelta(hours=4)


def get_current_block(bolivia_time: datetime = None) -> Dict[str, Any]:
    """
    Get the current 15-min killzone block.
    Returns block info or None if outside killzone (09:00-13:00).
    """
    if bolivia_time is None:
        bolivia_time = get_bolivia_time()
    
    hour = bolivia_time.hour
    minute = bolivia_time.minute
    
    # Check if in killzone (09:00-13:00)
    if hour < 9 or hour >= 13:
        return {
            "in_killzone": False,
            "session_heat": 0.0,
            "state": "OUTSIDE_KILLZONE",
            "action": "WAIT_FOR_KILLZONE"
        }
    
    # Find matching block
    for block_start, block_info in KILLZONE_BLOCKS.items():
        start_parts = block_start.split(":")
        start_hour = int(start_parts[0])
        start_min = int(start_parts[1])
        
        end_parts = block_info["end"].split(":")
        end_hour = int(end_parts[0])
        end_min = int(end_parts[1])
        
        # Check if current time is in this block
        current_minutes = hour * 60 + minute
        start_minutes = start_hour * 60 + start_min
        end_minutes = end_hour * 60 + end_min
        
        if start_minutes <= current_minutes <= end_minutes:
            return {
                "in_killzone": True,
                "block_start": block_start,
                "block_end": block_info["end"],
                "session_heat": block_info["session_heat"],
                "expansion_prob": block_info["expansion_prob"],
                "state": block_info["state"].value,
                "action": block_info["action"].value,
                "volume_threshold": block_info["volume_threshold"],
                "description": block_info["description"],
                "notes": block_info["notes"],
                "is_peak": block_info.get("is_peak", False),
                "is_fix_event": block_info.get("fix_event", False)
            }
    
    return {"in_killzone": False, "session_heat": 0.0}


def get_all_blocks_ml_features() -> Dict[str, Dict]:
    """
    Generate ML features for all blocks.
    Used for training the neural network.
    """
    features = {}
    
    for block_start, block_info in KILLZONE_BLOCKS.items():
        features[block_start] = {
            "session_heat": block_info["session_heat"],
            "expansion_prob": block_info["expansion_prob"],
            "state_encoded": list(LiquidityState).index(block_info["state"]),
            "action_encoded": list(ActionRecommendation).index(block_info["action"]),
            "volume_threshold": block_info["volume_threshold"],
            "is_peak": 1 if block_info.get("is_peak") else 0,
            "is_fix": 1 if block_info.get("fix_event") else 0
        }
    
    return features


def calculate_alignment_factor(
    d1_direction: str,
    h4_direction: str,
    h1_direction: str
) -> Dict[str, Any]:
    """
    Calculate alignment factor (-1.0 to +1.0) for ML input.
    
    +1.0 = All bullish aligned
    -1.0 = All bearish aligned
    0.0 = Conflict or neutral
    """
    def direction_to_value(d: str) -> float:
        d = d.upper() if d else "NEUTRAL"
        if d == "BULLISH":
            return 1.0
        elif d == "BEARISH":
            return -1.0
        return 0.0
    
    d1_val = direction_to_value(d1_direction)
    h4_val = direction_to_value(h4_direction)
    h1_val = direction_to_value(h1_direction)
    
    # Weighted average (D1 and H4 have more weight)
    weighted_factor = (d1_val * 0.35 + h4_val * 0.40 + h1_val * 0.25)
    
    # Check for full alignment
    all_same = (d1_val == h4_val == h1_val) and d1_val != 0
    
    # Conflict detection
    has_conflict = (d1_val * h4_val < 0) or (d1_val * h1_val < 0) or (h4_val * h1_val < 0)
    
    # Size recommendation
    if all_same:
        size_mult = 1.0
        alignment_type = "FULL"
    elif has_conflict:
        size_mult = 0.25
        alignment_type = "CONFLICT"
    elif d1_val == 0 and h4_val != 0:
        size_mult = 0.5
        alignment_type = "PARTIAL_H4"
    else:
        size_mult = 0.5
        alignment_type = "PARTIAL"
    
    return {
        "alignment_factor": round(weighted_factor, 3),
        "alignment_type": alignment_type,
        "size_multiplier": size_mult,
        "all_aligned": all_same,
        "has_conflict": has_conflict,
        "d1": d1_direction,
        "h4": h4_direction,
        "h1": h1_direction
    }


def check_h1_confirmation(
    h1_closed_outside: bool,
    minutes_since_close: int,
    retest_occurred: bool,
    retest_rejected: bool,
    current_volume_ratio: float,
    in_killzone: bool = False
) -> Dict[str, Any]:
    """
    Check H1 confirmation level.
    
    Returns confirmation level and whether to proceed.
    """
    score = 0
    reasons = []
    
    # H1 closed outside level
    if h1_closed_outside:
        score += 1
        reasons.append("H1 closed outside level")
    
    # Within retest window (5-30 min)
    if H1_CONFIRMATION["retest_window_min"] <= minutes_since_close <= H1_CONFIRMATION["retest_window_max"]:
        if retest_occurred and retest_rejected:
            score += 2
            reasons.append("Retest with rejection")
        elif retest_occurred:
            score += 1
            reasons.append("Retest occurred")
    
    # Volume confirmation
    if current_volume_ratio >= H1_CONFIRMATION["volume_threshold"]:
        score += 1
        reasons.append(f"Volume {current_volume_ratio:.1f}x")
    
    # Killzone boost
    if in_killzone:
        score += 0.5
        reasons.append("In killzone")
    
    # Determine level
    if score >= 3.5:
        level = ConfirmationLevel.CONFIRMED
    elif score >= 2.5:
        level = ConfirmationLevel.STRONG
    elif score >= 1.5:
        level = ConfirmationLevel.MODERATE
    elif score >= 0.5:
        level = ConfirmationLevel.WEAK
    else:
        level = ConfirmationLevel.NONE
    
    return {
        "level": level.name,
        "score": round(score, 1),
        "proceed": score >= 2.5,
        "reasons": reasons
    }


def generate_temporal_ml_features(
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    current_volume_ratio: float = 1.0,
    h1_confirmation_score: float = 0.0
) -> Dict[str, Any]:
    """
    Generate complete ML feature vector for current moment.
    This is what the neural network receives as input.
    """
    # Parse time
    if server_time_utc:
        try:
            if isinstance(server_time_utc, str):
                if "T" in server_time_utc:
                    utc_time = datetime.fromisoformat(server_time_utc.replace("Z", "+00:00"))
                else:
                    parts = server_time_utc.split(":")
                    utc_time = datetime.now(timezone.utc).replace(hour=int(parts[0]), minute=int(parts[1]))
            else:
                utc_time = server_time_utc
        except:
            utc_time = datetime.now(timezone.utc)
    else:
        utc_time = datetime.now(timezone.utc)
    
    bolivia_time = get_bolivia_time(utc_time)
    
    # Get current block
    block = get_current_block(bolivia_time)
    
    # Calculate alignment
    alignment = calculate_alignment_factor(d1_direction, h4_direction, h1_direction)
    
    # Build ML feature vector
    ml_features = {
        # Time features
        "bolivia_hour": bolivia_time.hour,
        "bolivia_minute": bolivia_time.minute,
        "hour_normalized": bolivia_time.hour / 23.0,
        "minute_normalized": bolivia_time.minute / 59.0,
        "weekday": bolivia_time.weekday(),
        
        # Killzone features
        "in_killzone": 1 if block.get("in_killzone") else 0,
        "session_heat": block.get("session_heat", 0.0),
        "expansion_prob": block.get("expansion_prob", 0.0),
        "is_peak_block": 1 if block.get("is_peak") else 0,
        "is_fix_event": 1 if block.get("is_fix_event") else 0,
        
        # Volume
        "volume_ratio": current_volume_ratio,
        "volume_above_threshold": 1 if current_volume_ratio >= block.get("volume_threshold", 1.0) else 0,
        
        # Alignment
        "alignment_factor": alignment["alignment_factor"],
        "all_aligned": 1 if alignment["all_aligned"] else 0,
        "has_conflict": 1 if alignment["has_conflict"] else 0,
        "size_multiplier": alignment["size_multiplier"],
        
        # Confirmation
        "h1_confirmation_score": h1_confirmation_score,
        
        # Combined probability (bayesian prior × evidence)
        "combined_probability": round(
            block.get("session_heat", 0.5) * 
            alignment["size_multiplier"] * 
            min(1.0, current_volume_ratio / block.get("volume_threshold", 1.0)) *
            (1.0 + h1_confirmation_score * 0.2),
            3
        )
    }
    
    return {
        "module": "FEAT_KILLZONE_INTELLIGENCE",
        "timestamp": utc_time.isoformat(),
        "bolivia_time": bolivia_time.strftime("%H:%M"),
        "current_block": block,
        "alignment": alignment,
        "ml_features": ml_features,
        "guidance": {
            "can_trade": block.get("in_killzone", False) and not block.get("is_fix_event", False),
            "action": block.get("action", "WAIT"),
            "size": alignment["size_multiplier"],
            "notes": block.get("notes", "")
        }
    }


# =============================================================================
# ASYNC MCP WRAPPERS
# =============================================================================

async def feat_get_current_killzone_block(server_time_utc: str = None) -> Dict[str, Any]:
    """MCP Tool: Get current 15-min killzone block."""
    if server_time_utc:
        try:
            utc_time = datetime.fromisoformat(server_time_utc.replace("Z", "+00:00"))
            bolivia_time = get_bolivia_time(utc_time)
        except:
            bolivia_time = get_bolivia_time()
    else:
        bolivia_time = get_bolivia_time()
    
    return get_current_block(bolivia_time)


async def feat_generate_temporal_features(
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    current_volume_ratio: float = 1.0
) -> Dict[str, Any]:
    """MCP Tool: Generate temporal ML features."""
    return generate_temporal_ml_features(
        server_time_utc, d1_direction, h4_direction, h1_direction, current_volume_ratio
    )


async def feat_check_h1_confirmation(
    h1_closed_outside: bool,
    minutes_since_close: int,
    retest_occurred: bool = False,
    retest_rejected: bool = False,
    current_volume_ratio: float = 1.0,
    in_killzone: bool = False
) -> Dict[str, Any]:
    """MCP Tool: Check H1 confirmation level."""
    return check_h1_confirmation(
        h1_closed_outside, minutes_since_close, retest_occurred, 
        retest_rejected, current_volume_ratio, in_killzone
    )
