import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from .models import ZoneType, VolatilityState, ProjectedZone, ActionPlan
from .calculations import FIB_LEVELS, EXTENSION_LEVELS
from app.core.config import settings

logger = logging.getLogger("feat.zones.engine")

class ZoneProjector:
    """Engine for projecting price zones based on structure and volatility."""
    def __init__(self):
        logger.info("[ZoneProjector] Engine initialized")

    def get_volatility_state(self, df: pd.DataFrame, window: int = 20) -> Tuple[VolatilityState, float]:
        if len(df) < window + 5: return VolatilityState.NORMAL, 1.0
        tr = np.maximum(df["high"]-df["low"], np.maximum(abs(df["high"]-df["close"].shift(1)), abs(df["low"]-df["close"].shift(1))))
        atr = tr.rolling(window).mean()
        ca, aa = atr.iloc[-1], atr.iloc[-window:].mean()
        if aa == 0: return VolatilityState.NORMAL, 1.0
        f = ca / aa
        s = VolatilityState.EXTREME if f > settings.ZONE_VOL_EXTREME_TH else \
            VolatilityState.HIGH if f > settings.ZONE_VOL_HIGH_TH else \
            VolatilityState.LOW if f < settings.ZONE_VOL_LOW_TH else \
            VolatilityState.NORMAL
        return s, f

    def get_current_killzone(self, utc_offset: int = -4) -> Tuple[bool, str]:
        try:
            from app.skills.liquidity_detector import get_current_kill_zone
            kz = get_current_kill_zone(utc_offset)
            return kz is not None, kz or "NONE"
        except: return False, "UNKNOWN"

    def identify_last_impulse(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        if len(df) < lookback: return {"found": False}
        recent = df.tail(lookback)
        atr = (recent["high"] - recent["low"]).rolling(14).mean().iloc[-1]
        moves = abs(recent["close"] - recent["open"])
        mi = moves.idxmax()
        mc = recent.loc[mi]
        ms = abs(mc["close"] - mc["open"])
        if ms < (atr * 1.2 if not pd.isna(atr) else 0.001): return {"found": False}
        return {"found": True, "direction": "BULLISH" if mc["close"] > mc["open"] else "BEARISH", "high": float(mc["high"]), "low": float(mc["low"]), "size": float(ms), "atr_multiple": float(ms/atr) if atr > 0 else 0}

    def calculate_retracement_zones(self, ih, il, d, vs, cp) -> List[ProjectedZone]:
        zones, ir = [], ih - il
        levels = FIB_LEVELS["aggressive"] if vs == VolatilityState.HIGH else FIB_LEVELS["deep"] if vs == VolatilityState.LOW else FIB_LEVELS["normal"]
        pb = settings.ZONE_PROB_BASE_HIGH_VOL if vs == VolatilityState.HIGH else \
             settings.ZONE_PROB_BASE_LOW_VOL if vs == VolatilityState.LOW else \
             settings.ZONE_PROB_BASE_NORMAL
        for i, fib in enumerate(levels):
            zm = ih - (ir * fib) if d == "BULLISH" else il + (ir * fib)
            zh, zl = zm + ir*0.02, zm - ir*0.02
            act = ("BUY" if zl < cp else "WAIT") if d == "BULLISH" else ("SELL" if zh > cp else "WAIT")
            zones.append(ProjectedZone(ZoneType.RETRACEMENT, zh, zl, max(0.3, pb - i*0.1), abs(cp - zm)*settings.ZONE_DIST_MULTIPLIER, 1.0, vs in [VolatilityState.HIGH, VolatilityState.EXTREME], f"Fib {fib*100:.1f}%", act))
        return zones

    def calculate_expansion_targets(self, ih, il, d, vs, cp) -> List[ProjectedZone]:
        zones, ir = [], ih - il
        levels = EXTENSION_LEVELS["aggressive"] if vs in [VolatilityState.HIGH, VolatilityState.EXTREME] else EXTENSION_LEVELS["conservative"] if vs == VolatilityState.LOW else EXTENSION_LEVELS["standard"]
        pb = 0.7 if vs in [VolatilityState.HIGH, VolatilityState.EXTREME] else 0.5 if vs == VolatilityState.LOW else 0.6
        for i, ext in enumerate(levels):
            t = il + (ir * ext) if d == "BULLISH" else ih - (ir * ext)
            zh, zl = t + ir*0.01, t - ir*0.01
            act = ("TAKE_PROFIT" if cp < t else "WAIT") if d == "BULLISH" else ("TAKE_PROFIT" if cp > t else "WAIT")
            zones.append(ProjectedZone(ZoneType.TARGET, zh, zl, max(0.2, pb - i*0.15), abs(cp - t)*10, 1.0, vs in [VolatilityState.HIGH, VolatilityState.EXTREME], f"Fib {ext*100:.1f}% Target", act))
        return zones

    def get_structure_zones(self, df: pd.DataFrame, cp: float) -> List[ProjectedZone]:
        zones = []
        try:
            from app.skills.liquidity_detector import detect_order_blocks, detect_fvg, detect_liquidity_pools
            obs = detect_order_blocks(df, 50)
            for o in obs: zones.append(ProjectedZone(ZoneType.BOUNCE, o.top, o.bottom, 0.7*o.strength, abs(cp-o.midpoint)*settings.ZONE_DIST_MULTIPLIER, 1.0, False, f"{o.zone_type} OB", "BUY" if "BULLISH" in o.zone_type else "SELL"))
            fvgs = detect_fvg(df, 30)
            for f in fvgs: zones.append(ProjectedZone(ZoneType.BOUNCE, f["top"], f["bottom"], settings.ZONE_FVG_PROB, abs(cp-f["midpoint"])*settings.ZONE_DIST_MULTIPLIER, 1.0, False, f"{f['type']} FVG", "BUY" if f["type"] == "BULLISH" else "SELL"))
            liq = detect_liquidity_pools(df, 50)
            if liq.get("liquidity_above", 0) > 0: zones.append(ProjectedZone(ZoneType.LIQUIDITY, liq["liquidity_above"]*1.001, liq["liquidity_above"]*0.999, 0.6, abs(cp-liq["liquidity_above"])*10, 1.0, True, "Sell-side Liq", "BREAKOUT_LONG"))
            if liq.get("liquidity_below", 0) > 0: zones.append(ProjectedZone(ZoneType.LIQUIDITY, liq["liquidity_below"]*1.001, liq["liquidity_below"]*0.999, 0.6, abs(cp-liq["liquidity_below"])*10, 1.0, True, "Buy-side Liq", "BREAKOUT_SHORT"))
        except Exception as e: logger.warning(f"Structure zones error: {e}")
        return zones

    def generate_action_plan(self, df: pd.DataFrame, cp: float) -> ActionPlan:
        vs, vf = self.get_volatility_state(df)
        in_kz, kz_n = self.get_current_killzone()
        imp = self.identify_last_impulse(df)
        az, structure, reasoning = [], "RANGING", []
        
        if imp["found"]:
            structure = imp["direction"]
            reasoning.append(f"Last impulse: {structure} ({imp['atr_multiple']:.1f}x ATR)")
            az.extend(self.calculate_retracement_zones(imp["high"], imp["low"], structure, vs, cp))
            az.extend(self.calculate_expansion_targets(imp["high"], imp["low"], structure, vs, cp))
        else: reasoning.append("No clear impulse detected")
        
        az.extend(self.get_structure_zones(df, cp))
        for z in az: z.volatility_factor = vf
        az.sort(key=lambda z: z.probability, reverse=True)
        
        it = next((z for z in az if z.zone_type == ZoneType.TARGET), None)
        bz = next((z for z in az if z.zone_type in [ZoneType.BOUNCE, ZoneType.RETRACEMENT]), None)
        bl = next((z for z in az if z.zone_type == ZoneType.LIQUIDITY), None)
        
        bias = "LONG" if structure == "BULLISH" else "SHORT" if structure == "BEARISH" else "NEUTRAL"
        conf = 0.5 + (settings.ZONE_CONF_KZ_BOOST if in_kz else 0) + \
               (settings.ZONE_CONF_VOL_BOOST if vs == VolatilityState.HIGH else 0) + \
               (0.1 if len([z for z in az if z.probability>0.6])>=3 else 0)
        
        return ActionPlan(cp, structure, vs, in_kz, kz_n, it, bz, bl, az[:10], bias, "WAIT", min(1.0, conf), reasoning)
