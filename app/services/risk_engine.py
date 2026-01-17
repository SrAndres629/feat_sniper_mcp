import logging
import MetaTrader5 as mt5
import json
import os
from datetime import datetime, time as dtime, timezone
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.core.mt5_conn import mt5_conn
from app.services.volatility_guard import volatility_guard
from app.services.spread_filter import spread_filter

logger = logging.getLogger("MT5_Bridge.Services.Risk")


# =============================================================================
# THE VAULT: Capital Protection System
# =============================================================================

class TheVault:
    """
    Sistema de protección de capital estilo "banco".
    
    Al duplicar el capital inicial:
    - 50% de las ganancias se "bloquean" (vault_balance)
    - 50% se reinvierten para interés compuesto
    - Se genera alerta para transferir a cuenta de resguardo
    """
    
    VAULT_STATE_FILE = "data/vault_state.json"
    COMPOUNDING_MULTIPLIER = 2.0  # Trigger at 2x initial capital
    VAULT_PERCENTAGE = 0.50  # 50% to vault
    
    def __init__(self, initial_capital: float = 30.0):
        self.initial_capital = initial_capital
        self.vault_balance = 0.0  # Protected profits (virtual)
        self.trading_capital = initial_capital  # Active trading equity
        self.total_vault_transfers = 0
        self.last_trigger_equity = initial_capital
        self.pending_transfer_alerts: List[Dict] = []
        self._load_state()
    
    def _load_state(self):
        """Carga estado persistido del Vault."""
        try:
            if os.path.exists(self.VAULT_STATE_FILE):
                with open(self.VAULT_STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.initial_capital = data.get("initial_capital", self.initial_capital)
                    self.vault_balance = data.get("vault_balance", 0.0)
                    self.trading_capital = data.get("trading_capital", self.initial_capital)
                    self.total_vault_transfers = data.get("total_vault_transfers", 0)
                    self.last_trigger_equity = data.get("last_trigger_equity", self.initial_capital)
                    logger.info(f"[VAULT] State loaded: Vault=${self.vault_balance:.2f}, Trading=${self.trading_capital:.2f}")
        except Exception as e:
            logger.warning(f"[VAULT] Could not load state: {e}")
    
    def _save_state(self):
        """Persiste estado del Vault de forma atomica."""
        import tempfile
        os.makedirs(os.path.dirname(self.VAULT_STATE_FILE) or ".", exist_ok=True)
        data = {
            "initial_capital": self.initial_capital,
            "vault_balance": self.vault_balance,
            "trading_capital": self.trading_capital,
            "total_vault_transfers": self.total_vault_transfers,
            "last_trigger_equity": self.last_trigger_equity,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Atomic Write
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(self.VAULT_STATE_FILE), suffix=".tmp")
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.VAULT_STATE_FILE)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"[VAULT] Atomic write failed: {e}")
    
    def check_vault_trigger(self, current_equity: float) -> Optional[Dict[str, Any]]:
        """
        Verifica si se debe activar el Vault (al duplicar capital).
        
        Returns:
            Dict con detalles de transferencia si se activa, None si no.
        """
        trigger_target = self.last_trigger_equity * self.COMPOUNDING_MULTIPLIER
        
        if current_equity >= trigger_target:
            profit = current_equity - self.last_trigger_equity
            vault_amount = profit * self.VAULT_PERCENTAGE
            reinvest_amount = profit - vault_amount
            
            # Update vault
            self.vault_balance += vault_amount
            self.trading_capital = self.last_trigger_equity + reinvest_amount
            self.last_trigger_equity = self.trading_capital  # New baseline
            self.total_vault_transfers += 1
            
            transfer_alert = {
                "type": "VAULT_TRIGGER",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "profit_total": round(profit, 2),
                "vault_amount": round(vault_amount, 2),
                "reinvest_amount": round(reinvest_amount, 2),
                "new_vault_balance": round(self.vault_balance, 2),
                "new_trading_capital": round(self.trading_capital, 2),
                "transfer_count": self.total_vault_transfers,
                "message": f"💰 VAULT TRIGGER #{self.total_vault_transfers}: Mover ${vault_amount:.2f} a cuenta de resguardo"
            }
            
            self.pending_transfer_alerts.append(transfer_alert)
            self._save_state()
            
            logger.warning(f"[VAULT] 🔐 TRIGGERED! Profit: ${profit:.2f} | To Vault: ${vault_amount:.2f} | Reinvest: ${reinvest_amount:.2f}")
            
            return transfer_alert
        
        return None
    
    def record_realized_profit(self, profit: float):
        """
        [SENIOR COMPOUNDING] 
        Splits realized profit 50/50 between Vault (Safety) and Reinvestment (Growth).
        """
        if profit <= 0:
            return

        vault_share = profit * self.VAULT_PERCENTAGE
        growth_share = profit - vault_share

        self.vault_balance += vault_share
        self.trading_capital += growth_share
        
        logger.info(f"[VAULT] Realized Profit Split: +${vault_share:.2f} Vault, +${growth_share:.2f} Compounding")
        self._save_state()
    
    def get_effective_margin(self, account_free_margin: float) -> float:
        """
        Retorna el margen efectivo para trading (excluyendo vault virtual).
        El bot NO debe usar el vault_balance como margen.
        """
        # El vault es virtual, pero limitamos operaciones basado en trading_capital
        effective = min(account_free_margin, self.trading_capital)
        return max(0, effective)
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado completo del Vault."""
        return {
            "initial_capital": self.initial_capital,
            "vault_balance": round(self.vault_balance, 2),
            "trading_capital": round(self.trading_capital, 2),
            "total_protected": round(self.vault_balance, 2),
            "total_transfers": self.total_vault_transfers,
            "last_trigger_equity": round(self.last_trigger_equity, 2),
            "next_trigger_at": round(self.last_trigger_equity * self.COMPOUNDING_MULTIPLIER, 2),
            "pending_alerts": len(self.pending_transfer_alerts)
        }
    
    def pop_pending_alert(self) -> Optional[Dict]:
        """Extrae y retorna la siguiente alerta pendiente."""
        if self.pending_transfer_alerts:
            return self.pending_transfer_alerts.pop(0)
        return None


# Singleton global del Vault
the_vault = TheVault(initial_capital=settings.INITIAL_CAPITAL if hasattr(settings, 'INITIAL_CAPITAL') else 30.0)


class RiskEngine:
    """
    Institutional Risk Management Engine.
    Handles adaptive lot sizing, drawdown protection, exposure limits, and Vault integration.
    """
    
    def __init__(self):
        self._daily_opening_balance: Optional[float] = None
        self._last_reset_date: Optional[str] = None
        self.vault = the_vault  # Integrate Vault

    async def _ensure_daily_balance(self):
        """Calcula o recupera el balance inicial del da para el drawdown."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            account_info = await mt5_conn.execute(mt5.account_info)
            if account_info:
                # En un sistema real, buscaramos el balance al 00:00 en el historial
                # Aqu usamos el balance actual como base si es el primer inicio del da
                self._daily_opening_balance = account_info.balance
                self._last_reset_date = today
                logger.info(f"RiskEngine: Daily balance reset. Base: ${self._daily_opening_balance:.2f}")

    async def calculate_dynamic_lot(self, confidence: float, volatility: float, symbol: str, sl_points: int = 200, market_data: Dict[str, Any] = {}) -> float:
        """
        [MODEL 3 REQUEST] Asignación Neuronal Dinámica.
        Integrado con Institutional Guards (POM Protocol).
        """
        allocation = await self.get_neural_allocation(confidence)
        
        if confidence < 0.60:
            return 0.01 # Minimum lot / Survival mode
        
        # 1. Get Circuit Breaker Multiplier (2%/4%/6% DD)
        from app.services.circuit_breaker import circuit_breaker
        cb_multiplier = await circuit_breaker.get_lot_multiplier()
        
        # 2. Visionary Physical Guardians: Volatility & Spread
        regime = volatility_guard.get_regime(market_data)
        can_trade_vol, vol_reason = volatility_guard.can_trade(market_data)
        
        # Check Spread Toxicity
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        current_spread = (tick.ask - tick.bid) if tick else 0
        avg_spread = market_data.get("avg_spread", 0)
        is_spread_toxic = spread_filter.is_spread_toxic(symbol, current_spread, avg_spread)

        if not can_trade_vol or is_spread_toxic or cb_multiplier <= 0:
            logger.warning(f"🛡️ RISK VETO on {symbol}: Regime={regime}, VolReason={vol_reason}, SpreadToxic={is_spread_toxic}, CB_Mult={cb_multiplier:.2f}")
            return 0.0
            
        if regime == "TURBULENT":
            regime_multiplier = 0.5 # 50% Reduction for turbulence
            logger.info(f"🌪️ TURBULENCE DETECTED: Scaling risk to 50% for {symbol}")

        # [LEVEL 41] BAYESIAN KELLY SIZING (The Body) - REFINED
        # Using exact user specification for Damped Kelly.
        
        uncertainty = market_data.get("brain_uncertainty", 0.05) 
        
        # Calculate Target Risk Percentage using Damped Kelly
        target_risk_pct = self._calculate_damped_kelly(confidence, uncertainty)
        
        # Convert Target Risk % to Neural Multiplier for get_adaptive_lots
        # get_adaptive_lots uses: risk_amount = equity * (settings.effective_risk_cap/100) * multiplier
        # We want: risk_amount = equity * target_risk_pct
        # So: multiplier = target_risk_pct / (settings.effective_risk_cap/100)
        
        base_risk_pct = settings.effective_risk_cap / 100.0
        if base_risk_pct <= 0: base_risk_pct = 0.01 # Avoid div by zero (fallback 1%)
        
        neural_multiplier = target_risk_pct / base_risk_pct
        
        # [LEVEL 52] PERFORMANCE ADAPTIVE SIZING (The Profit Pulse)
        from app.services.drift_monitor import drift_monitor
        drift_status = drift_monitor.check_drift()
        performance_multiplier = 1.0
        
        if drift_status["status"] != "INSUFFICIENT_DATA":
            pf = drift_status["metrics"].get("profit_factor", 1.5)
            # Reward PF > 1.5, Scale back if PF < 1.2
            if pf < 1.2:
                performance_multiplier = max(0.2, pf / 1.5)  # Damping
                logger.warning(f"📉 PERFORMANCE DAMPING: PF={pf:.2f} | Scaling risk to {performance_multiplier:.0%}")
            elif pf > 2.0:
                performance_multiplier = 1.2  # Institutional reward cap
                logger.info(f"🏆 PERFORMANCE REWARD: PF={pf:.2f} | Boosting risk to 120%")

        # Final Lot Calculation
        base_lot = await self.get_adaptive_lots(symbol, sl_points, neural_multiplier)
        final_lot = base_lot * cb_multiplier * regime_multiplier * performance_multiplier
        
        # 4. Shadow Force (Master Directive)
        trading_mode = os.getenv("TRADING_MODE", "SHADOW")
        if trading_mode == "SHADOW":
            logger.info(f"🌌 SHADOW EXECUTION: Signal valid, simulated lot: {final_lot:.2f} (KellyRisk: {target_risk_pct*100:.2f}%, Unc: {uncertainty:.3f})")
        
        return max(0.01, final_lot) if final_lot > 0 else 0.0

    def _calculate_damped_kelly(self, win_prob: float, uncertainty: float, risk_reward_ratio: float = 1.5) -> float:
        """
        [LEVEL 41] Damped Kelly Criterion (User Specified).
        
        Formula:
        1. Safety Clamp: Unc > MAX_UNCERTAINTY_THRESHOLD -> Risk = 0
        2. Kelly: f = (p(b+1) - 1) / b
        3. Damping: factor = 0.5 * (1 - (Unc / MAX_UNCERTAINTY_THRESHOLD))
        4. Max Risk: Cap at RISK_PER_TRADE_PERCENT
        """
        max_unc = getattr(settings, "MAX_UNCERTAINTY_THRESHOLD", 0.08)
        
        # 1. SAFETY CLAMP
        if uncertainty > max_unc:
            return 0.0
            
        # 2. GENERAL KELLY FORMULA
        kelly_fraction = (win_prob * (risk_reward_ratio + 1) - 1) / risk_reward_ratio
        
        if kelly_fraction <= 0:
            return 0.0
            
        # 3. UNCERTAINTY DAMPING
        damping_factor = 0.5 * (1.0 - (uncertainty / max_unc))
        final_fraction = kelly_fraction * max(0.0, damping_factor)
        
        # 4. HARD LIMITS (Account Protection)
        max_risk_per_trade = (settings.RISK_PER_TRADE_PERCENT / 100.0) if hasattr(settings, "RISK_PER_TRADE_PERCENT") else 0.02
        final_risk_pct = min(final_fraction, max_risk_per_trade)
        
        return final_risk_pct

    async def get_neural_allocation(self, alpha_confidence: float) -> Dict[str, float]:
        """
        Legacy allocation helper. Kept for aggressive/defensive labels only.
        """
        allocation = {
            "aggressiveness": "TEPID"
        }
        if alpha_confidence > 0.85: allocation["aggressiveness"] = "SNIPER"
        elif alpha_confidence > 0.70: allocation["aggressiveness"] = "ASSERTIVE"
        elif alpha_confidence < 0.60: allocation["aggressiveness"] = "DEFENSIVE"
            
        return allocation


    async def get_adaptive_lots(self, symbol: str, sl_points: int, neural_multiplier: float = 1.0) -> float:
        """
        Calculates lot size based on account equity, risk percent, SL distance AND Neural Multiplier.
        
        Formula: Lot = (Equity * Risk% * NeuralMult) / (SL_Points * PointValue)
        """
        if neural_multiplier <= 0:
            return 0.0
        
        if sl_points <= 0:
            logger.warning("[RISK] SL points is zero or negative, returning minimum lot")
            return 0.01
        
        try:
            account_info = await mt5_conn.execute(mt5.account_info)
            symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
            
            if not account_info or not symbol_info:
                logger.error("[RISK] Failed to get account/symbol info for lot calculation")
                return 0.01
            
            equity = account_info.equity
            risk_percent = settings.effective_risk_cap / 100  # Use context-aware risk cap
            
            # Point value calculation (value of 1 point movement for 1 lot)
            point_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size * symbol_info.point
            if point_value <= 0:
                point_value = 10  # Fallback for standard forex
            
            # Core calculation
            risk_amount = equity * risk_percent * neural_multiplier
            lot_size = risk_amount / (sl_points * point_value)
            
            # [SNIPER MICRO-SCALING]
            # If lot_size is e.g. 0.004 (< 0.01 min), normally we floor to 0.
            # But in Asymmetric Warfare ($20 account), we need to check if 0.01 is 'tolerable'.
            # Tolerable means: Actual Risk USD < Settings.MAX_RISK_USD ($6 for 30%)
            min_lot = symbol_info.volume_min
            if lot_size < min_lot and min_lot > 0:
                actual_risk_at_min = min_lot * sl_points * point_value
                max_tolerable_risk = equity * (settings.effective_risk_cap / 100) # e.g. 30%
                
                if actual_risk_at_min <= max_tolerable_risk:
                     logger.info(f"🔫 SNIPER FORCE: Calculated {lot_size:.4f} < Min {min_lot}, but Risk ${actual_risk_at_min:.2f} is within limit (${max_tolerable_risk:.2f}). Forcing {min_lot}.")
                     lot_size = min_lot
            
            # Clamp to symbol limits
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
            
            # Round to step
            lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
            lot_size = round(lot_size, 2)
            
            logger.debug(f"[RISK] Adaptive Lot: {lot_size} (Eq:{equity:.0f}, Risk:{risk_percent*100:.1f}%, SL:{sl_points}pts)")
            return lot_size
            
        except Exception as e:
            logger.error(f"[RISK] Lot calculation error: {e}")
            return 0.01

    async def check_drawdown_limit(self) -> bool:
        """
        Vetoes trading if the daily drawdown limit is reached.
        Phantom Mode: Pauses execution but continues analysis.
        """
        await self._ensure_daily_balance()
        account_info = await mt5_conn.execute(mt5.account_info)
        if not account_info or not self._daily_opening_balance:
            return False

        real_loss = self._daily_opening_balance - account_info.equity
        current_drawdown = (real_loss / self._daily_opening_balance) * 100
        
        if current_drawdown > settings.MAX_DAILY_DRAWDOWN_PERCENT:
            logger.warning(f" PHANTOM MODE ACTIVE: Daily DD {current_drawdown:.2f}% (Limit: {settings.MAX_DAILY_DRAWDOWN_PERCENT}%)")
            logger.warning("Operativa pausada. Iniciando recalibracin interna...")
            return False
        
        return True

    async def apply_trailing_stop(self, symbol: str, ticket: int, min_profit_pips: int = 10):
        """
        Aplica Trailing Stop basado en volatilidad (ATR).
        Si el precio se mueve 1.5 * ATR a favor, mueve el SL a Break Even.
        """
        pos = await mt5_conn.execute(mt5.positions_get, ticket=ticket)
        if not pos: return
        
        pos = pos[0]
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        
        # Obtener ATR para el clculo dinmico
        from app.skills.market import get_volatility_metrics
        vol = await get_volatility_metrics(symbol)
        atr = vol.get("atr", 0)
        if atr <= 0: return

        trail_points = (atr * settings.ATR_TRAILING_MULTIPLIER) / symbol_info.point
        
        if pos.type == mt5.POSITION_TYPE_BUY:
            current_profit_points = (tick.bid - pos.price_open) / symbol_info.point
            # Lgica Breakeven (1.5 * ATR)
            if current_profit_points > (trail_points) and pos.sl < pos.price_open:
                logger.info(f"RiskEngine: Moving ticket {ticket} to BREAKEVEN (ATR Trail)")
                await self._modify_sl(ticket, pos.price_open, pos.tp)
            
            # Trailing dinmico
            new_sl = tick.bid - (trail_points * symbol_info.point)
            if new_sl > pos.sl and new_sl < tick.bid:
                await self._modify_sl(ticket, new_sl, pos.tp)
        
        elif pos.type == mt5.POSITION_TYPE_SELL:
            current_profit_points = (pos.price_open - tick.ask) / symbol_info.point
            # Lgica Breakeven (1.5 * ATR)
            if current_profit_points > (trail_points) and (pos.sl == 0 or pos.sl > pos.price_open):
                logger.info(f"RiskEngine: Moving ticket {ticket} to BREAKEVEN (ATR Trail)")
                await self._modify_sl(ticket, pos.price_open, pos.tp)
                
            # Trailing dinmico
            new_sl = tick.ask + (trail_points * symbol_info.point)
            if (pos.sl == 0 or new_sl < pos.sl) and new_sl > tick.ask:
                await self._modify_sl(ticket, new_sl, pos.tp)

    async def _modify_sl(self, ticket: int, sl: float, tp: float):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp
        }
        result = await mt5_conn.execute(mt5.order_send, request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"RiskEngine: Trailing SL updated for ticket {ticket} -> {sl}")

    # =========================================================================
    # TWIN-ENGINE HYBRID STRATEGY
    # =========================================================================
    
    async def get_capital_allocation(self) -> Dict[str, Any]:
        """
        Regla 50/50 Dinmica: Divide el margen libre para Scalp vs Swing.
        """
        account = await mt5_conn.execute(mt5.account_info)
        if not account:
            return {"scalp_capital": 0, "swing_capital": 0, "can_dual": False}
        
        free_margin = account.margin_free
        equity = account.equity
        
        # 50/50 Split
        scalp_capital = free_margin * 0.5
        swing_capital = free_margin * 0.5
        
        # Can we open 2 trades of 0.01?
        can_dual = await self.can_open_dual_trade("XAUUSD")
        
        return {
            "scalp_capital": round(scalp_capital, 2),
            "swing_capital": round(swing_capital, 2),
            "free_margin": round(free_margin, 2),
            "equity": round(equity, 2),
            "can_dual": can_dual,
            "max_positions": await self.max_positions_allowed()
        }
    
    async def can_open_dual_trade(self, symbol: str) -> bool:
        """
        Verifica si hay margen suficiente para 2 operaciones de 0.01.
        """
        account = await mt5_conn.execute(mt5.account_info)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        
        if not account or not symbol_info:
            return False
        
        # Margin required for 1 lot (then scale to 0.01)
        margin_per_lot = symbol_info.margin_initial if symbol_info.margin_initial > 0 else 1000
        margin_for_micro = (margin_per_lot / 100) * 0.01  # 0.01 lot
        
        # We need margin for 2 micro trades
        required_margin = margin_for_micro * 2
        
        return account.margin_free >= required_margin
    
    async def max_positions_allowed(self) -> int:
        """
        Growth Trigger: Determina cuntas posiciones podemos tener basado en equity.
        """
        account = await mt5_conn.execute(mt5.account_info)
        if not account:
            return 1
        
        equity = account.equity
        
        # Escala: $20=2pos, $50=3pos, $100=4pos
        if equity >= 100:
            return 4
        elif equity >= settings.EQUITY_UNLOCK_THRESHOLD:
            return 3  # Unlock 3rd position
        elif equity >= settings.INITIAL_CAPITAL:
            return 2  # Twin-Engine mode
        else:
            return 1  # Survival mode - Scalp only

risk_engine = RiskEngine()
