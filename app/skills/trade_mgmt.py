import logging
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Logger
logger = logging.getLogger("feat.trade_mgmt")

from app.core.config import settings

class TradeManager:
    """
    The Executor: Gestiona el ciclo de vida de las ordenes via ZMQ/MT5.
    Phase 13: Includes Exhaustion Exit & Time-Limit logic.
    """
    def __init__(self, zmq_bridge):
        self.zmq_bridge = zmq_bridge
        self.pending_orders = {}
        self.active_positions = {} # {ticket: {open_time, entry_price, atr, regime}}
        self.start_time = time.time()
        
        # [LEVEL 57] Doctoral Config Injection
        self.exhaustion_atr_threshold = settings.EXHAUSTION_ATR_THRESHOLD
        self.scalp_time_limit_seconds = settings.SCALP_TIME_LIMIT_SECONDS
        self.warmup_period = settings.WARMUP_PERIOD_SECONDS
        
        logger.info(f"[EXECUTION] Trade Manager Online (Warm-up: {self.warmup_period}s, Exhaustion Exit: ON)")

    async def execute_order(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envia orden al bridge ZMQ.
        Warm-up Protocol: Blocks executions for first 60s to stabilize physics.
        """
        warmup_remaining = self.warmup_period - (time.time() - self.start_time)
        if warmup_remaining > 0:
            logger.warning(f"[WAIT] WARM-UP ACTIVE: Blocking {action}. {int(warmup_remaining)}s remaining.")
            return {"status": "WAITING_FOR_WARMUP", "remaining": int(warmup_remaining)}
        # 0. Check Simulation Mode (Fase 5 Hook)
        mode = settings.TRADING_MODE
        
        if mode == "SHADOW" or mode == "PAPER":
            import json
            ticket = int(time.time() % 100000) 
            params['ticket'] = ticket
            params['status'] = f"{mode}_VALID"
            params['timestamp'] = datetime.now().isoformat()
            
            # Phase 12: HUD Projection via ZMQ
            from app.core.zmq_bridge import zmq_bridge
            asyncio.create_task(zmq_bridge.send_command(
                "SHADOW_RESULT",
                shadow_action=action, # Renamed to avoid duplicate 'action'
                symbol=params.get('symbol'),
                lot=params.get('volume'),
                p_win=params.get('p_win', 0.5),
                regime=params.get('regime', 'LAMINAR')
            ))
            
            logger.info(f"🌌 {mode} EXECUTION (PILOT): {action} {params}")
            # El Visionario manda: Shadow Mode es Sagrado. 
            # No se envían órdenes al ZMQ Bridge reales.
            
            # Save to Sim Ledger
            try:
                ledger_path = "sim_ledger.json"
                entry = {"action": action, **params}
                if os.path.exists(ledger_path):
                    with open(ledger_path, 'r+') as f:
                        data = json.load(f)
                        data.append(entry)
                        f.seek(0)
                        json.dump(data, f, indent=2)
                else:
                    with open(ledger_path, 'w') as f:
                        json.dump([entry], f, indent=2)
            except Exception as e:
                logger.error(f"Sim Ledger Write Error: {e}")
                
            return {"retcode": 0, "ticket": ticket, "comment": "SIM_OK"}

        try:
            # Validar Inputs Basicos
            symbol = params.get("symbol", "XAUUSD")
            volume = float(params.get("volume", 0.01))
            
            # Construir Payload ZMQ
            payload = {
                "action": action.upper(),
                "symbol": symbol,
                "volume": volume,
                "magic": settings.MT5_MAGIC_NUMBER,
                "comment": settings.MT5_ORDER_COMMENT
            }
            
            # Parametros opcionales
            if "price" in params: payload["price"] = float(params["price"])
            if "sl" in params: payload["sl"] = float(params["sl"])
            if "tp" in params: payload["tp"] = float(params["tp"])
            if "ticket" in params: payload["ticket"] = int(params["ticket"])

            # 0.5 Physical Rate Limiting & Safety Guard
            from app.services.circuit_breaker import circuit_breaker
            if not circuit_breaker.can_execute():
                status = circuit_breaker.get_hierarchical_status()
                logger.error(f"🚨 ORDER BLOCKED BY SRE: {status}")
                return {"status": "BLOCKED_BY_GAURDIAN", "reason": status}

            logger.info(f"Sending Execution Command: {action} on {symbol}")
            
            # REAL ACK IMPLEMENTATION: Send command and wait for confirmation
            if self.zmq_bridge:
                # import asyncio - REDUNDANT/SHADOWING REMOVED
                from app.core.config import settings
                
                max_retries = settings.MAX_ORDER_RETRIES
                base_backoff_ms = settings.RETRY_BACKOFF_BASE_MS
                timeout_seconds = 5.0
                
                last_error = None
                for attempt in range(1, max_retries + 1):
                    try:
                        logger.info(f"[ACK] Attempt {attempt}/{max_retries} for {action}")
                        
                        # Send command and await response (with timeout)
                        response = await asyncio.wait_for(
                            self.zmq_bridge.send_command_with_ack(action, **params),
                            timeout=timeout_seconds
                        )
                        
                        # Parse ACK response
                        if response and response.get("result") == "OK":
                            ticket = response.get("ticket", 0)
                            logger.info(f"[ACK] Order confirmed: Ticket #{ticket}")
                            return {
                                "status": "EXECUTED",
                                "ticket": ticket,
                                "timestamp": datetime.now().isoformat(),
                                "attempts": attempt
                            }
                        else:
                            error_msg = response.get("error", "Unknown error") if response else "No response"
                            logger.warning(f"[ACK] Order rejected: {error_msg}")
                            last_error = error_msg
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"[ACK] Timeout on attempt {attempt}")
                        last_error = "Timeout waiting for MT5 response"
                    except Exception as e:
                        logger.warning(f"[ACK] Error on attempt {attempt}: {e}")
                        last_error = str(e)
                    
                    # Exponential backoff before retry
                    if attempt < max_retries:
                        backoff_ms = base_backoff_ms * (3 ** (attempt - 1))  # 50 -> 150 -> 450ms
                        await asyncio.sleep(backoff_ms / 1000.0)
                
                # All retries exhausted
                logger.error(f"[ACK] All {max_retries} attempts failed: {last_error}")
                return {
                    "status": "FAILED",
                    "ticket": 0,
                    "error": last_error,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.warning("No ZMQ bridge - order not sent")
                return {"status": "NO_BRIDGE", "ticket": 0}

        except Exception as e:
            logger.error(f"Execution Failed: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def modify_position(self, ticket: int, sl: float, tp: float) -> Dict:
        return await self.execute_order("MODIFY", {"ticket": ticket, "sl": sl, "tp": tp})

    async def close_position(self, ticket: int, profit_pips: float = 0.0) -> Dict:
        """Closes a position and sends feedback to the ML engine & RLAIF Critic."""
        res = await self.execute_order("CLOSE", {"ticket": ticket})
        
        if ticket in self.active_positions:
            pos = self.active_positions[ticket]
            
            # 1. RLAIF Integration (AI Critic)
            try:
                from app.ml.rlaif_critic import rlaif_critic
                trade_result = {
                     "ticket": ticket,
                     "symbol": pos.get("symbol", "XAUUSD"),
                     "entry_price": pos["entry_price"],
                     "profit": profit_pips,
                     "duration": (datetime.now() - pos["open_time"]).total_seconds(),
                     "neural_alpha": pos.get("neural_alpha", 1.0),
                     "volatility_regime": pos.get("volatility_regime", "LAMINAR")
                }
                trade_context = pos.get("context", {})
                rlaif_critic.critique_trade(trade_result, trade_context)
                logger.info(f"👨‍⚖️ RLAIF Critic Judged Ticket {ticket}: Profit={profit_pips:.2f}")
            except Exception as e:
                logger.error(f"RLAIF Link Error: {e}")

            # 2. Online Learning & Metrics Update
            try:
                from app.ml.ml_engine import ml_engine
                if ml_engine:
                     ml_engine.record_trade_result(
                         direction_prob=pos.get("neural_prob", 0.5), 
                         win_confidence=pos.get("win_confidence", 0.5),
                         result_pips=profit_pips,
                         alpha=pos.get("neural_alpha", 1.0),
                         regime=pos.get("volatility_regime", "LAMINAR")
                     )
                
                # Update Supabase Stats
                from app.services.supabase_sync import supabase_sync
                await supabase_sync.log_trade_execution({
                    "ticket": ticket,
                    "action": "CLOSE",
                    "profit": profit_pips,
                    "status": "SUCCESS" if res.get("status") == "EXECUTED" else "ERROR"
                })
            except Exception as e:
                logger.error(f"ML Feedback Error: {e}")
                 
            self.unregister_position(ticket)
            
        return res

    async def close_all_positions(self) -> Dict:
        return await self.execute_order("CLOSE_ALL", {})

    async def check_exhaustion_exit(self, ticket: int, current_price: float, 
                                     current_l4_slope: float, current_time: datetime) -> Dict:
        """
        Phase 13: Exhaustion Exit Logic.
        Monitors open positions and triggers early exit or breakeven based on physics.
        """
        if ticket not in self.active_positions:
            return {"status": "NO_POSITION", "ticket": ticket}
        
        pos = self.active_positions[ticket]
        entry_price = pos.get("entry_price", current_price)
        atr = pos.get("atr", 0.0)
        open_time = pos.get("open_time", current_time)
        is_buy = pos.get("is_buy", True)
        
        # Calculate current profit in ATR units
        price_diff = (current_price - entry_price) if is_buy else (entry_price - current_price)
        atr_profit = price_diff / atr if atr > 0 else 0
        
        # Time elapsed
        time_elapsed = (current_time - open_time).total_seconds()
        
        result = {"status": "HOLDING", "ticket": ticket, "atr_profit": round(atr_profit, 2)}
        
        # Rule 1: Exhaustion Exit (0.5 ATR profit + Slope dying)
        if atr_profit >= self.exhaustion_atr_threshold:
            if current_l4_slope < 0:
                # Physics exhaustion detected - Move to Breakeven
                # Symbol Agnostic Pip logic: Gold(0.01), Forex(0.0001)
                pip_val = 0.01 if "XAU" in pos.get("symbol", "") or "JPY" in pos.get("symbol", "") else 0.0001
                new_sl = entry_price + (pip_val if is_buy else -pip_val) # +1 pip
                logger.info(f"🛡️ EXHAUSTION EXIT: Ticket {ticket} moving SL to Breakeven (L4_Slope={current_l4_slope:.4f})")
                await self.modify_position(ticket, sl=new_sl, tp=pos.get("tp", 0))
                result["status"] = "BREAKEVEN_SET"
                result["reason"] = "Exhaustion (L4_Slope < 0 at 0.5 ATR)"
        
        # Rule 2: Time-Limit Exit (5 minutes for scalping)
        if time_elapsed > self.scalp_time_limit_seconds:
            if abs(current_l4_slope) < 0.01: # Physics erratic/neutral
                logger.warning(f"⏰ TIME-LIMIT EXIT: Ticket {ticket} closed (Elapsed: {time_elapsed:.0f}s, L4_Slope erratic)")
                await self.close_position(ticket)
                result["status"] = "TIME_EXIT"
                result["reason"] = f"Time limit exceeded ({time_elapsed:.0f}s)"
        
        return result

    async def update_positions_logic(self, probability: float):
        """
        Active Management Loop (The Hands).
        Checks if open positions should be closed based on probability decay (AI Feedback).
        """
        if not self.active_positions:
            return

        to_close = []
        for ticket, pos in self.active_positions.items():
            is_buy = pos.get("is_buy", True)
            
            # Probability Decay Rules
            # If BUY and Prob drops below 0.40 -> CLOSE
            if is_buy and probability < 0.40:
                 to_close.append((ticket, "Prob Decay (Buy)"))
            # If SELL and Prob rises above 0.60 -> CLOSE
            elif not is_buy and probability > 0.60:
                 to_close.append((ticket, "Prob Reversal (Sell)"))
                 
        for ticket, reason in to_close:
             logger.info(f"⚡ ACTIVE MGMT: Closing Ticket {ticket} due to {reason}. Prob={probability:.2f}")
             await self.close_position(ticket)

    def register_position(self, ticket: int, entry_price: float, atr: float, is_buy: bool, tp: float = 0, context: Dict = None):
        """Registers a new position for Exhaustion Exit monitoring."""
        self.active_positions[ticket] = {
            "entry_price": entry_price,
            "atr": atr,
            "is_buy": is_buy,
            "tp": tp,
            "open_time": datetime.now(),
            "context": context or {}
        }
        logger.info(f"📝 Position Registered: Ticket {ticket}, Entry: {entry_price}, Context_Keys: {list(context.keys()) if context else 'None'}")

    def unregister_position(self, ticket: int):
        """Removes a closed position from monitoring."""
        if ticket in self.active_positions:
            del self.active_positions[ticket]
            logger.info(f"🗑️ Position Unregistered: Ticket {ticket}")

    async def cleanup(self):
        """Cleanup logic for TradeManager."""
        self.pending_orders.clear()
        self.active_positions.clear()
        logger.info("[EXECUTION] TradeManager cleanup complete.")
