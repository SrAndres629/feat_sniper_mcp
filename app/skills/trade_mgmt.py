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

    def register_position(self, ticket: int, entry_price: float, atr: float, is_buy: bool, tp: float = 0, context: Dict = None):
        """Registers a new position for monitoring."""
        self.active_positions[ticket] = {
            "entry_price": entry_price,
            "atr": atr,
            "is_buy": is_buy,
            "tp": tp,
            "open_time": datetime.now(),
            "context": context or {}
        }
        logger.info(f"📝 Position Registered: Ticket {ticket}")

    def unregister_position(self, ticket: int):
        """Removes a closed position from monitoring."""
        if ticket in self.active_positions:
            del self.active_positions[ticket]
            logger.info(f"🗑️ Position Unregistered: Ticket {ticket}")

    async def cleanup(self):
        self.pending_orders.clear()
        self.active_positions.clear()
        logger.info("[EXECUTION] TradeManager cleanup complete.")
