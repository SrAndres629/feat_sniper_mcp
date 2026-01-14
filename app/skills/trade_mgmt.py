import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

# Logger
logger = logging.getLogger("feat.trade_mgmt")

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
        
        # Phase 13: Exhaustion Exit Parameters
        self.exhaustion_atr_threshold = 0.5 # 50% ATR profit triggers BE
        self.scalp_time_limit_seconds = 300 # 5 minutes
        
        logger.info("[EXECUTION] Trade Manager Online (Warm-up: 60s, Exhaustion Exit: ON)")

    async def execute_order(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envia orden al bridge ZMQ.
        Warm-up Protocol: Blocks executions for first 60s to stabilize physics.
        """
        warmup_remaining = 60 - (time.time() - self.start_time)
        if warmup_remaining > 0:
            logger.warning(f"🕒 WARM-UP ACTIVE: Blocking {action}. {int(warmup_remaining)}s remaining.")
            return {"status": "WAITING_FOR_WARMUP", "remaining": int(warmup_remaining)}
        # 0. Check Simulation Mode (Fase 5 Hook)
        import os
        import json
        mode = os.getenv("TRADING_MODE", "SHADOW") # Default to Shadow for Safety
        
        if mode == "SHADOW" or mode == "SIMULATION":
            params['ticket'] = ticket
            params['status'] = f"{mode}_VALID"
            params['timestamp'] = datetime.now().isoformat()
            
            # Phase 12: HUD Projection via ZMQ
            from app.core.zmq_bridge import zmq_bridge
            asyncio.create_task(zmq_bridge.send_command(
                "SHADOW_RESULT",
                action=action,
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
                "magic": 123456, # FEAT Magic Number
                "comment": "FEAT_AI_Sniper"
            }
            
            # Parametros opcionales
            if "price" in params: payload["price"] = float(params["price"])
            if "sl" in params: payload["sl"] = float(params["sl"])
            if "tp" in params: payload["tp"] = float(params["tp"])
            if "ticket" in params: payload["ticket"] = int(params["ticket"])

            logger.info(f"Sending Execution Command: {action} on {symbol}")
            
            # Enviar comando async (fire and forget por ahora, o request/reply si bridge soporta)
            if self.zmq_bridge:
                # Asumimos que zmq_bridge tiene un metodo para enviar comandos
                # Si es PUB/SUB, publicamos. Si es REQ/REP, esperamos.
                # En mcp_server.py original era PULL/PUSH o PUB/SUB.
                # Usaremos un metodo generico 'send_command' si existe, o 'broadcast'.
                # Revisar zmq_bridge implementation...
                # Asumiremos send_string o similar.
                # Para simplificar Módulo 6, mockeamos el envio real si no hay metodo claro.
                 # TODO: Implementar ACK real.
                # ACTIVATION: Sending command via ZMQ PUB socket
                await self.zmq_bridge.send_command(action, **params) 
            
            return {
                "status": "SENT",
                "ticket": 0, # Pending confirmation
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Execution Failed: {e}")
            return {"status": "ERROR", "error": str(e)}

    async def modify_position(self, ticket: int, sl: float, tp: float) -> Dict:
        return await self.execute_order("MODIFY", {"ticket": ticket, "sl": sl, "tp": tp})

    async def close_position(self, ticket: int) -> Dict:
        return await self.execute_order("CLOSE", {"ticket": ticket})

    async def close_all_positions(self) -> Dict:
        return await self.execute_order("CLOSE_ALL", {})

    async def check_exhaustion_exit(self, ticket: int, current_price: float, 
                                     current_l4_slope: float, current_time: datetime) -> Dict:
        """
        Phase 13: Exhaustion Exit Logic.
        Monitors open positions and triggers early exit or breakeven based on physics.
        
        Rules:
        1. If price reaches 0.5 ATR profit AND L4_Slope < 0 (momentum dying), move SL to Breakeven+1pip.
        2. If position open > 5 minutes AND physics becomes erratic, close at market.
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
                new_sl = entry_price + (0.0001 if is_buy else -0.0001) # +1 pip
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

    def register_position(self, ticket: int, entry_price: float, atr: float, is_buy: bool, tp: float = 0):
        """Registers a new position for Exhaustion Exit monitoring."""
        self.active_positions[ticket] = {
            "entry_price": entry_price,
            "atr": atr,
            "is_buy": is_buy,
            "tp": tp,
            "open_time": datetime.now()
        }
        logger.info(f"📝 Position Registered: Ticket {ticket}, Entry: {entry_price}, ATR: {atr}")

    def unregister_position(self, ticket: int):
        """Removes a closed position from monitoring."""
        if ticket in self.active_positions:
            del self.active_positions[ticket]
            logger.info(f"🗑️ Position Unregistered: Ticket {ticket}")
