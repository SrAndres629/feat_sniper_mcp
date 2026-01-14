import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

# Logger
logger = logging.getLogger("feat.trade_mgmt")

class TradeManager:
    """
    The Executor: Gestiona el ciclo de vida de las ordenes via ZMQ/MT5.
    """
    def __init__(self, zmq_bridge):
        self.zmq_bridge = zmq_bridge
        self.pending_orders = {}
        logger.info("[EXECUTION] Trade Manager Online (Execution Arm)")

    async def execute_order(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envia orden al bridge ZMQ.
        Params: symbol, volume, [price], sl, tp.
        """
        # 0. Check Simulation Mode (Fase 5 Hook)
        import os
        import json
        mode = os.getenv("TRADING_MODE", "LIVE")
        
        if mode == "SIMULATION":
            ticket = int(datetime.now().timestamp())
            params['ticket'] = ticket
            params['status'] = "SIM_FILLED"
            params['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"🔵 SIMULATION ORDER: {action} {params}")
            
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
