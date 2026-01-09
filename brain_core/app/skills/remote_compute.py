import logging
import paramiko
import os
from typing import Optional, Dict
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.RemoteCompute")

class RemoteWorker:
    """
    Gestor de ejecución remota vía SSH para Offloading de cómputo.
    """
    def __init__(self):
        self.host = settings.SSH_HOST
        self.user = settings.SSH_USER
        self.key_path = settings.SSH_KEY_PATH
        self.enabled = settings.ENABLE_SSH_OFFLOADING

    def _get_client(self) -> Optional[paramiko.SSHClient]:
        if not self.enabled or not self.host or not self.user:
            return None
            
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if self.key_path and os.path.exists(self.key_path):
                client.connect(self.host, username=self.user, key_filename=self.key_path, timeout=10)
            else:
                # Si no hay llave, asumimos que se requiere configuración o falla
                logger.warning("SSH Key no encontrada o no configurada.")
                return None
                
            return client
        except Exception as e:
            logger.error(f"Error conectando al Worker Remoto SSH: {e}")
            return None

    async def execute_remote_cmd(self, command: str) -> Dict:
        """Ejecuta un comando en el servidor remoto y retorna la salida."""
        client = self._get_client()
        if not client:
            return {"status": "error", "message": "SSH Worker no configurado o inaccesible."}

        try:
            logger.info(f"🛰️ Enviando comando al Worker Remoto: {command}")
            import anyio
            
            def run_cmd():
                stdin, stdout, stderr = client.exec_command(command)
                return {
                    "stdout": stdout.read().decode(),
                    "stderr": stderr.read().decode(),
                    "exit_code": stdout.channel.recv_exit_status()
                }

            result = await anyio.to_thread.run_sync(run_cmd)
            return {"status": "success", **result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            client.close()

remote_worker = RemoteWorker()

def register_remote_skills(mcp):
    """Registra las herramientas de cómputo remoto en el MCP."""
    
    @mcp.tool(name="skill_execute_remote_task")
    async def execute_remote_task(command: str):
        """
        Ejecuta una tarea de computo pesado (ML training, Optuna) en un servidor remoto via SSH.
        Requiere configuración previa en .env (SSH_HOST, SSH_USER, SSH_KEY_PATH).
        """
        return await remote_worker.execute_remote_cmd(command)

    @mcp.tool(name="skill_check_remote_status")
    async def check_remote_status():
        """Verifica la conectividad con el servidor de compute offloading."""
        client = remote_worker._get_client()
        if client:
            client.close()
            return {"status": "connected", "host": remote_worker.host}
        return {"status": "disconnected", "reason": "Configuración incompleta o timeout."}
