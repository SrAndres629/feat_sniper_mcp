import logging
import sqlite3
import pandas as pd
from typing import Dict, Any, List
import os
import anyio

logger = logging.getLogger("MT5_Bridge.Skills.UnifiedModel")

# Ruta por defecto asumiendo estructura del proyecto
DB_DEFAULT_PATH = os.path.join(os.getcwd(), "FEAT_Sniper_Master_Core", "Python", "unified_model.db")

class UnifiedModelDB:
    def __init__(self, db_path: str = DB_DEFAULT_PATH):
        self.db_path = db_path

    def _get_connection(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        return sqlite3.connect(self.db_path)

    def _run_query_sync(self, sql_query: str) -> Dict[str, Any]:
        """Helper sincrónico para ejecutar la query en un hilo separado."""
        conn = None
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(sql_query, conn)
            return {
                "status": "success",
                "rows_count": len(df),
                "data": df.to_dict('records')
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            if conn:
                conn.close()

    async def query_custom_sql(self, sql_query: str) -> Dict[str, Any]:
        """
        Ejecuta una consulta SQL de solo lectura sobre el Unified Model.
        """
        # Seguridad básica: impedir DROP, DELETE, INSERT, UPDATE
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
        if any(cmd in sql_query.upper() for cmd in forbidden):
             return {"status": "error", "message": "Only SELECT queries are allowed for safety."}

        return await anyio.to_thread.run_sync(self._run_query_sync, sql_query)

    async def get_fsm_transition_matrix(self, period: str = "H1") -> Dict[str, Any]:
        """
        Recupera la matriz de probabilidad de transición de estados de la FSM.
        """
        # Ejemplo hipotético de consulta
        sql = f"""
            SELECT previous_state, current_state, count(*) as frequency
            FROM fsm_transitions
            WHERE timeframe = '{period}'
            GROUP BY previous_state, current_state
            ORDER BY frequency DESC
        """
        # Si la tabla no existe, fallará elegantemente
        return await self.query_custom_sql(sql)

unified_db = UnifiedModelDB()
