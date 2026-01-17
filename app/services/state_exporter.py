import json
import os
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("feat.exporter")

class StateExporter:
    def __init__(self, file_path="data/live_state.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    async def export(self, state_data: dict):
        """
        Exports the current Nexus state to a JSON file for the dashboard.
        Include metadata like timestamp for heartbeat detection.
        """
        try:
            full_state = {
                "timestamp": datetime.now().isoformat(),
                "status": "OPERATIONAL",
                **state_data
            }
            
            # Atomic write to avoid partial reads by Streamlit
            tmp_path = self.file_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(full_state, f, indent=2)
            
            # Use os.replace for atomic operation on Windows/Linux
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            os.rename(tmp_path, self.file_path)
            
        except Exception as e:
            logger.error(f"Failed to export state: {e}")

state_exporter = StateExporter()
