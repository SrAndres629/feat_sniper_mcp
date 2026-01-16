import os
import asyncio
import json
import logging
from datetime import datetime
from supabase import create_client, Client

class NexusMemory:
    """
    Institutional-Grade Evolutionary Memory System.
    Synchronizes local neural states with Supabase for long-term genetic recall.
    """
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        self.client: Client = None
        self.logger = logging.getLogger("NexusMemory")
        self.enabled = False
        
        if self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                self.enabled = True
                self.logger.info("✅ Evolutionary Memory Connected (Supabase)")
            except Exception as e:
                self.logger.error(f"❌ Memory Link Failed: {e}")
        else:
            self.logger.warning("⚠️ Supabase Credentials Missing - Memory Disabled")

    async def save_tick(self, tick_data: dict, table_name="market_ticks"):
        """Fire-and-forget save of market data"""
        if not self.enabled: return
        try:
            # Run in executor to avoid blocking event loop
            await asyncio.to_thread(self._insert, table_name, tick_data)
        except Exception as e:
            self.logger.error(f"Memory Write Error ({table_name}): {e}")

    async def save_concept(self, name: str, definition: dict):
        """Register or update a neural concept"""
        if not self.enabled: return
        payload = {
            "name": name,
            "definition": definition,
            "last_updated": datetime.utcnow().isoformat()
        }
        try:
            await asyncio.to_thread(self._upsert, "neural_concepts", payload, "name")
        except Exception as e:
            self.logger.error(f"Concept Save Error: {e}")

    async def save_evolution(self, generation: int, genome: dict, fitness: float):
        """Archive a generation's genome"""
        if not self.enabled: return
        payload = {
            "generation": generation,
            "genome": genome,
            "fitness_score": fitness,
            "created_at": datetime.utcnow().isoformat()
        }
        try:
            await asyncio.to_thread(self._insert, "neural_evolutions", payload)
        except Exception as e:
            self.logger.error(f"Evolution Archive Error: {e}")

    def _insert(self, table, data):
        self.client.table(table).insert(data).execute()

    def _upsert(self, table, data, on_conflict):
        self.client.table(table).upsert(data, on_conflict=on_conflict).execute()

# Singleton
nexus_memory = NexusMemory()
