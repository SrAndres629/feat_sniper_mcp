import os
import logging
import sys
from fastmcp import FastMCP
import chromadb
from chromadb.utils import embedding_functions
from db_engine import UnifiedModelDB
from datetime import datetime
from typing import List, Optional

# Configuracin de Logging para MCP
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("FEAT_Memory_Brain")

# Rutas - Basadas en la ubicacin del archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "unified_model.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Inicializar FastMCP
mcp = FastMCP("FEAT_Memory_Brain")

# Inicializar ChromaDB (Persistente)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Usar SentenceTransformers para embeddings de alta calidad localmente
# Esto descargar el modelo 'all-MiniLM-L6-v2' la primera vez (aprox 80MB)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Crear o recuperar coleccin
collection = chroma_client.get_or_create_collection(
    name="market_memories", 
    embedding_function=sentence_transformer_ef
)

@mcp.tool()
async def ingest_memories(days: int = 30):
    """
    Extrae la historia narrativa de la base de datos SQLite y la indexa en la memoria vectorial.
    Llamar peridicamente para mantener la memoria actualizada.
    """
    logger.info(f"Iniciando ingesta de memorias (ltimos {days} das)...")
    
    if not os.path.exists(DB_PATH):
        return f"Error: No se encontr la base de datos en {DB_PATH}"

    db = UnifiedModelDB(DB_PATH)
    try:
        narratives = db.get_narrative_history(days=days)
        if not narratives:
            return "No se encontraron nuevas memorias narrativas en el periodo especificado."
        
        # Generar IDs nicos basados en timestamp para evitar duplicados en la sesin
        batch_ids = [f"mem_{datetime.now().timestamp()}_{i}" for i in range(len(narratives))]
        
        collection.add(
            documents=narratives,
            ids=batch_ids
        )
        
        return f"xito: Se han procesado e indexado {len(narratives)} fragmentos de memoria narrativa en el almacn vectorial."
    except Exception as e:
        logger.error(f"Fallo en la ingesta: {str(e)}")
        return f"Error durante la ingesta: {str(e)}"
    finally:
        db.close()

@mcp.tool()
async def query_memory(question: str):
    """
    Realiza una bsqueda semntica en la memoria histrica del bot. 
    Ejemplo: 'Qu pas la ltima vez que el score fue bajo?' o 'Busca patrones de expansin en EURUSD'.
    """
    logger.info(f"Consultando memoria vectorial para: '{question}'")
    
    try:
        results = collection.query(
            query_texts=[question],
            n_results=5
        )
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            return "No se han encontrado recuerdos relevantes. Has ejecutado 'ingest_memories' primero?"
            
        context_list = results['documents'][0]
        formatted_results = "\n\n---\n\n".join(context_list)
        
        return f"### Memoria Recuperada (Top 5 eventos similares):\n\n{formatted_results}"
    except Exception as e:
        logger.error(f"Error en consulta: {str(e)}")
        return f"Error al consultar la memoria: {str(e)}"

if __name__ == "__main__":
    # Iniciar servidor MCP
    mcp.run()
