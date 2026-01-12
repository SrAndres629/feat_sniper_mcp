"""
RAG Memory Engine - Memoria Persistente con ChromaDB
=====================================================
Proporciona almacenamiento y bsqueda semntica de informacin
para que la IA tenga "memoria ilimitada" persistente.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("MT5_Bridge.RAGMemory")

# Lazy imports para optimizar arranque
_chromadb_client = None
_collection = None
_embedding_function = None


def _get_embedding_function():
    """Lazy load del modelo de embeddings."""
    global _embedding_function
    if _embedding_function is None:
        try:
            from chromadb.utils import embedding_functions
            _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info(" Modelo de embeddings cargado: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise
    return _embedding_function


def _get_collection():
    """Lazy load de la coleccin ChromaDB."""
    global _chromadb_client, _collection
    if _collection is None:
        try:
            import chromadb
            from chromadb.config import Settings
            
            persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
            os.makedirs(persist_dir, exist_ok=True)
            
            _chromadb_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            _collection = _chromadb_client.get_or_create_collection(
                name="feat_sniper_memory",
                embedding_function=_get_embedding_function(),
                metadata={"description": "FEAT Sniper AI Memory Store"}
            )
            
            logger.info(f" ChromaDB inicializado en {persist_dir} con {_collection.count()} memorias")
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            raise
    return _collection


class RAGMemory:
    """
    Singleton para gestin de memoria RAG.
    
    Uso:
        from app.services.rag_memory import rag_memory
        
        # Almacenar
        rag_memory.store("El EUR/USD tiene soporte en 1.0850", category="analysis")
        
        # Buscar
        results = rag_memory.search("soporte EUR/USD", k=5)
        
        # Olvidar
        rag_memory.forget(category="analysis")
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGMemory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("RAGMemory Singleton inicializado (lazy load).")
    
    def store(
        self, 
        text: str, 
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Almacena texto en memoria permanente.
        
        Args:
            text: Texto a almacenar
            category: Categora para organizacin (analysis, trade, news, etc.)
            metadata: Metadatos adicionales opcionales
        
        Returns:
            ID nico del documento almacenado
        """
        collection = _get_collection()
        
        doc_id = f"{category}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        doc_metadata = {
            "category": category,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "mcp_tool"
        }
        if metadata:
            doc_metadata.update(metadata)
        
        collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[doc_metadata]
        )
        
        logger.info(f" Memoria almacenada: {doc_id[:30]}... (categoria: {category})")
        return doc_id
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Bsqueda semntica en memoria.
        
        Args:
            query: Texto de bsqueda
            k: Nmero mximo de resultados
            category: Filtrar por categora (opcional)
        
        Returns:
            Lista de documentos relevantes con score
        """
        collection = _get_collection()
        
        where_filter = {"category": category} if category else None
        
        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count() or 1),
            where=where_filter
        )
        
        # Formatear resultados
        formatted = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "id": results["ids"][0][i] if results["ids"] else None,
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None
                })
        
        logger.info(f" Bsqueda '{query[:30]}...' - {len(formatted)} resultados")
        return formatted
    
    def forget(
        self, 
        category: Optional[str] = None,
        doc_ids: Optional[List[str]] = None
    ) -> int:
        """
        Elimina memorias por categora o IDs especficos.
        
        Args:
            category: Eliminar todas las memorias de esta categora
            doc_ids: Lista de IDs especficos a eliminar
        
        Returns:
            Nmero de documentos eliminados
        """
        collection = _get_collection()
        
        if doc_ids:
            collection.delete(ids=doc_ids)
            count = len(doc_ids)
        elif category:
            # Obtener IDs por categora
            results = collection.get(where={"category": category})
            if results["ids"]:
                collection.delete(ids=results["ids"])
                count = len(results["ids"])
            else:
                count = 0
        else:
            # Sin filtro = limpiar todo (peligroso)
            count = collection.count()
            # No permitir borrado total sin confirmacin explcita
            logger.warning(" Intento de borrar toda la memoria sin filtro. Operacin cancelada.")
            return 0
        
        logger.info(f" {count} memorias eliminadas (categoria: {category or 'especfico'})")
        return count
    
    def count(self, category: Optional[str] = None) -> int:
        """Cuenta el nmero de memorias almacenadas."""
        collection = _get_collection()
        if category:
            results = collection.get(where={"category": category})
            return len(results["ids"]) if results["ids"] else 0
        return collection.count()
    
    def get_categories(self) -> List[str]:
        """Obtiene lista de categoras nicas en la memoria."""
        collection = _get_collection()
        results = collection.get()
        if results["metadatas"]:
            categories = set(m.get("category", "unknown") for m in results["metadatas"])
            return list(categories)
        return []


# Singleton exportable
rag_memory = RAGMemory()
