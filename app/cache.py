"""
app/cache.py
Módulo de Caché Semántico para Preguntas Frecuentes del Chat.
Permite responder consultas repetidas o parafraseadas con 0 tokens de LLM y latencia instantánea (<50ms).
"""

import time
from typing import Dict, Any, Optional, List
import numpy as np

class SemanticCacheManager:
    def __init__(self, max_size: int = 150, similarity_threshold: float = 0.92):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.entries: List[Dict[str, Any]] = []

    def search(self, query_vector: List[float], unidad: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Busca si existe una consulta previa en caché con una similitud del coseno >= threshold.
        """
        if not self.entries or not query_vector:
            return None

        q_vec = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)
        if norm_q == 0:
            return None

        best_score = 0.0
        best_entry = None

        for entry in self.entries:
            # Si se especifica unidad, comparar solo entradas de la misma unidad
            if unidad is not None and entry.get("unidad") is not None and entry.get("unidad") != unidad:
                continue

            c_vec = entry["query_vector"]
            norm_c = np.linalg.norm(c_vec)
            if norm_c == 0:
                continue

            similarity = float(np.dot(q_vec, c_vec) / (norm_q * norm_c))
            if similarity > best_score:
                best_score = similarity
                best_entry = entry

        if best_score >= self.similarity_threshold and best_entry is not None:
            return {
                "score": round(best_score, 4),
                "query_text": best_entry["query_text"],
                "response": best_entry["response"],
                "sources": best_entry["sources"]
            }

        return None

    def add(self, query_text: str, query_vector: List[float], response: str, sources: List[Dict[str, Any]], unidad: Optional[int] = None):
        """
        Almacena una nueva respuesta exitosa en la caché semántica en memoria.
        """
        if not response or not query_vector:
            return

        # Limitar tamaño de caché eliminando el elemento más antiguo (FIFO)
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)

        self.entries.append({
            "query_text": query_text,
            "query_vector": np.array(query_vector, dtype=np.float32),
            "response": response,
            "sources": sources,
            "unidad": unidad,
            "timestamp": time.time()
        })

semantic_cache = SemanticCacheManager()
