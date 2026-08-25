"""
app/rag.py
Módulo de recuperación de información semántica (RAG) basado en Cosine Similarity en RAM.
Calcula los embeddings localmente en CPU con ONNX/FastEmbed (rápido, ~100MB RAM)
sin depender de llamadas HTTP externas ni cuotas para la inferencia en tiempo real.
"""

import os
import json
import time
import requests
import numpy as np

# Modelo multilingüe optimizado de 384 dimensiones para FastEmbed (ONNX)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_EMBEDDING_MODEL = None

def get_local_model():
    """Lazy load del modelo en la primera consulta para no consumir RAM en el startup de uvicorn."""
    global LOCAL_EMBEDDING_MODEL
    if LOCAL_EMBEDDING_MODEL is None:
        try:
            from fastembed import TextEmbedding
            LOCAL_EMBEDDING_MODEL = TextEmbedding(EMBEDDING_MODEL_NAME)
            print(f"✔ Modelo local ONNX ({EMBEDDING_MODEL_NAME}) inicializado en RAM.")
        except Exception as e:
            print(f"⚠ No se pudo cargar FastEmbed ({e}). Se usará HF API como fallback.")
    return LOCAL_EMBEDDING_MODEL

HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL_NAME}"

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.matrix = None

    def load_embeddings(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"⚠ WARNING: Archivo {filepath} no encontrado. RAG semántico no estará disponible hasta generarlo.")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if self.chunks:
            vectors = [c["embedding"] for c in self.chunks]
            self.matrix = np.array(vectors, dtype=np.float32)
            print(f"✔ Cargados {len(self.chunks)} vectores ({self.matrix.shape[1]} dims) en RAM para RAG.")

    def search(self, query_vector: list[float], k: int = 5, unidad: int = None, lab_number: int = None):
        if self.matrix is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        norm_q = np.linalg.norm(q_vec)
        norm_m = np.linalg.norm(self.matrix, axis=1)
        
        denom = norm_m * norm_q
        denom[denom == 0] = 1e-10

        scores = (self.matrix @ q_vec) / denom

        valid_indices = []
        for idx, chunk in enumerate(self.chunks):
            if unidad is not None and chunk.get("unidad") != unidad:
                continue
            if lab_number is not None and chunk.get("labNumber") != lab_number:
                continue
            valid_indices.append(idx)

        if not valid_indices:
            valid_indices = list(range(len(self.chunks)))

        sub_scores = scores[valid_indices]
        top_k_sub_idx = np.argsort(sub_scores)[-k:][::-1]
        
        results = []
        for sub_i in top_k_sub_idx:
            orig_idx = valid_indices[sub_i]
            chunk = self.chunks[orig_idx]
            results.append({
                "id": chunk["id"],
                "type": chunk.get("type", "lesson"),
                "unidad": chunk.get("unidad"),
                "labNumber": chunk.get("labNumber"),
                "slug": chunk["slug"],
                "title": chunk["title"],
                "text": chunk["text"],
                "url": chunk["url"],
                "score": float(scores[orig_idx])
            })
        return results

# Instancia singleton global
vector_store = VectorStore()

def get_query_embedding(query: str, hf_token: str = None) -> list[float]:
    """
    Genera el embedding de la consulta.
    Intenta primero hacerlo 100% local en CPU (ONNX int8). Si falla, hace fallback a la API de HF.
    """
    # 1. Intentar modelo local con Lazy Loading
    local_model = get_local_model()
    if local_model is not None:
        try:
            embeddings_generator = local_model.embed([query])
            vector = list(next(embeddings_generator))
            return vector
        except Exception as err:
            print(f"⚠ Fallo al calcular embedding local ({err}). Reintentando con HF API...")

    # 2. Fallback a HF API
    if not hf_token:
        hf_token = os.getenv("HF_API_TOKEN", "")

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": [query], "options": {"wait_for_model": True}}
    
    for attempt in range(3):
        try:
            res = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
            if res.status_code == 200:
                data = res.json()
                vector = data[0]
                if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                    vector = vector[0]
                return vector
            elif res.status_code == 503:
                time.sleep(3)
        except Exception:
            time.sleep(2)

    raise Exception("No se pudo obtener el embedding de la consulta (Local ni HF API).")

def create_snippet(text: str, query: str, max_chars: int = 160) -> str:
    """Genera un snippet recortado alrededor de la primera coincidencia o inicio del texto."""
    if not text:
        return ""
    text_clean = " ".join(text.split())
    if len(text_clean) <= max_chars:
        return text_clean
    return text_clean[:max_chars] + "..."

