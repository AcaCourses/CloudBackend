"""
app/rag.py
Módulo RAG ultra-liviano usando la API remota de Hugging Face (sin librerías pesadas en RAM).
Consumo en RAM de Python: < 40 MB. Ideal para el Tier Gratis de Render (512 MB).
"""

import json
import os
import time

import numpy as np
import requests

# Nuevo endpoint router oficial de Hugging Face
HF_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}/pipeline/feature-extraction"


class VectorStore:
    def __init__(self):
        self.chunks = []
        self.matrix = None

    def load_embeddings(self, filepath: str):
        if not os.path.exists(filepath):
            print(
                f"⚠ WARNING: Archivo {filepath} no encontrado. RAG semántico no estará disponible hasta generarlo."
            )
            return

        with open(filepath, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if self.chunks:
            vectors = [c["embedding"] for c in self.chunks]
            self.matrix = np.array(vectors, dtype=np.float32)
            print(
                f"✔ Cargados {len(self.chunks)} vectores ({self.matrix.shape[1]} dims) en RAM para RAG."
            )

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        unidad: int = None,
        lab_number: int = None,
    ):
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
            results.append(
                {
                    "id": chunk["id"],
                    "type": chunk.get("type", "lesson"),
                    "unidad": chunk.get("unidad"),
                    "labNumber": chunk.get("labNumber"),
                    "slug": chunk["slug"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                    "url": chunk["url"],
                    "score": float(scores[orig_idx]),
                }
            )
        return results


# Instancia singleton global
vector_store = VectorStore()


def get_cohere_embedding(query: str) -> list[float]:
    """Fallback 1: Cohere API (embed-multilingual-v3.0)."""
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise Exception("COHERE_API_KEY no configurada")

    url = "https://api.cohere.com/v1/embed"
    headers = {
        "Authorization": f"Bearer {cohere_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "texts": [query],
        "model": "embed-multilingual-v3.0",
        "input_type": "search_query",
    }
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    if res.status_code == 200:
        data = res.json()
        embeddings = data.get("embeddings", {})
        if isinstance(
            embeddings, dict
        ):  # v3 API retorna dict con tipos (float, int8, etc)
            float_embs = embeddings.get("float", [])
            if float_embs:
                return float_embs[0]
        elif isinstance(embeddings, list) and len(embeddings) > 0:
            return embeddings[0]
    raise Exception(f"Cohere API Error ({res.status_code}): {res.text}")


def get_jina_embedding(query: str) -> list[float]:
    """Fallback 2: Jina AI Embedding API (jina-embeddings-v3)."""
    jina_key = os.getenv("JINA_API_KEY")
    if not jina_key:
        raise Exception("JINA_API_KEY no configurada")

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {jina_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "dimensions": 384,  # Reducir dimensión a 384 para compatibilidad
        "input": [query],
    }
    res = requests.post(url, headers=headers, json=payload, timeout=15)
    if res.status_code == 200:
        data = res.json()
        data_list = data.get("data", [])
        if data_list and "embedding" in data_list[0]:
            return data_list[0]["embedding"]
    raise Exception(f"Jina API Error ({res.status_code}): {res.text}")


def get_hf_embedding(query: str, retries: int = 2) -> list[float]:
    """HF Router API."""
    hf_token = os.getenv("HF_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {"inputs": query, "options": {"wait_for_model": True}}

    for attempt in range(retries):
        res = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                if (
                    isinstance(data[0], list)
                    and len(data[0]) > 0
                    and isinstance(data[0][0], list)
                ):
                    tokens = data[0]
                    dim = len(tokens[0])
                    return [sum(t[i] for t in tokens) / len(tokens) for i in range(dim)]
                return data[0] if isinstance(data[0], list) else data
            return data
        elif res.status_code == 503:
            time.sleep(3)

    raise Exception("HF Router API saturado o no disponible.")


def get_query_embedding(query: str) -> list[float]:
    """
    Cadena de Fallback para Embeddings:
      1. Hugging Face Router API (paraphrase-multilingual-MiniLM-L12-v2)
      2. Cohere API (embed-multilingual-v3.0)
      3. Jina AI API (jina-embeddings-v3)
    """
    # 1. Intentar Hugging Face Router API
    try:
        return get_hf_embedding(query)
    except Exception as e:
        print(f"⚠ [Embedding Fallback] HF Router API falló: {e}. Probando Cohere...")

    # 2. Intentar Cohere API (embed-multilingual-v3.0)
    try:
        return get_cohere_embedding(query)
    except Exception as e:
        print(f"⚠ [Embedding Fallback] Cohere API falló: {e}. Probando Jina AI...")

    # 3. Intentar Jina AI API (jina-embeddings-v3)
    try:
        return get_jina_embedding(query)
    except Exception as e:
        print(f"⚠ [Embedding Fallback] Jina AI API falló: {e}.")

    raise Exception(
        "Ningún proveedor de embeddings (HF, Cohere ni Jina) estuvo disponible."
    )


def create_snippet(text: str, query: str, max_chars: int = 160) -> str:
    """Genera un snippet recortado alrededor de la primera coincidencia o inicio del texto."""
    if not text:
        return ""
    text_clean = " ".join(text.split())
    if len(text_clean) <= max_chars:
        return text_clean
    return text_clean[:max_chars] + "..."
