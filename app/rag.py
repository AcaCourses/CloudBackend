"""
app/rag.py
Módulo RAG ultra-liviano usando la API remota de Hugging Face (sin librerías pesadas en RAM).
Consumo en RAM de Python: < 40 MB. Ideal para el Tier Gratis de Render (512 MB).
"""

import os
import json
import time
import requests
import numpy as np

# Nuevo endpoint router oficial de Hugging Face
HF_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}/pipeline/feature-extraction"

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

def get_query_embedding(query: str, retries: int = 3) -> list[float]:
    """
    Genera el embedding de la consulta usando la API remota de Hugging Face (Endpoint Router).
    Incluye backoff para reintentar si el modelo está despertando (503).
    """
    hf_token = os.getenv("HF_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {"inputs": query, "options": {"wait_for_model": True}}

    for attempt in range(retries):
        try:
            res = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                # Si viene una matriz por token, aplicar mean pooling
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
                        tokens = data[0]
                        dim = len(tokens[0])
                        return [sum(t[i] for t in tokens) / len(tokens) for i in range(dim)]
                    return data[0] if isinstance(data[0], list) else data
                return data
            elif res.status_code == 503:
                print(f"[RAG HF] Modelo en HF despertando (503)... esperando {5 * (attempt + 1)}s")
                time.sleep(5 * (attempt + 1))
            else:
                raise Exception(f"HF Router Error ({res.status_code}): {res.text}")
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise Exception(f"Error al conectar con la API de Hugging Face: {e}")
            time.sleep(3)

    raise Exception("No se pudo obtener el embedding tras varios reintentos en Hugging Face.")

def create_snippet(text: str, query: str, max_chars: int = 160) -> str:
    """Genera un snippet recortado alrededor de la primera coincidencia o inicio del texto."""
    if not text:
        return ""
    text_clean = " ".join(text.split())
    if len(text_clean) <= max_chars:
        return text_clean
    return text_clean[:max_chars] + "..."
