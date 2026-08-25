"""
scripts/generate-embeddings.py

Genera embeddings vectoriales consumiendo el endpoint router oficial de Hugging Face:
  https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction

A partir de data/knowledge-chunks.json y los guarda en data/knowledge-embeddings.json.

Uso:
  python scripts/generate-embeddings.py
"""

import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "knowledge-chunks.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "knowledge-embeddings.json")

HF_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_ROUTER_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}/pipeline/feature-extraction"

def get_hf_token():
    token = os.getenv("HF_API_TOKEN")
    if not token:
        env_path = os.path.join(BASE_DIR, ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("HF_API_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
    return token

def query_hf_embeddings(text: str, headers: dict) -> list:
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    res = requests.post(HF_ROUTER_URL, headers=headers, json=payload, timeout=30)
    
    if res.status_code == 200:
        data = res.json()
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
                tokens = data[0]
                dim = len(tokens[0])
                return [sum(t[i] for t in tokens) / len(tokens) for i in range(dim)]
            return data[0] if isinstance(data[0], list) else data
        return data
    elif res.status_code == 503:
        print("   [HF Router] Modelo cargándose... esperando 5 segundos")
        time.sleep(5)
        return query_hf_embeddings(text, headers)
    else:
        raise Exception(f"Error HF Router [{res.status_code}]: {res.text}")

def main():
    hf_token = get_hf_token()
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    if not os.path.exists(CHUNKS_FILE):
        print(f"Error: No se encontro {CHUNKS_FILE}. Corre primero el script de extraccion en el frontend.")
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[RAG] Leidos {len(chunks)} chunks. Generando embeddings mediante HF Router API...")
    embedded_chunks = []
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        text = f"{chunk['title']}\n{chunk['text']}"
        print(f"   Procesando chunk {i + 1}/{len(chunks)}: {chunk['title'][:40]}...")
        
        try:
            vector = query_hf_embeddings(text, headers)
            embedded_chunks.append({
                "id": chunk["id"],
                "type": chunk.get("type", "lesson"),
                "unidad": chunk.get("unidad"),
                "labNumber": chunk.get("labNumber"),
                "slug": chunk["slug"],
                "title": chunk["title"],
                "text": chunk["text"],
                "url": chunk["url"],
                "embedding": vector
            })
        except Exception as e:
            print(f"Error procesando chunk {i}: {e}")
            return

        time.sleep(0.3)

    elapsed = time.time() - start_time
    print(f"[RAG] Generacion completada en {elapsed:.2f} segundos.")

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    print(f"[RAG] Exito: Generados {len(embedded_chunks)} embeddings -> {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()
