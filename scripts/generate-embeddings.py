"""
scripts/generate-embeddings.py

Genera embeddings vectoriales 100% locales en CPU con FastEmbed (ONNX)
usando el modelo multilingüe: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
a partir de data/knowledge-chunks.json y los guarda en data/knowledge-embeddings.json.

Uso:
  python scripts/generate-embeddings.py
"""

import os
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "knowledge-chunks.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "knowledge-embeddings.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    if not os.path.exists(CHUNKS_FILE):
        print(f"❌ Error: No se encontró {CHUNKS_FILE}. Corre primero el script de extracción en el frontend.")
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"🔄 Leídos {len(chunks)} chunks de conocimiento.")
    print(f"🤖 Cargando modelo local FastEmbed ({EMBEDDING_MODEL_NAME})...")

    try:
        from fastembed import TextEmbedding
        model = TextEmbedding(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"❌ Error al cargar FastEmbed: {e}")
        print("Asegúrate de haber instalado 'fastembed' con: pip install fastembed")
        return

    texts = [f"{c['title']}\n{c['text']}" for c in chunks]
    
    print(f"⚡ Generando embeddings locales en CPU...")
    start_time = time.time()
    
    embeddings_generator = model.embed(texts)
    embedded_chunks = []

    for i, vector in enumerate(embeddings_generator):
        chunk = chunks[i]
        embedded_chunks.append({
            "id": chunk["id"],
            "type": chunk.get("type", "lesson"),
            "unidad": chunk.get("unidad"),
            "labNumber": chunk.get("labNumber"),
            "slug": chunk["slug"],
            "title": chunk["title"],
            "text": chunk["text"],
            "url": chunk["url"],
            "embedding": list(vector)
        })

    elapsed = time.time() - start_time
    print(f"✔ Generación completada en {elapsed:.2f} segundos.")

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    print(f"✔ ¡Éxito! Generados {len(embedded_chunks)} embeddings multilingües -> {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()
