"""
scripts/generate-embeddings.py

Genera embeddings vectoriales de los chunks de conocimiento usando FastEmbed localmente en CPU.
Guarda los 54 vectores en data/knowledge-embeddings.json con sus URLs absolutas.
"""

import json
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "knowledge-chunks.json")
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "knowledge-embeddings.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    if not os.path.exists(CHUNKS_FILE):
        print(
            f"Error: No se encontro {CHUNKS_FILE}. Corre primero el script de extraccion en el frontend."
        )
        return

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[RAG] Leidos {len(chunks)} chunks de conocimiento.")
    print(f"[RAG] Cargando modelo local FastEmbed ({MODEL_NAME})...")

    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(MODEL_NAME)
    except Exception as e:
        print(f"Error al cargar FastEmbed: {e}")
        return

    texts = [f"{c['title']}\n{c['text']}" for c in chunks]
    print("[RAG] Generando embeddings vectoriales...")
    start_time = time.time()

    embeddings_generator = model.embed(texts)
    embedded_chunks = []

    for i, vector in enumerate(embeddings_generator):
        chunk = chunks[i]
        embedded_chunks.append(
            {
                "id": chunk["id"],
                "type": chunk.get("type", "lesson"),
                "unidad": chunk.get("unidad"),
                "labNumber": chunk.get("labNumber"),
                "slug": chunk["slug"],
                "title": chunk["title"],
                "text": chunk["text"],
                "url": chunk["url"],
                "embedding": list(vector),
            }
        )

    elapsed = time.time() - start_time
    print(f"[RAG] Generacion completada en {elapsed:.2f} segundos.")

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    print(
        f"[RAG] Exito: Generados {len(embedded_chunks)} embeddings -> {EMBEDDINGS_FILE}"
    )


if __name__ == "__main__":
    main()
