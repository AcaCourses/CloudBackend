# CloudBackend (FastAPI + RAG + Render)

Backend de microservicios para la plataforma del curso de Cloud Computing.

## 🚀 Estructura del Proyecto

```
CloudBackend/
├── app/
│   ├── main.py                # Aplicación FastAPI con endpoints (/search, /chat, /exam)
│   └── rag.py                 # Motor de búsqueda vectorial (Cosine Similarity en RAM)
├── data/
│   ├── knowledge-chunks.json  # Chunks de lecciones y laboratorios extraídos del frontend
│   ├── style-examples.json    # Ejemplos de Quizzes, Escenarios y Labs para Few-Shot
│   └── knowledge-embeddings.json # Chunks vectorizados (se suben al repo para Render)
├── scripts/
│   └── generate-embeddings.py # Genera los vectores de Hugging Face de forma manual
├── .env.example               # Plantilla de variables de entorno
├── requirements.txt           # Dependencias para Render
└── README.md
```

## 🛠 Pasos de Generación de Embeddings (Paso Manual)

Cada vez que actualices o extraigas nuevo contenido desde el repositorio del frontend (`CloudComputing`), ejecuta en este repositorio:

1. Crea o configura la variable en un archivo `.env`:
   ```bash
   HF_API_TOKEN=tu_token_de_huggingface
   ```

2. Ejecuta el script de vectorización:
   ```bash
   python scripts/generate-embeddings.py
   ```

3. Commitea los 3 archivos JSON generados en `data/` al repositorio Git.

## 📡 Endpoints de la API

- `POST /search`: Búsqueda semántica sobre las 54 unidades/labs.
- `POST /chat`: Asistente tutor con contexto RAG y streaming.
- `POST /exam`: Generación estructurada de evaluaciones y pocos disparadores Few-Shot.

## 🌐 Deploy en Render

- **Build Command**: 
  ```bash
  pip install -r requirements.txt && python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
  ```
- **Start Command**: 
  ```bash
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```
- **Environment Variables**:
  - `GEMINI_API_KEY`
  - `GROQ_API_KEY`
  - `GEMINI_MODELS`
  - `GROQ_MODELS`
  - `ALLOWED_ORIGINS` (`https://cloud-computing-beta-plum.vercel.app,http://localhost:3000`)
