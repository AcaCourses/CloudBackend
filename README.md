# 🚀 CloudBackend (FastAPI + RAG + Groq LLM + Supabase)

Backend de microservicios para la plataforma interactiva del curso de **Cloud Computing y Google Cloud Platform (GCP)**. Proporciona búsqueda semántica RAG en memoria RAM, asistente tutor con IA multimodelo (Groq), persistencia en segundo plano en Supabase, caché semántico y analíticas docentes en tiempo real.

---

## 🏗️ Arquitectura del Sistema

```
CloudBackend/
├── app/
│   ├── main.py                # Servidor FastAPI principal y endpoints (/chat, /search, /exam, /chat/stats, /chat/feedback)
│   ├── db.py                  # Capa de persistencia asíncrona no bloqueante en Supabase (PostgREST API)
│   ├── cache.py               # Caché Semántico en memoria (<50ms, 0 tokens)
│   ├── generation.py          # Generador multimodelo Groq con cadena de fallback automática
│   ├── guardrails.py          # Seguridad contra Prompt Injections, Spam y desbordamiento de contexto
│   └── rag.py                 # Motor de búsqueda vectorial (Cosine Similarity en RAM con embeddings de HF/Cohere/Jina)
├── data/
│   ├── knowledge-chunks.json  # Lecciones y laboratorios estructurados del curso
│   ├── style-examples.json    # Ejemplos Few-Shot para evaluaciones del curso
│   └── knowledge-embeddings.json # Embeddings vectorizados de 384 dimensiones
├── scripts/
│   └── supabase_schema.sql    # Script SQL de migración e índices para Supabase
├── .env.example               # Plantilla de variables de entorno
├── requirements.txt           # Dependencias para Render / servidor
└── README.md
```

---

## ⚡ Flujo de Procesamiento en `/chat`

1. **Guardrails de Seguridad (`app/guardrails.py`)**: Filtra Prompt Injections (ej. *"ignore previous instructions"*), spam de letras y controla límites de caracteres.
2. **Generación de Embedding**: Convierte la duda a un vector de 384 dimensiones.
3. **Caché Semántica (`app/cache.py`)**: Si una duda similar ($\ge 92\%$ de similitud) fue respondida previamente, se devuelve instantáneamente (**<50ms**, 0 tokens de Groq).
4. **Búsqueda Vectorial RAG (`app/rag.py`)**: Si es una duda nueva, recupera el Top-3 de fragmentos de lecciones/labs en RAM.
5. **Generación Streaming con Groq (`app/generation.py`)**: Genera la respuesta en Markdown con cadena de fallback automática:
   1. `openai/gpt-oss-120b` (Modelo principal)
   2. `openai/gpt-oss-20b` (Fallback 1)
   3. `qwen/qwen3.6-27b` (Fallback 2)
6. **Persistencia en Segundo Plano (`app/db.py`)**: Al terminar el streaming, registra la consulta, respuesta, latencia, fuentes e IP en Supabase mediante un hilo secundario (`ThreadPoolExecutor`), sin retrasar la pantalla del alumno.

---

## 📡 Endpoints de la API API REST

### 1. `POST /chat`
- **Descripción:** Endpoint principal del Asistente Tutor IA con Streaming SSE (Server-Sent Events).
- **Headers:** `X-Access-Key: <clave_del_curso>`
- **Body JSON:**
  ```json
  {
    "query": "¿Qué es Compute Engine y cuándo usarlo?",
    "unidad": 2,
    "history": [],
    "access_key": "rgm8dh"
  }
  ```
- **Respuesta SSE:** Stream de eventos `data: {"text": "..."}` finalizando con `data: {"sources": [...], "log_id": "<uuid>", "cached": true/false}`.

### 2. `POST /chat/feedback`
- **Descripción:** Registra la calificación dada por el alumno (👍 / 👎) sobre una respuesta recibida.
- **Body JSON:**
  ```json
  {
    "log_id": "0059c20a-34b9-4f56-ae0c-5a065d7e777e",
    "rating": 1,
    "comment": "Explicación clara"
  }
  ```

### 3. `GET /chat/stats`
- **Descripción:** Proporciona el resumen de analíticas completas para el Dashboard Docente.
- **Query Params:** `access_key=rgm8dh`
- **Respuesta:** Total de consultas, tasa de satisfacción ($\% 	ext{ de } 	ext{👍}$), ratio de éxito de Caché Semántica, latencia media, desglose por Unidad Temática y lista de consultas recientes.

### 4. `POST /search`
- **Descripción:** Búsqueda semántica pura RAG sobre las 54 unidades y laboratorios del curso.

### 5. `POST /exam`
- **Descripción:** Generación estructurada de preguntas de evaluación (Quizzes, Escenarios, Matching).

---

## 🔑 Variables de Entorno (`.env`)

| Variable | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Clave API de Groq Console (Obligatoria) | `gsk_...` |
| `GROQ_MODELS` | Cadena de fallback de modelos | `openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.6-27b` |
| `SUPABASE_URL` | URL del proyecto Supabase | `https://xqpypnjnmojbwuujasnz.supabase.co` |
| `SUPABASE_KEY` | Clave `anon` o `service_role` de Supabase | `eyJh...` |
| `CHAT_ACCESS_KEY` | Clave de acceso requerida para el curso | `rgm8dh` |
| `ALLOWED_ORIGINS` | Dominios permitidos por CORS | `https://cloud-computing-beta-plum.vercel.app,http://localhost:3000` |

---

## 🗄️ Configuración de Supabase

Ejecuta el script [`scripts/supabase_schema.sql`](file:///d:/dr871/Projects/CloudBackend/scripts/supabase_schema.sql) en el **SQL Editor** de Supabase para crear la tabla `chat_logs` con índices y políticas RLS.
