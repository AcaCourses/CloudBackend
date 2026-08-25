"""
app/main.py
Servidor FastAPI para RAG, Chat Asistente e Exámenes del Curso Cloud Computing
"""

import os
import json
import time
import hashlib
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests

from app.rag import vector_store, get_query_embedding, create_snippet
from app.generation import generate_chat_stream

app = FastAPI(
    title="Cloud Computing Course Backend API",
    description="Backend RAG FastAPI para búsqueda semántica, asistente IA y generación de exámenes.",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE CORS MULTI-ORIGEN
# ─────────────────────────────────────────────────────────────
raw_origins = os.getenv("ALLOWED_ORIGINS", "https://cloud-computing-beta-plum.vercel.app,http://localhost:3000,http://localhost:5173")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# 2. RATE LIMITING BÁSICO POR IP (20 requests / minuto por IP)
# ─────────────────────────────────────────────────────────────
RATE_LIMIT_STORE = {} # {ip: [timestamps]}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Excluir health check de rate limit
    if request.url.path in ["/", "/health"]:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = 60 # 60 segundos
    max_requests = 30 # máximo 30 peticiones por minuto por IP

    timestamps = [t for t in RATE_LIMIT_STORE.get(client_ip, []) if now - t < window]
    if len(timestamps) >= max_requests:
        raise HTTPException(
            status_code=429,
            detail="Demasiadas peticiones. Por favor espera un minuto antes de reintentar."
        )

    timestamps.append(now)
    RATE_LIMIT_STORE[client_ip] = timestamps
    response = await call_next(request)
    return response

# ─────────────────────────────────────────────────────────────
# CACHÉ EN RAM PARA BUSQUEDAS
# ─────────────────────────────────────────────────────────────
SEARCH_CACHE = {}

@app.on_event("startup")
def startup_event():
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not gemini_key and not groq_key:
        print("⚠ WARNING: Ni GEMINI_API_KEY ni GROQ_API_KEY están configuradas. El endpoint /chat no podrá generar respuestas.")
    else:
        print("✔ Llaves de API para generación configuradas (Gemini / Groq).")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_file = os.path.join(BASE_DIR, "data", "knowledge-embeddings.json")
    vector_store.load_embeddings(embeddings_file)

# ─────────────────────────────────────────────────────────────
# SCHEMAS (PYDANTIC)
# ─────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    unidad: Optional[int] = None
    labNumber: Optional[int] = None

class SearchResultItem(BaseModel):
    id: str
    title: str
    unidad: Optional[int] = None
    labNumber: Optional[int] = None
    slug: str
    snippet: str
    url: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class ChatMessage(BaseModel):
    role: str # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []
    unidad: Optional[int] = None

class ExamRequest(BaseModel):
    unidad: Optional[int] = None
    labNumber: Optional[int] = None
    tipo: Optional[str] = "all" # "quiz" | "scenario" | "matching" | "classify" | "all"
    cantidad: Optional[int] = 5

# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
def health_check():
    """Endpoint de salud para verificaciones de Render y Uptime Kuma."""
    return {
        "status": "ok",
        "service": "Cloud Computing RAG & Exam API",
        "vector_chunks": len(vector_store.chunks),
        "timestamp": time.time()
    }

@app.post("/search", response_model=SearchResponse)
def search_knowledge(req: SearchRequest):
    """Endpoint /search: Recibe consulta, genera embedding HF y devuelve Top-K con snippet."""
    if not req.query.strip():
        return SearchResponse(results=[])

    cache_key = hashlib.md5(f"{req.query.strip().lower()}_{req.k}_{req.unidad}_{req.labNumber}".encode()).hexdigest()
    if cache_key in SEARCH_CACHE:
        return SEARCH_CACHE[cache_key]

    try:
        q_vec = get_query_embedding(req.query)
        raw_results = vector_store.search(q_vec, k=req.k, unidad=req.unidad, lab_number=req.labNumber)
        
        formatted_results = []
        for item in raw_results:
            formatted_results.append(SearchResultItem(
                id=item["id"],
                title=item["title"],
                unidad=item.get("unidad"),
                labNumber=item.get("labNumber"),
                slug=item["slug"],
                snippet=create_snippet(item["text"], req.query),
                url=item["url"],
                score=item["score"]
            ))

        response = SearchResponse(results=formatted_results)
        SEARCH_CACHE[cache_key] = response
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_assistant(req: ChatRequest):
    """Endpoint /chat: RAG + Fallback Dinámico (Gemini -> Groq) con streaming SSE real."""
    # 1. Recuperar contexto relevante mediante RAG
    q_vec = get_query_embedding(req.query)
    top_chunks = vector_store.search(q_vec, k=3, unidad=req.unidad)
    
    context_str = "\n\n".join([f"--- CHUNK [{c['title']}] ---\n{c['text']}" for c in top_chunks])
    sources = [{"title": c["title"], "url": c["url"], "slug": c["slug"]} for c in top_chunks]

    # 2. Prompt del sistema
    system_prompt = (
        "Eres el Asistente Tutor IA oficial del curso de Cloud Computing y Google Cloud Platform (GCP).\n"
        "Tu objetivo es dar respuestas visualmente impecables, estructuradas y didácticas para los alumnos.\n\n"
        "REGLAS DE FORMATO OBLIGATORIAS:\n"
        "1. Estructura la respuesta usando Markdown elegante (Títulos ##, negritas, listas con viñetas y tablas comparativas cuando aplique).\n"
        "2. Mantén un tono profesional, claro y directo.\n"
        "3. Basado en el contexto del curso proporcionado a continuación, incluye explicaciones detalladas con ventajas, desventajas o ejemplos prácticos si la pregunta lo amerita.\n"
        "4. Al final de tu respuesta, bajo la sección '### 📚 Fuentes del curso', incluye siempre las referencias con enlace explícito en Markdown [Título](URL).\n\n"
        f"CONTEXTO DEL CURSO:\n{context_str}\n"
    )

    messages = [{"role": msg.role, "content": msg.content} for msg in (req.history or [])]
    messages.append({"role": "user", "content": req.query})

    def generate_stream():
        try:
            for text_chunk in generate_chat_stream(system_prompt, messages):
                yield f"data: {json.dumps({'text': text_chunk})}\n\n"

            yield f"data: {json.dumps({'sources': sources})}\n\n"
        except Exception as err:
            yield f"data: {json.dumps({'error': str(err)})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

@app.post("/exam")
def generate_exam(req: ExamRequest):
    """
    Endpoint /exam: Obtiene ejemplos de estilo (style-examples.json) estructurados
    matching los componentes del frontend (Quiz, Scenario, Matching, Classify).
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    style_file = os.path.join(BASE_DIR, "data", "style-examples.json")

    examples = []
    if os.path.exists(style_file):
        try:
            with open(style_file, "r", encoding="utf-8") as f:
                all_examples = json.load(f)
                
                examples = [
                    ex for ex in all_examples
                    if (req.unidad is None or ex.get("unidad") == req.unidad)
                    and (req.labNumber is None or ex.get("labNumber") == req.labNumber)
                    and (req.tipo == "all" or ex.get("tipo") == req.tipo)
                ][:req.cantidad]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parseando style-examples.json: {str(e)}")

    return {
        "unidad": req.unidad,
        "labNumber": req.labNumber,
        "tipo": req.tipo,
        "total_examples": len(examples),
        "questions": [ex["data"] for ex in examples]
    }
