"""
app/db.py
Módulo de persistencia y estadísticas en Supabase.
Maneja el guardado y actualización asíncrona de preguntas, respuestas y calificaciones (rating 👍/👎) en el chat.
"""

import os
import json
import time
import concurrent.futures
from typing import Dict, Any, Optional, List
import requests

# Executor global de hilos para procesos en segundo plano (non-blocking)
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=5)

def _get_supabase_config():
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")
    return url, key

def is_supabase_configured() -> bool:
    url, key = _get_supabase_config()
    return bool(url and key)

def _save_chat_log_sync(
    user_query: str,
    assistant_response: Optional[str],
    unidad: Optional[int],
    sources: Optional[List[Dict[str, Any]]],
    response_time_ms: int,
    status: str,
    error_message: Optional[str],
    client_ip: Optional[str],
    log_id: Optional[str] = None
):
    """
    Guarda el registro en Supabase mediante la API REST de PostgREST / Supabase SDK.
    Esta función corre en un hilo secundario para no bloquear la respuesta al usuario.
    """
    url, key = _get_supabase_config()
    if not url or not key:
        print("ℹ [Supabase] Inserción omitida: SUPABASE_URL o SUPABASE_KEY no configuradas en env vars.", flush=True)
        return

    endpoint = f"{url}/rest/v1/chat_logs"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    payload = {
        "user_query": user_query,
        "assistant_response": assistant_response,
        "unidad": unidad,
        "sources": sources or [],
        "response_time_ms": response_time_ms,
        "status": status,
        "error_message": error_message,
        "client_ip": client_ip
    }

    if log_id:
        payload["id"] = log_id

    print(f"🚀 [Supabase BG Task] Enviando registro a {endpoint}...", flush=True)

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        if res.status_code in (200, 201):
            print(f"✔ [Supabase BG Task] Chat log guardado exitosamente en DB ({response_time_ms}ms).", flush=True)
        else:
            print(f"❌ [Supabase BG Task] Error al guardar chat log! HTTP {res.status_code}: {res.text}", flush=True)
    except Exception as e:
        print(f"❌ [Supabase BG Task] Excepción en segundo plano al intentar guardar en Supabase: {type(e).__name__}: {e}", flush=True)

def _update_chat_rating_sync(log_id: str, rating: int, comment: Optional[str] = None):
    """
    Actualiza la calificación (rating 👍/👎) y comentario de un mensaje existente en Supabase.
    """
    url, key = _get_supabase_config()
    if not url or not key:
        print("ℹ [Supabase] Actualización omitida: SUPABASE_URL o SUPABASE_KEY no configuradas.", flush=True)
        return

    endpoint = f"{url}/rest/v1/chat_logs?id=eq.{log_id}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    payload = {"rating": rating}
    if comment:
        payload["feedback_comment"] = comment

    print(f"👍 [Supabase Rating BG Task] Actualizando rating ({rating}) para log {log_id}...", flush=True)

    try:
        res = requests.patch(endpoint, headers=headers, json=payload, timeout=10)
        if res.status_code in (200, 204):
            print(f"✔ [Supabase Rating BG Task] Rating actualizado con éxito para ID {log_id}.", flush=True)
        else:
            print(f"❌ [Supabase Rating BG Task] Error al actualizar rating! HTTP {res.status_code}: {res.text}", flush=True)
    except Exception as e:
        print(f"❌ [Supabase Rating BG Task] Excepción al actualizar rating: {type(e).__name__}: {e}", flush=True)

def _on_future_done(future):
    try:
        future.result()
    except Exception as exc:
        print(f"❌ [Supabase BG Task Thread Error] {exc}", flush=True)

def save_chat_log_async(
    user_query: str,
    assistant_response: Optional[str] = None,
    unidad: Optional[int] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    response_time_ms: int = 0,
    status: str = "success",
    error_message: Optional[str] = None,
    client_ip: Optional[str] = None,
    log_id: Optional[str] = None
):
    """
    Lanza el guardado en Supabase de forma totalmente asíncrona / en segundo plano.
    """
    future = _EXECUTOR.submit(
        _save_chat_log_sync,
        user_query,
        assistant_response,
        unidad,
        sources,
        response_time_ms,
        status,
        error_message,
        client_ip,
        log_id
    )
    future.add_done_callback(_on_future_done)

def update_chat_rating_async(log_id: str, rating: int, comment: Optional[str] = None):
    """
    Lanza la actualización de calificación (👍/👎) en segundo plano.
    """
    future = _EXECUTOR.submit(_update_chat_rating_sync, log_id, rating, comment)
    future.add_done_callback(_on_future_done)

def fetch_chat_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas básicas desde la tabla chat_logs en Supabase.
    """
    url, key = _get_supabase_config()
    if not url or not key:
        return {"configured": False, "message": "Supabase no está configurado."}

    endpoint = f"{url}/rest/v1/chat_logs?select=id,status,unidad,response_time_ms,rating,created_at&order=created_at.desc&limit=500"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}"
    }

    try:
        res = requests.get(endpoint, headers=headers, timeout=10)
        if res.status_code == 200:
            logs = res.json()
            total_chats = len(logs)
            success_count = sum(1 for log in logs if log.get("status") in ("success", "success_cached"))
            cached_count = sum(1 for log in logs if log.get("status") == "success_cached")
            error_count = sum(1 for log in logs if log.get("status") == "error")
            blocked_count = sum(1 for log in logs if log.get("status") == "blocked_guardrails")
            
            positive_ratings = sum(1 for log in logs if log.get("rating") == 1)
            negative_ratings = sum(1 for log in logs if log.get("rating") == -1)

            times = [log["response_time_ms"] for log in logs if log.get("response_time_ms")]
            avg_time = sum(times) / len(times) if times else 0

            return {
                "configured": True,
                "total_queries_sample": total_chats,
                "success_count": success_count,
                "cached_hits_count": cached_count,
                "error_count": error_count,
                "blocked_guardrails_count": blocked_count,
                "positive_ratings": positive_ratings,
                "negative_ratings": negative_ratings,
                "avg_response_time_ms": round(avg_time, 2),
                "recent_logs_count": len(logs)
            }
        else:
            return {"configured": True, "error": f"HTTP {res.status_code}: {res.text}"}
    except Exception as e:
        return {"configured": True, "error": str(e)}
