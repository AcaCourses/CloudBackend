"""
app/db.py
Módulo de persistencia y estadísticas en Supabase.
Maneja el guardado asíncrono en segundo plano de preguntas y respuestas del chat.
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
    client_ip: Optional[str]
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

    print(f"🚀 [Supabase BG Task] Enviando registro a {endpoint}...", flush=True)

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        if res.status_code in (200, 201):
            print(f"✔ [Supabase BG Task] Chat log guardado exitosamente en DB ({response_time_ms}ms).", flush=True)
        else:
            print(f"❌ [Supabase BG Task] Error al guardar chat log! HTTP {res.status_code}: {res.text}", flush=True)
    except Exception as e:
        print(f"❌ [Supabase BG Task] Excepción en segundo plano al intentar guardar en Supabase: {type(e).__name__}: {e}", flush=True)

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
    client_ip: Optional[str] = None
):
    """
    Lanza el guardado en Supabase de forma totalmente asíncrona / en segundo plano.
    No bloquea la ejecución ni el streaming del usuario.
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
        client_ip
    )
    future.add_done_callback(_on_future_done)

def fetch_chat_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas básicas desde la tabla chat_logs en Supabase.
    """
    url, key = _get_supabase_config()
    if not url or not key:
        return {"configured": False, "message": "Supabase no está configurado."}

    endpoint = f"{url}/rest/v1/chat_logs?select=id,status,unidad,response_time_ms,created_at&order=created_at.desc&limit=500"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}"
    }

    try:
        res = requests.get(endpoint, headers=headers, timeout=10)
        if res.status_code == 200:
            logs = res.json()
            total_chats = len(logs)
            success_count = sum(1 for log in logs if log.get("status") == "success")
            error_count = sum(1 for log in logs if log.get("status") == "error")
            blocked_count = sum(1 for log in logs if log.get("status") == "blocked_guardrails")
            
            times = [log["response_time_ms"] for log in logs if log.get("response_time_ms")]
            avg_time = sum(times) / len(times) if times else 0

            return {
                "configured": True,
                "total_queries_sample": total_chats,
                "success_count": success_count,
                "error_count": error_count,
                "blocked_guardrails_count": blocked_count,
                "avg_response_time_ms": round(avg_time, 2),
                "recent_logs_count": len(logs)
            }
        else:
            return {"configured": True, "error": f"HTTP {res.status_code}: {res.text}"}
    except Exception as e:
        return {"configured": True, "error": str(e)}
