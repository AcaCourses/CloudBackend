"""
app/generation.py
Módulo de generación enfocado en la API de Groq con cadena de fallback automática.

Modelos ordenados por prioridad:
1. openai/gpt-oss-120b
2. openai/gpt-oss-20b
3. qwen/qwen3.6-27b
"""

import json
import os
from collections.abc import Generator

import requests


class ModelUnavailableError(Exception):
    pass


class RateLimitError(Exception):
    pass


# Cargar lista de modelos candidatos de Groq desde variables de entorno
def get_groq_models() -> list[str]:
    raw = os.getenv(
        "GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.6-27b"
    )
    return [m.strip() for m in raw.split(",") if m.strip()]


# ─────────────────────────────────────────────────────────────
# GROQ PROVIDER (HTTP Streaming SSE)
# ─────────────────────────────────────────────────────────────
def call_groq_stream(
    model: str, system_prompt: str, messages: list[dict[str, str]], api_key: str
) -> Generator[str, None, None]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    groq_messages = [{"role": "system", "content": system_prompt}] + messages
    payload = {
        "model": model,
        "messages": groq_messages,
        "temperature": 0.5,
        "max_completion_tokens": 1000,
        "stream": True,
    }

    res = requests.post(url, headers=headers, json=payload, stream=True, timeout=20)

    if res.status_code in (404, 400):
        raise ModelUnavailableError(
            f"Modelo de Groq {model} no disponible ({res.status_code}): {res.text}"
        )
    if res.status_code == 413:
        raise ModelUnavailableError(
            f"Payload demasiado grande (413) para modelo {model}: {res.text}"
        )
    if res.status_code == 429:
        raise RateLimitError(f"Modelo de Groq {model} saturado por Rate Limit (429)")

    res.raise_for_status()

    for line in res.iter_lines():
        if line:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data_json = json.loads(data_str)
                    choices = data_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    pass


# ─────────────────────────────────────────────────────────────
# FUNCIONALIDAD PRINCIPAL CON CADENA DE FALLBACK EN GROQ
# ─────────────────────────────────────────────────────────────
def generate_chat_stream(
    system_prompt: str, messages: list[dict[str, str]]
) -> Generator[str, None, None]:
    """
    Intenta en orden cada modelo de Groq (openai/gpt-oss-120b -> openai/gpt-oss-20b -> qwen/qwen3.6-27b).
    Si uno falla (413, 429, 400), salta inmediatamente al siguiente modelo.
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY no está configurada en las variables de entorno."
        )

    models = get_groq_models()

    for model in models:
        try:
            print(
                f"🤖 [LLM Groq] Intentando generación con modelo: {model}...",
                flush=True,
            )
            for text_chunk in call_groq_stream(
                model, system_prompt, messages, groq_key
            ):
                yield text_chunk
            return
        except (ModelUnavailableError, RateLimitError) as err:
            print(
                f"⚠ [LLM Groq Fallback] Modelo {model} falló: {err}. Probando siguiente candidato...",
                flush=True,
            )
        except Exception as err:
            print(
                f"⚠ [LLM Groq Fallback] Error inesperado en modelo {model}: {err}. Probando siguiente...",
                flush=True,
            )

    raise RuntimeError(
        "Ningún modelo de Groq estuvo disponible en este momento. Revisa GROQ_API_KEY o las cuotas."
    )
