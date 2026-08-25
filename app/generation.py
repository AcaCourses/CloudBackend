"""
app/generation.py
Módulo de generación multimodelo con cadena de fallback automática:
Gemini (Primero) -> Groq (Fallback)

Evita hardcodear un solo modelo. Lee listas ordenadas por variables de entorno:
  GEMINI_MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash
  GROQ_MODELS=openai/gpt-oss-20b,openai/gpt-oss-120b
"""

import os
import json
import requests
from typing import Generator, Tuple, List, Dict, Any

class ModelUnavailableError(Exception):
    pass

class RateLimitError(Exception):
    pass

# Cargar listas de modelos candidatos desde variables de entorno
def get_gemini_models() -> List[str]:
    raw = os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash")
    return [m.strip() for m in raw.split(",") if m.strip()]

def get_groq_models() -> List[str]:
    raw = os.getenv("GROQ_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b")
    return [m.strip() for m in raw.split(",") if m.strip()]

# ─────────────────────────────────────────────────────────────
# 1. GEMINI PROVIDER (HTTP Streaming & Standard)
# ─────────────────────────────────────────────────────────────
def call_gemini_stream(model: str, system_prompt: str, messages: List[Dict[str, str]], api_key: str) -> Generator[str, None, None]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}"
    
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 1000}
    }

    res = requests.post(url, json=payload, stream=True, timeout=20)
    if res.status_code in (404, 400):
        raise ModelUnavailableError(f"Gemini model {model} no disponible: {res.text}")
    if res.status_code == 429:
        raise RateLimitError(f"Gemini model {model} saturado (429)")
    res.raise_for_status()

    for line in res.iter_lines():
        if line:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                try:
                    data_json = json.loads(data_str)
                    candidates = data_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    pass

# ─────────────────────────────────────────────────────────────
# 2. GROQ PROVIDER (HTTP Streaming & Standard)
# ─────────────────────────────────────────────────────────────
def call_groq_stream(model: str, system_prompt: str, messages: List[Dict[str, str]], api_key: str) -> Generator[str, None, None]:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    groq_messages = [{"role": "system", "content": system_prompt}] + messages
    payload = {
        "model": model,
        "messages": groq_messages,
        "temperature": 0.5,
        "max_tokens": 1000,
        "stream": True
    }

    res = requests.post(url, headers=headers, json=payload, stream=True, timeout=20)
    if res.status_code in (404, 400):
        raise ModelUnavailableError(f"Groq model {model} no disponible: {res.text}")
    if res.status_code == 429:
        raise RateLimitError(f"Groq model {model} saturado (429)")
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
# 3. FUNCIONALIDAD PRINCIPAL CON CADENA DE FALLBACK
# ─────────────────────────────────────────────────────────────
def generate_chat_stream(system_prompt: str, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    """
    Intenta en orden cada modelo de Gemini. Si falla por 429 o 404, conmuta al siguiente
    o salta a la lista de modelos de Groq.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # 1. Intentar candidatos de Gemini
    if gemini_key:
        for model in get_gemini_models():
            try:
                print(f"🤖 [LLM] Intentando con Gemini ({model})...")
                for text_chunk in call_gemini_stream(model, system_prompt, messages, gemini_key):
                    yield text_chunk
                return
            except (ModelUnavailableError, RateLimitError) as err:
                print(f"⚠ [LLM Fallback] Gemini {model} falló: {err}. Intentando siguiente...")
            except Exception as err:
                print(f"⚠ [LLM Fallback] Error en Gemini {model}: {err}")

    # 2. Fallback a candidatos de Groq
    if groq_key:
        for model in get_groq_models():
            try:
                print(f"🤖 [LLM Fallback] Intentando con Groq ({model})...")
                for text_chunk in call_groq_stream(model, system_prompt, messages, groq_key):
                    yield text_chunk
                return
            except (ModelUnavailableError, RateLimitError) as err:
                print(f"⚠ [LLM Fallback] Groq {model} falló: {err}. Intentando siguiente...")
            except Exception as err:
                print(f"⚠ [LLM Fallback] Error en Groq {model}: {err}")

    raise RuntimeError("Ningún proveedor de generación (Gemini ni Groq) está disponible en este momento.")
