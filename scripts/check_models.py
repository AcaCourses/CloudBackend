"""
scripts/check_models.py

Script independiente para validar semanalmente que los modelos configurados
en Gemini y Groq siguen existiendo y vigentes en sus APIs.

Uso:
  python scripts/check_models.py
"""

import os
import requests

def check_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠ GEMINI_API_KEY no configurada. Omitiendo validación de Gemini.")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    res = requests.get(url, timeout=15)
    if res.status_code != 200:
        print(f"❌ Error al consultar la API de Gemini: {res.status_code} - {res.text}")
        return

    available = {m["name"].replace("models/", "") for m in res.json().get("models", [])}
    raw_models = os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash")
    configured = [m.strip() for m in raw_models.split(",") if m.strip()]

    print("🔎 Verificando modelos de Gemini...")
    for model in configured:
        if model in available:
            print(f"   ✔ {model}: ACTIVO Y DISPONIBLE")
        else:
            print(f"   ❌ {model}: NO ENCONTRADO O DEPRECADO en Gemini!")

def check_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠ GROQ_API_KEY no configurada. Omitiendo validación de Groq.")
        return

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    res = requests.get(url, headers=headers, timeout=15)
    if res.status_code != 200:
        print(f"❌ Error al consultar la API de Groq: {res.status_code} - {res.text}")
        return

    available = {m["id"] for m in res.json().get("data", [])}
    raw_models = os.getenv("GROQ_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b")
    configured = [m.strip() for m in raw_models.split(",") if m.strip()]

    print("🔎 Verificando modelos de Groq...")
    for model in configured:
        if model in available:
            print(f"   ✔ {model}: ACTIVO Y DISPONIBLE")
        else:
            print(f"   ❌ {model}: NO ENCONTRADO O DEPRECADO en Groq!")

def main():
    print("==================================================")
    print("   CHEQUEO DE DISPONIBILIDAD DE MODELOS LLM")
    print("==================================================")
    check_gemini()
    print()
    check_groq()
    print("==================================================")

if __name__ == "__main__":
    main()
