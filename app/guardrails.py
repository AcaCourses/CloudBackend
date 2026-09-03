"""
app/guardrails.py
Módulo de Seguridad y Guardrails para el Backend (Cloud Computing RAG API)
Protege contra Prompt Injections, Jailbreaks, Spam de palabras, Spam de caracteres y contenido fuera de contexto.
"""

import re

from fastapi import HTTPException

# 1. Patrones de Prompt Injection / Jailbreak reconocidos
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"olvida\s+(todas\s+las\s+)?instrucciones",
    r"desregla\s+tus\s+instrucciones",
    r"you\s+are\s+now\s+dan",
    r"act\s+as\s+an?\s+unfiltered",
    r"actua\s+como\s+un\s+modelo\s+sin\s+filtros",
    r"system\s+prompt",
    r"muestra\s+tu\s+prompt\s+de\s+sistema",
    r"revela\s+tus\s+instrucciones",
    r"bypass\s+safety",
    r"salta\s+la\s+seguridad",
    r"modo\n\s*desarrollador",
    r"developer\s+mode",
]

# 2. Configuración de límites cuantitativos
MAX_QUERY_CHARS = (
    400  # Máximo de caracteres por pregunta (evita consumo masivo de contexto)
)
MIN_QUERY_CHARS = 3  # Mínimo de caracteres válidos
MAX_WORD_COUNT = 80  # Máximo de palabras por mensaje
MAX_SINGLE_WORD_LENGTH = (
    45  # Máximo de letras en una sola palabra (bloquea asdfghjkl...)
)


def validate_chat_query(query: str) -> str:
    """
    Valida y sanitiza la consulta enviada por el usuario antes de procesarla en la pipeline RAG/LLM.
    Devuelve la consulta limpia o lanza una HTTPException (400 / 422).
    """
    cleaned = query.strip()

    # Guardrail 1: Longitud mínima y máxima
    if len(cleaned) < MIN_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail="La pregunta es demasiado corta. Por favor escribe una consulta más descriptiva sobre Cloud Computing.",
        )

    if len(cleaned) > MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"La pregunta excede el límite de {MAX_QUERY_CHARS} caracteres. Por favor sintetiza tu duda.",
        )

    # Guardrail 2: Conteo de palabras máximo
    words = cleaned.split()
    if len(words) > MAX_WORD_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"La pregunta excede el límite de {MAX_WORD_COUNT} palabras.",
        )

    # Guardrail 3: Bloqueo de palabras monstruo / spam de letras (ej. aaaaa... o asdfghjkl...)
    for word in words:
        if len(word) > MAX_SINGLE_WORD_LENGTH:
            raise HTTPException(
                status_code=400,
                detail="La consulta contiene palabras inválidas o demasiado largas (posible texto aleatorio).",
            )

    # Guardrail 4: Sanitización de Prompt Injection / Jailbreak
    lowered = cleaned.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            print(
                f"🚨 [Guardrails Alert] Intento de Prompt Injection detectado: '{cleaned[:50]}...'"
            )
            raise HTTPException(
                status_code=400,
                detail="Consulta rechazada por razones de seguridad (patrón de prompt injection detectado).",
            )

    # Guardrail 5: Prevenir envío de caracteres nulos o comandos de control
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)

    return cleaned
