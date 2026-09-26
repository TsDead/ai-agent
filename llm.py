"""Провайдеро-независимый LLM-клиент. По умолчанию — бесплатный Groq."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# qwen следует текстовому ReAct-протоколу; gpt-oss на Groq принудительно
# уходит в нативный tool-calling и ломает наш «ReAct с нуля».
MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b").strip()


def available() -> bool:
    return bool(GROQ_KEY)


def chat(messages, max_tokens=900, temperature=0.0):
    """Возвращает (content, usage). Провайдер спрятан за этой функцией —
    сменить Groq на любой OpenAI-совместимый endpoint = поменять URL/ключ."""
    if not GROQ_KEY:
        raise RuntimeError("нет GROQ_API_KEY в .env")
    payload = {"model": MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if "gpt-oss" in MODEL:
        payload["reasoning_effort"] = "low"
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"].get("content", "") or "").strip(), data.get("usage", {})
