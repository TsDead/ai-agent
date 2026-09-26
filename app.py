"""AI-агент с инструментами — FastAPI. Задай вопрос → агент сам решает,
какие инструменты вызвать (Википедия / калькулятор / время), и показывает
каждый шаг рассуждения (ReAct).
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import agent
import evaluate
import llm
from tools import TOOLS

app = FastAPI(title="AI Agent · ReAct")
HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

EXAMPLES = [
    "Сколько лет назад родился Алан Тьюринг, если сейчас 2026 год?",
    "В каком городе находится штаб-квартира Nintendo и сколько будет 1980 * 12?",
    "Кто написал «Преступление и наказание» и в каком году он родился?",
    "Посчитай 2^16 и объясни, что это за число в информатике.",
]


class AskReq(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/meta")
def meta():
    return {
        "ready": llm.available(),
        "tools": [{"name": n, "desc": t["desc"]} for n, t in TOOLS.items()],
        "examples": EXAMPLES,
        "max_steps": agent.MAX_STEPS,
    }


@app.post("/api/ask")
def ask(r: AskReq):
    if not r.question.strip():
        return {"error": "Пустой вопрос."}
    if not llm.available():
        return {"error": "Не задан GROQ_API_KEY в .env — см. .env.example."}
    try:
        return agent.run(r.question)
    except Exception as e:
        return {"error": f"Сбой агента: {e}"}


@app.post("/api/eval")
def run_eval():
    if not llm.available():
        return {"error": "Не задан GROQ_API_KEY в .env."}
    try:
        return evaluate.run()
    except Exception as e:
        return {"error": f"Сбой эвала: {e}"}
