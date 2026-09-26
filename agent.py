"""ReAct-агент на чистом Python — без LangChain и прочих фреймворков.

Цикл: LLM рассуждает (thought) → выбирает инструмент (action + input) →
получает результат (observation) → повторяет, пока не даст финальный ответ.
Формат шага — строгий JSON, это надёжнее парсинга свободного текста.
"""

import json
import re

import llm
from tools import TOOLS, run_tool, tools_description

MAX_STEPS = 6  # предохранитель от зацикливания

SYSTEM = """Ты — рассуждающий агент. Ты решаешь задачу пошагово, используя инструменты.

Доступные инструменты:
{tools}

На КАЖДОМ шаге отвечай СТРОГО одним JSON-объектом, без markdown и текста вокруг:
  {{"thought": "<рассуждение на 1 фразу>", "action": "<имя инструмента>", "action_input": "<аргумент>"}}
Когда ответ готов — верни:
  {{"thought": "<итоговое рассуждение>", "action": "final", "action_input": "<финальный ответ пользователю>"}}

Правила:
- Не выдумывай факты — для фактов вызывай wikipedia_search.
- Для любых расчётов вызывай calculator, не считай в уме.
- Используй результаты инструментов (observation) из истории, не повторяй один и тот же вызов.
- Отвечай на языке вопроса. Только валидный JSON, ничего кроме него."""


def _parse_step(raw: str) -> dict:
    """Достаёт JSON-объект из ответа модели, даже если он обёрнут в текст/```."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"action": "final", "action_input": raw or "(пустой ответ модели)"}


def run(question: str):
    """Гоняет ReAct-цикл. Возвращает {answer, steps[], usage, stopped}."""
    system = SYSTEM.format(tools=tools_description())
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question.strip()}]
    steps = []
    total = {"prompt_tokens": 0, "completion_tokens": 0}

    for i in range(MAX_STEPS):
        raw, usage = llm.chat(messages, max_tokens=700)
        total["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total["completion_tokens"] += usage.get("completion_tokens", 0)

        step = _parse_step(raw)
        action = (step.get("action") or "final").strip()
        thought = (step.get("thought") or "").strip()
        arg = step.get("action_input")
        arg = "" if arg is None else str(arg)

        if action == "final":
            steps.append({"n": i + 1, "thought": thought, "action": "final",
                          "input": arg, "observation": None})
            return {"answer": arg, "steps": steps, "usage": total, "stopped": False}

        # вызов инструмента
        observation = run_tool(action, arg) if action in TOOLS \
            else f"Нет инструмента «{action}». Доступны: {', '.join(TOOLS)}."
        steps.append({"n": i + 1, "thought": thought, "action": action,
                      "input": arg, "observation": observation})

        # возвращаем модели её же шаг + наблюдение
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",
                         "content": f"Observation: {observation}\nПродолжай (JSON)."})

    # исчерпали лимит шагов — просим финальный ответ по собранному контексту
    messages.append({"role": "user",
                     "content": "Лимит шагов исчерпан. Дай финальный ответ JSON с action=final."})
    raw, usage = llm.chat(messages, max_tokens=500)
    total["prompt_tokens"] += usage.get("prompt_tokens", 0)
    total["completion_tokens"] += usage.get("completion_tokens", 0)
    final = _parse_step(raw)
    answer = str(final.get("action_input") or "Не удалось прийти к ответу за отведённые шаги.")
    return {"answer": answer, "steps": steps, "usage": total, "stopped": True}
