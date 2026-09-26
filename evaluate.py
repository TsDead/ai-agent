"""Мини-эвал агента на golden-наборе (eval/tasks.json).

Две проверки на задачу:
  • tool-accuracy — вызвал ли агент нужный инструмент хотя бы раз;
  • answer-accuracy — содержит ли финальный ответ ожидаемую подстроку.
Плюс продакшн-метрики: среднее число шагов, токенов и latency на задачу.
Так проверяют агентов в проде — не «качество модели», а надёжность и цена.

Запуск: python evaluate.py   (или кнопка «Запустить» в UI).
"""

import json
import time
from pathlib import Path

import agent

TASKS = json.loads((Path(__file__).parent / "eval" / "tasks.json").read_text(encoding="utf-8"))


def run():
    rows = []
    steps_sum = tok_sum = lat_sum = 0
    tool_ok = ans_ok = 0

    for t in TASKS:
        t0 = time.time()
        r = agent.run(t["q"])
        latency = time.time() - t0

        used_tools = {s["action"] for s in r["steps"]}
        tool_hit = t["tool"] in used_tools
        answer = (r.get("answer") or "").lower()
        ans_hit = any(e.lower() in answer for e in t["expect"])

        tool_ok += tool_hit
        ans_ok += ans_hit
        steps_sum += len(r["steps"])
        toks = r["usage"].get("prompt_tokens", 0) + r["usage"].get("completion_tokens", 0)
        tok_sum += toks
        lat_sum += latency

        rows.append({"q": t["q"], "tool": t["tool"], "tool_hit": tool_hit,
                     "answer_hit": ans_hit, "steps": len(r["steps"]),
                     "latency": round(latency, 2)})

    n = len(TASKS)
    return {
        "n": n,
        "tool_accuracy": round(tool_ok / n, 3),
        "answer_accuracy": round(ans_ok / n, 3),
        "avg_steps": round(steps_sum / n, 2),
        "avg_tokens": round(tok_sum / n),
        "avg_latency": round(lat_sum / n, 2),
        "rows": rows,
    }


if __name__ == "__main__":
    res = run()
    print(f"Задач: {res['n']}")
    print(f"Tool-accuracy:   {res['tool_accuracy']*100:.0f}%")
    print(f"Answer-accuracy: {res['answer_accuracy']*100:.0f}%")
    print(f"Avg steps: {res['avg_steps']} · avg tokens: {res['avg_tokens']} · avg latency: {res['avg_latency']}s")
    for r in res["rows"]:
        mark = "✓" if r["tool_hit"] and r["answer_hit"] else "✗"
        print(f"  {mark} [{r['tool']}] {r['q'][:50]}  ({r['steps']} шагов, {r['latency']}s)")
