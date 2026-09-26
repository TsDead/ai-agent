"""Инструменты агента. Все — бесплатные, без ключей и внешних SDK.

Каждый инструмент = функция (query: str) -> str. Реестр TOOLS собирает их
в словарь + описание, которое подставляется в системный промпт агента.
"""

import ast
import operator
import datetime
import requests

WIKI_URL = "https://ru.wikipedia.org/w/api.php"
UA = {"User-Agent": "novacode-ai-agent/1.0 (portfolio demo)"}


# ── calculator ────────────────────────────────────────────────────────────
# Безопасный вычислитель на AST — без eval(), только арифметика.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("только числа")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("недопустимое выражение")


def calculator(query: str) -> str:
    """Считает арифметическое выражение, напр. '(1234*7 + 89) / 3'."""
    expr = query.strip().replace("^", "**").replace(",", ".").replace("×", "*").replace("÷", "/")
    try:
        result = _eval_node(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError:
        return "Ошибка: деление на ноль."
    except Exception:
        return f"Не смог вычислить '{query}'. Оставь только числа и + - * / ** % ( )."
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expr} = {result}"


# ── current_time ──────────────────────────────────────────────────────────
def current_time(query: str = "") -> str:
    """Текущая дата и время сервера (UTC). Аргумент не нужен."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("Сейчас %Y-%m-%d %H:%M UTC (%A)")


# ── wikipedia_search ──────────────────────────────────────────────────────
def wikipedia_search(query: str) -> str:
    """Ищет статью в Википедии и возвращает вводный абзац (extract)."""
    q = query.strip()
    if not q:
        return "Пустой запрос."
    try:
        # 1. поиск заголовка
        s = requests.get(WIKI_URL, headers=UA, timeout=20, params={
            "action": "query", "list": "search", "srsearch": q,
            "srlimit": 1, "format": "json"}).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return f"В Википедии ничего не найдено по запросу «{q}»."
        title = hits[0]["title"]
        # 2. вводный абзац статьи
        e = requests.get(WIKI_URL, headers=UA, timeout=20, params={
            "action": "query", "prop": "extracts", "exintro": 1,
            "explaintext": 1, "titles": title, "format": "json"}).json()
        pages = e.get("query", {}).get("pages", {})
        extract = next(iter(pages.values())).get("extract", "").strip()
        if not extract:
            return f"Статья «{title}» найдена, но без вводного текста."
        extract = " ".join(extract.split())
        if len(extract) > 900:
            extract = extract[:900].rsplit(" ", 1)[0] + "…"
        return f"[Википедия · {title}] {extract}"
    except Exception as ex:
        return f"Ошибка обращения к Википедии: {ex}"


# ── реестр ──────────────────────────────────────────────────────────────
TOOLS = {
    "wikipedia_search": {
        "fn": wikipedia_search,
        "desc": "Найти факты о теме, человеке, месте или событии в Википедии. Вход: поисковый запрос.",
    },
    "calculator": {
        "fn": calculator,
        "desc": "Точно посчитать арифметическое выражение. Вход: выражение, напр. '2^10 / 8'.",
    },
    "current_time": {
        "fn": current_time,
        "desc": "Узнать текущие дату и время (UTC). Вход: не нужен (пусто).",
    },
}


def run_tool(name: str, arg: str) -> str:
    tool = TOOLS.get(name)
    if not tool:
        return f"Нет такого инструмента: {name}. Доступны: {', '.join(TOOLS)}."
    return tool["fn"](arg or "")


def tools_description() -> str:
    return "\n".join(f"- {n}: {t['desc']}" for n, t in TOOLS.items())
