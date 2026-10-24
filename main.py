from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
DB_PATH = DATA_DIR / "memory.sqlite3"
OUTPUT_PATH = DATA_DIR / "latest_plan.txt"

CHAT_MODEL = "qwen2.5:3b"
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "studyflow_knowledge"

PRIORITY_ORDER = {"высокий": 0, "средний": 1, "низкий": 2}

VECTORSTORE: Chroma | None = None
MCP_WRITE_TOOL = None


class StudyState(TypedDict, total=False):
    tasks: list[dict[str, Any]]
    daily_hours: float
    errors: list[str]
    knowledge: str
    knowledge_sources: list[str]
    schedule: list[dict[str, Any]]
    unallocated: list[dict[str, Any]]
    previous_plans: list[dict[str, Any]]
    advice: str
    final_plan: str
    mcp_saved: bool


#Память
def init_memory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                task_summary TEXT NOT NULL,
                plan TEXT NOT NULL
            )
            """
        )
        conn.commit()


def load_memory(limit: int = 3) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, task_summary, plan
            FROM plans
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_memory(tasks: list[dict[str, Any]], plan: str) -> None:
    summary = "; ".join(
        f"{task['title']} (до {task['deadline']}, {task['hours']} ч.)"
        for task in tasks
    )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO plans (created_at, task_summary, plan)
            VALUES (?, ?, ?)
            """,
            (datetime.now().isoformat(timespec="seconds"), summary, plan),
        )
        conn.commit()


#RAG
def init_rag() -> None:
    global VECTORSTORE

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        #разбиение на смысловые фрагменты по абзацам.
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunk = ""
        for paragraph in paragraphs:
            if len(chunk) + len(paragraph) > 900 and chunk:
                documents.append(
                    Document(page_content=chunk, metadata={"source": path.name})
                )
                chunk = paragraph
            else:
                chunk = f"{chunk}\n\n{paragraph}".strip()
        if chunk:
            documents.append(
                Document(page_content=chunk, metadata={"source": path.name})
            )

    #проверка наличия
    if not documents:
        raise RuntimeError(
            f"В папке {KNOWLEDGE_DIR} нет непустых Markdown-документов."
        )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if CHROMA_DIR.exists():
        try:
            store = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
            )
            if store._collection.count() > 0:
                VECTORSTORE = store
                return
        except Exception as exc:
            print(f"RAG: существующий индекс не удалось прочитать: {exc}")

    VECTORSTORE = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


#MCP
async def init_mcp() -> None:
    global MCP_WRITE_TOOL

    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "cmd",
                "args": [
                    "/c",
                    "npx",
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(DATA_DIR.resolve()),
                ],
            }
        }
    )

    tools = await client.get_tools()

    candidates = [
        tool for tool in tools
        if tool.name == "write_file" or tool.name.endswith("_write_file")
    ]
    if not candidates:
        raise RuntimeError(
            "MCP-сервер подключился, но инструмент write_file не найден. "
            "Проверить вывод списка инструментов и версию сервера."
        )

    MCP_WRITE_TOOL = candidates[0]


def validation_agent(state: StudyState) -> dict[str, Any]:
    tasks = state.get("tasks", [])
    daily_hours = state.get("daily_hours", 0)
    errors: list[str] = []

    if not tasks:
        errors.append("Нужно указать хотя бы одну учебную задачу.")
    if daily_hours <= 0:
        errors.append("Количество доступных часов в день должно быть больше нуля.")

    today = date.today()
    for index, task in enumerate(tasks, start=1):
        if not str(task.get("title", "")).strip():
            errors.append(f"Задача №{index}: не указано название.")
            continue

        try:
            deadline = date.fromisoformat(str(task["deadline"]))
            if deadline < today:
                errors.append(
                    f"Задача «{task['title']}»: дедлайн уже прошёл ({deadline})."
                )
        except (KeyError, ValueError):
            errors.append(
                f"Задача №{index}: дата должна быть в формате ГГГГ-ММ-ДД."
            )

        try:
            hours = float(task["hours"])
            if hours <= 0:
                errors.append(f"Задача «{task['title']}»: часы должны быть больше нуля.")
        except (KeyError, TypeError, ValueError):
            errors.append(f"Задача «{task['title']}»: некорректное число часов.")

        priority = str(task.get("priority", "")).lower().strip()
        if priority not in PRIORITY_ORDER:
            errors.append(
                f"Задача «{task['title']}»: приоритет должен быть высокий, средний или низкий."
            )

    if errors:
        raise ValueError("\n".join(errors))

    normalized_tasks = []
    for task in tasks:
        normalized_tasks.append(
            {
                "title": str(task["title"]).strip(),
                "deadline": date.fromisoformat(str(task["deadline"])).isoformat(),
                "hours": float(task["hours"]),
                "priority": str(task["priority"]).lower().strip(),
            }
        )

    return {"tasks": normalized_tasks, "errors": []}

#поиск RAG
def rag_agent(state: StudyState) -> dict[str, Any]:
    if VECTORSTORE is None:
        raise RuntimeError("RAG не инициализирован.")

    query = " ".join(task["title"] for task in state["tasks"])
    query += " правила планирования учёбы дедлайны и методики подготовки"

    docs = VECTORSTORE.similarity_search(query, k=4)
    knowledge_parts = []
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "неизвестный источник")
        sources.append(source)
        knowledge_parts.append(f"[Источник: {source}]\n{doc.page_content}")

    return {
        "knowledge": "\n\n".join(knowledge_parts),
        "knowledge_sources": sorted(set(sources)),
    }


def planner_agent(state: StudyState) -> dict[str, Any]:
    today = date.today()
    daily_capacity = float(state["daily_hours"])

    tasks = sorted(
        state["tasks"],
        key=lambda task: (
            date.fromisoformat(task["deadline"]),
            PRIORITY_ORDER[task["priority"]],
        ),
    )

    #учёт часов
    day_capacity: dict[date, float] = {}
    schedule: list[dict[str, Any]] = []
    unallocated: list[dict[str, Any]] = []

    for task in tasks:
        remaining = float(task["hours"])
        deadline = date.fromisoformat(task["deadline"])
        current_day = today

        while current_day <= deadline and remaining > 0:
            free = day_capacity.get(current_day, daily_capacity)
            if free > 0:
                assigned = min(free, remaining)
                schedule.append(
                    {
                        "date": current_day.isoformat(),
                        "title": task["title"],
                        "hours": round(assigned, 2),
                        "priority": task["priority"],
                    }
                )
                day_capacity[current_day] = round(free - assigned, 2)
                remaining = round(remaining - assigned, 2)
            current_day += timedelta(days=1)

        if remaining > 0:
            unallocated.append(
                {
                    "title": task["title"],
                    "hours": round(remaining, 2),
                    "deadline": task["deadline"],
                    "priority": task["priority"],
                }
            )

    schedule.sort(key=lambda row: (row["date"], row["title"]))

    return {"schedule": schedule, "unallocated": unallocated}


def memory_agent(state: StudyState) -> dict[str, Any]:
    previous = load_memory(limit=3)

    return {"previous_plans": previous}


def advisor_agent(state: StudyState) -> dict[str, Any]:
    model = ChatOllama(model=CHAT_MODEL, temperature=0)

    history_text = "\n\n".join(
        f"Дата: {item['created_at']}\nЗадачи: {item['task_summary']}\n"
        f"Фрагмент старого плана: {item['plan'][:500]}"
        for item in state.get("previous_plans", [])
    ) or "Предыдущих планов пока нет."

    
    prompt = f"""
Ты — помощник по планированию учёбы. Отвечай по-русски, конкретно и простым языком.

Твоя задача — дать практические рекомендации по выполнению учебных задач
и организации учебного процесса.

Правила ответа:
- Пиши обычным текстом без Markdown-разметки.
- Не используй заголовки с символом #, жирный текст со звёздочками,
  маркированные и нумерованные списки, декоративные разделители.
- Не переписывай расписание: оно уже рассчитано программой.
- Не изменяй рассчитанные даты и часы.
- Не придумывай факты о пользователе и не утверждай, что задачи уже выполнены.
- Используй найденные фрагменты базы знаний RAG для подготовки советов.
- Не перечисляй названия файлов базы знаний и не цитируй их отдельным списком.
- Учитывай прошлые планы только в том случае, если они действительно
  помогают дать полезный совет.
- Если часть задач не удалось распределить по дням, объясни простым языком,
  как можно справиться с перегрузкой.
- Не добавляй вступление, общие фразы и заключение.
- Не повторяй одну и ту же рекомендацию разными словами.

ЗАДАЧИ:
{json.dumps(state['tasks'], ensure_ascii=False, indent=2)}

РАСПИСАНИЕ, рассчитанное программой:
{json.dumps(state.get('schedule', []), ensure_ascii=False, indent=2)}

НЕРАСПРЕДЕЛЁННАЯ НАГРУЗКА:
{json.dumps(state.get('unallocated', []), ensure_ascii=False, indent=2)}

ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ RAG:
{state.get('knowledge', '')}

ИСТОРИЯ ПРЕДЫДУЩИХ ПЛАНОВ:
{history_text}

Сформируй краткие практические рекомендации:
Как приступить к выполнению учебных задач, как организовать занятия,
какие подходящие методы обучения использовать и что делать при перегрузке.

Выводи только рекомендации по существу обычным текстом.
"""

    response = model.invoke(prompt)
    advice = str(response.content).strip()
    return {"advice": advice}



def render_plan(state: StudyState) -> str:
    lines = [
        "План выполнения учебных задач",
        "",
    ]

    if state.get("schedule"):
        by_day: dict[str, list[dict[str, Any]]] = {}

        for item in state["schedule"]:
            by_day.setdefault(item["date"], []).append(item)

        for day in sorted(by_day):
            total = sum(item["hours"] for item in by_day[day])
            lines.append(f"{day}. Всего часов: {total:g}.")

            for item in by_day[day]:
                lines.append(
                    f"{item['title']} — {item['hours']:g} ч."
                )

            lines.append("")
    else:
        lines.append("Расписание не сформировано.")
        lines.append("")

    if state.get("unallocated"):
        lines.append("Не удалось распределить всю нагрузку.")

        for item in state["unallocated"]:
            lines.append(
                f"{item['title']} — осталось {item['hours']:g} ч.; "
                f"срок: {item['deadline']}."
            )

        lines.append("")

    lines.extend([
        "Рекомендации по организации учёбы",
        "",
        state.get("advice", "Рекомендации не сформированы.").strip(),
        "",
    ])
    return "\n".join(lines)



async def save_agent(state: StudyState) -> dict[str, Any]:
    if MCP_WRITE_TOOL is None:
        raise RuntimeError("MCP-инструмент записи не инициализирован.")

    plan = render_plan(state)

    #запись выполняется через инструмент MCP
    result = await MCP_WRITE_TOOL.ainvoke(
        {
            "path": str(OUTPUT_PATH.resolve()),
            "content": plan,
        }
    )

    #история сохраняется после успешного вызова MCP
    save_memory(state["tasks"], plan)
    print(f"План сохранён в: {OUTPUT_PATH}")

    return {"final_plan": plan, "mcp_saved": True}


def build_graph():
    graph = StateGraph(StudyState)

    graph.add_node("validation_agent", validation_agent)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("memory_agent", memory_agent)
    graph.add_node("advisor_agent", advisor_agent)
    graph.add_node("save_agent", save_agent)

    graph.add_edge(START, "validation_agent")
    graph.add_edge("validation_agent", "rag_agent")
    graph.add_edge("rag_agent", "planner_agent")
    graph.add_edge("planner_agent", "memory_agent")
    graph.add_edge("memory_agent", "advisor_agent")
    graph.add_edge("advisor_agent", "save_agent")
    graph.add_edge("save_agent", END)

    return graph.compile()


#ввод данных
def read_tasks() -> list[dict[str, Any]]:
    print(
        "\nВведите задачи по одной в формате:\n"
        "Название | ГГГГ-ММ-ДД | часы | высокий/средний/низкий\n"
        "По окончание ввода, нажмите Enter в пустой строке."
    )

    tasks = []
    while True:
        line = input("> ").strip()
        if not line:
            break

        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4:
            print("Ошибка формата. Нужно 4 поля, разделённые символом |.")
            continue

        title, deadline, hours_text, priority = parts
        try:
            hours = float(hours_text.replace(",", "."))
        except ValueError:
            print("Количество часов должно быть числом.")
            continue

        tasks.append(
            {
                "title": title,
                "deadline": deadline,
                "hours": hours,
                "priority": priority.lower(),
            }
        )
    return tasks



async def main() -> None:

    print("Учебный планировщик\n")

    init_memory()
    init_rag()
    await init_mcp()

    daily_hours_text = input("Сколько часов в день доступно для учёбы? ").strip()
    try:
        daily_hours = float(daily_hours_text.replace(",", "."))
    except ValueError as exc:
        raise ValueError("Введите количество часов числом.") from exc

    tasks = read_tasks()
    graph = build_graph()

    initial_state: StudyState = {
        "tasks": tasks,
        "daily_hours": daily_hours,
        "errors": [],
    }

    result = await graph.ainvoke(initial_state)

    print("ГОТОВЫЙ ПЛАН")

    print(result["final_plan"])
    print(f"\nФайл: {OUTPUT_PATH}")
    print(f"MCP-сохранение подтверждено: {result.get('mcp_saved', False)}")


if __name__ == "__main__":
    asyncio.run(main())
