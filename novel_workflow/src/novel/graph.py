from __future__ import annotations

from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class State(TypedDict):
    """Shared state for the novel workflow."""

    user_input: dict
    outline: Optional[List[str]]
    schema: Optional[List[dict]]
    chapter_index: int
    chapter_schema: Optional[dict]
    chapter_text: Optional[str]
    human_feedback: Optional[str]


llm = ChatOpenAI()


# 1. Outline agent generates a list of chapter titles

def generate_outline(state: State) -> dict:
    user = state["user_input"]
    prompt = (
        "请根据以下信息给出小说的章节列表，格式为JSON数组，只包含章节名。\n"
        f"标题：{user['title']}\n"
        f"构思：{user['idea']}\n"
        f"章节数：{user['chapter_cnt']}\n"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    import json, re

    text = str(resp.content).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    chapters = json.loads(text)
    return {"outline": chapters, "chapter_index": 0}


# 2. Schema agent prepares summary for a single chapter

def chapter_schema_agent(state: State) -> dict:
    idx = state["chapter_index"]
    title = state["outline"][idx]
    prompt = (
        f"为章节'{title}'编写200字以内的大纲，返回JSON: {{'name': str, 'summary': str}}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    import json, re

    text = str(resp.content).strip()
    match = re.search(r"{.*}", text, re.DOTALL)
    if match:
        text = match.group(0)
    schema = json.loads(text)
    return {"chapter_schema": schema}


# 3. Chapter writer turns schema into story text

def chapter_writer(state: State) -> dict:
    schema = state["chapter_schema"]
    feedback = state.get("human_feedback") or ""
    prompt = (
        f"根据以下大纲撰写小说章节：\n标题：{schema['name']}\n大纲：{schema['summary']}\n"
    )
    if feedback:
        prompt += f"人类建议：{feedback}\n"
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {"chapter_text": str(resp.content).strip(), "human_feedback": None}


# 4. Optional human in the loop after each chapter

def human_review(state: State) -> dict:
    text = state.get("chapter_text", "")
    if not text:
        return {}
    data = interrupt({"preview": text[:100], "prompt": "输入修改意见，或accept继续:"})
    feedback = ""
    if isinstance(data, dict) and data:
        feedback = list(data.values())[0]
    else:
        feedback = str(data)
    if feedback.lower().strip() == "accept":
        return {"chapter_index": state["chapter_index"] + 1}
    else:
        return {"human_feedback": feedback}


# Control logic

def should_continue(state: State):
    if state.get("human_feedback"):
        return "chapter_writer"
    if state["chapter_index"] < len(state["outline"]):
        return "chapter_schema_agent"
    return END


builder = StateGraph(State)
builder.add_node("generate_outline", generate_outline)
builder.add_node("chapter_schema_agent", chapter_schema_agent)
builder.add_node("chapter_writer", chapter_writer)
builder.add_node("human_review", human_review)

builder.add_edge(START, "generate_outline")
builder.add_edge("generate_outline", "chapter_schema_agent")
builder.add_edge("chapter_schema_agent", "chapter_writer")
builder.add_edge("chapter_writer", "human_review")

builder.add_conditional_edges(
    "human_review", should_continue, {"chapter_writer": "chapter_writer", "chapter_schema_agent": "chapter_schema_agent", END: END}
)

novel_graph = builder.compile()
