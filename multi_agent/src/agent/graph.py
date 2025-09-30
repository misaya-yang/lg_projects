"""Deep research work graph orchestrated with LangGraph."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from deepagent import DeepResearchAgent, Finding, PlanItem
from src.agent.configuration import Configuration
from src.agent.tools import markdown_to_docx, search
from src.agent.utils import load_chat_model


class ResearchState(TypedDict, total=False):
    """State container used throughout the deep research workflow."""

    query: str
    title: str
    plan: list[PlanItem]
    current_step: int
    findings: list[Finding]
    report_markdown: str
    docx_path: str
    exported_at: str


def _build_agent(configuration: Configuration) -> DeepResearchAgent:
    api_key = (
        configuration.api_key.get_secret_value()
        if hasattr(configuration.api_key, 'get_secret_value')
        else configuration.api_key
    )
    llm = load_chat_model(
        configuration.model,
        configuration.base_url,
        api_key,
    )
    return DeepResearchAgent(
        llm=llm,
        system_prompt=configuration.research_system_prompt,
        planning_prompt=configuration.planning_prompt,
        analysis_prompt=configuration.analysis_prompt,
        synthesis_prompt=configuration.synthesis_prompt,
    )


async def plan_research(state: ResearchState) -> ResearchState:
    configuration = Configuration.from_context()
    agent = _build_agent(configuration)
    if "query" not in state:
        raise ValueError("Research query must be provided in the initial state.")
    if state.get("plan"):
        plan = [PlanItem(**step) if not isinstance(step, PlanItem) else step for step in state["plan"]]
    else:
        plan = await agent.plan_research(
            query=state["query"],
            max_branches=configuration.max_research_branches,
        )
    enriched_state = dict(state)
    enriched_state.setdefault("current_step", 0)
    enriched_state["plan"] = plan
    return enriched_state


async def execute_research_step(state: ResearchState) -> ResearchState:
    configuration = Configuration.from_context()
    agent = _build_agent(configuration)
    plan = state.get("plan", [])
    index = state.get("current_step", 0)
    if index >= len(plan):
        return state
    plan_item = plan[index]
    if not isinstance(plan_item, PlanItem):
        plan_item = PlanItem(**plan_item)
    finding = await agent.gather_evidence(
        query=state["query"],
        step=plan_item,
        search_tool=search,
        max_results=configuration.max_search_results,
    )
    findings = list(state.get("findings", []))
    findings.append(finding)
    updated_state = dict(state)
    updated_state["findings"] = findings
    updated_state["current_step"] = index + 1
    return updated_state


async def synthesise_report(state: ResearchState) -> ResearchState:
    configuration = Configuration.from_context()
    agent = _build_agent(configuration)
    plan = [PlanItem(**step) if not isinstance(step, PlanItem) else step for step in state.get("plan", [])]
    findings = [Finding(**finding) if not isinstance(finding, Finding) else finding for finding in state.get("findings", [])]
    report = await agent.synthesise(
        query=state["query"],
        plan=plan,
        findings=findings,
    )
    enriched_state = dict(state)
    enriched_state["report_markdown"] = report
    return enriched_state


async def export_docx(state: ResearchState) -> ResearchState:
    if not state.get("report_markdown"):
        return state
    title = state.get("title") or f"Deep Research Report - {state['query']}"
    docx_path = await markdown_to_docx(state["report_markdown"], title=title)
    enriched_state = dict(state)
    enriched_state["docx_path"] = docx_path
    enriched_state["exported_at"] = datetime.now(tz=UTC).isoformat()
    return enriched_state


def route_after_plan(state: ResearchState) -> Literal["research", "__end__"]:
    if state.get("plan"):
        return "research"
    return "__end__"


def route_research(state: ResearchState) -> Literal["research", "synthesise"]:
    plan = state.get("plan", [])
    index = state.get("current_step", 0)
    if index < len(plan):
        return "research"
    return "synthesise"


def route_after_synthesis(state: ResearchState) -> Literal["export", "__end__"]:
    if state.get("report_markdown"):
        return "export"
    return "__end__"


builder = StateGraph(ResearchState, config_schema=Configuration)
builder.add_node("plan", plan_research)
builder.add_node("research", execute_research_step)
builder.add_node("synthesise", synthesise_report)
builder.add_node("export", export_docx)

builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", route_after_plan, {"research": "research", "__end__": END})
builder.add_conditional_edges("research", route_research, {"research": "research", "synthesise": "synthesise"})
builder.add_conditional_edges("synthesise", route_after_synthesis, {"export": "export", "__end__": END})
builder.add_edge("export", END)

graph = builder.compile(name="deep_research_work_graph")
