"""Lightweight DeepAgent-style helpers for deep research workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.utils import get_message_text


class PlanItem(dict):
    """Typed dictionary representing an individual research step."""

    title: str
    objective: str
    questions: list[str]

    def __init__(self, *, title: str, objective: str, questions: Iterable[str] | None = None) -> None:
        super().__init__(
            title=title.strip(),
            objective=objective.strip(),
            questions=[q.strip() for q in (questions or []) if q.strip()],
        )


class Finding(dict):
    """Typed dictionary representing synthesized evidence for a single step."""

    step: str
    summary: str
    sources: list[dict[str, str]]

    def __init__(self, *, step: str, summary: str, sources: Iterable[dict[str, str]] | None = None) -> None:
        super().__init__(
            step=step.strip(),
            summary=summary.strip(),
            sources=[source for source in (sources or []) if source.get("url")],
        )


@dataclass(slots=True)
class DeepResearchAgent:
    """A minimal DeepAgent-style helper built on top of an LLM."""

    llm: BaseChatModel
    system_prompt: str
    planning_prompt: str
    analysis_prompt: str
    synthesis_prompt: str

    async def plan_research(self, *, query: str, max_branches: int) -> list[PlanItem]:
        """Create a structured multi-step research plan."""

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=self.planning_prompt.format(
                        query=query.strip(),
                        max_branches=max_branches,
                    )
                ),
            ]
        )
        return self._parse_plan(get_message_text(response))

    async def gather_evidence(
        self,
        *,
        query: str,
        step: PlanItem,
        search_tool: Callable[[str], Awaitable[dict[str, Any] | None]],
        max_results: int,
    ) -> Finding:
        """Execute a single research step using a search tool and summarize findings."""

        focus_query = self._compose_focus_query(query=query, step=step)
        search_payload = await search_tool(focus_query)
        documents = self._normalise_results(search_payload, max_results=max_results)
        digest = self._format_documents(documents)

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=self.analysis_prompt.format(
                        query=query.strip(),
                        step_title=step["title"],
                        step_objective=step.get("objective", ""),
                        guiding_questions="\n".join(step.get("questions", [])),
                        search_digest=digest,
                    )
                ),
            ]
        )
        summary = get_message_text(response)
        return Finding(step=step["title"], summary=summary, sources=documents)

    async def synthesise(self, *, query: str, plan: Sequence[PlanItem], findings: Sequence[Finding]) -> str:
        """Produce a final markdown research report."""

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=self.synthesis_prompt.format(
                        query=query.strip(),
                        plan=json.dumps(list(plan), ensure_ascii=False, indent=2),
                        findings=json.dumps(list(findings), ensure_ascii=False, indent=2),
                    )
                ),
            ]
        )
        return get_message_text(response).strip()

    def _parse_plan(self, raw: str) -> list[PlanItem]:
        raw = raw.strip()
        json_blob = self._extract_json(raw)
        if json_blob is not None:
            return [
                PlanItem(
                    title=item.get("title", item.get("step", "")) or f"Step {index + 1}",
                    objective=item.get("objective", item.get("goal", "")) or "",
                    questions=item.get("questions") or item.get("subtasks") or [],
                )
                for index, item in enumerate(json_blob)
            ]
        return self._parse_markdown_plan(raw)

    def _parse_markdown_plan(self, raw: str) -> list[PlanItem]:
        items: list[PlanItem] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].isdigit():
                _, _, rest = stripped.partition(".")
                items.append(
                    PlanItem(
                        title=rest.strip() or stripped,
                        objective="",
                    )
                )
            elif stripped.startswith("-") and items:
                items[-1]["questions"].append(stripped.removeprefix("-").strip())
        if not items:
            items.append(PlanItem(title=raw, objective=""))
        return items

    def _extract_json(self, raw: str) -> list[dict[str, Any]] | None:
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
        except ValueError:
            return None
        snippet = raw[start:end]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            return None
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return None

    def _compose_focus_query(self, *, query: str, step: PlanItem) -> str:
        questions = step.get("questions") or []
        question_suffix = "; ".join(questions)
        pieces = [query, step.get("objective", ""), question_suffix]
        return " - ".join(part for part in pieces if part)

    def _normalise_results(self, payload: dict[str, Any] | None, *, max_results: int) -> list[dict[str, str]]:
        if not payload:
            return []
        results = payload.get("results")
        if not isinstance(results, Sequence):
            return []
        documents: list[dict[str, str]] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("url") or "Source")
            url = str(item.get("url") or "")
            snippet = str(item.get("content") or item.get("snippet") or "")
            documents.append({"title": title, "url": url, "content": snippet})
        return documents

    def _format_documents(self, documents: Sequence[dict[str, str]]) -> str:
        if not documents:
            return "No supporting documents were found for this step."
        formatted = []
        for index, doc in enumerate(documents, start=1):
            formatted.append(
                f"[{index}] {doc.get('title', 'Source')}\nURL: {doc.get('url', 'N/A')}\nSnippet: {doc.get('content', '').strip()}"
            )
        return "\n\n".join(formatted)


__all__ = ["DeepResearchAgent", "PlanItem", "Finding"]
