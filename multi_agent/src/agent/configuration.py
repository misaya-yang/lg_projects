"""Define the configurable parameters for the deep research agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated
import os

from dotenv import load_dotenv
from langchain_core.runnables import ensure_config
from langgraph.config import get_config
from pydantic import SecretStr

from src.agent import prompts

# 加载multi_agent目录下的.env文件
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


@dataclass(kw_only=True)
class Configuration:
    """Configuration for the deep research work graph."""

    research_system_prompt: str = field(
        default=prompts.RESEARCH_SYSTEM_PROMPT,
        metadata={"description": "High-level system instructions shared across the research workflow."},
    )

    planning_prompt: str = field(
        default=prompts.RESEARCH_PLANNING_PROMPT,
        metadata={"description": "Prompt template used to request a structured research plan."},
    )

    analysis_prompt: str = field(
        default=prompts.RESEARCH_ANALYSIS_PROMPT,
        metadata={"description": "Prompt template guiding evidence synthesis for each step."},
    )

    synthesis_prompt: str = field(
        default=prompts.RESEARCH_SYNTHESIS_PROMPT,
        metadata={"description": "Prompt template for generating the final markdown report."},
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o-mini",
        metadata={"description": "The chat model identifier used for reasoning and writing."},
    )

    base_url: str = field(
        default="https://api.openai-proxy.org/v1",
        metadata={"description": "Base URL for the model provider."},
    )

    api_key: SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("api_key", "")),
        metadata={"description": "API key for the configured model provider."},
    )

    max_search_results: int = field(
        default=8,
        metadata={"description": "Maximum number of Tavily search results to request per query."},
    )

    max_research_branches: int = field(
        default=4,
        metadata={"description": "Maximum number of plan steps the planner should explore."},
    )

    default_docx_directory: str = field(
        default="artifacts/reports",
        metadata={"description": "Default directory used when exporting markdown research to DOCX."},
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""

        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
