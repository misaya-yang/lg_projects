"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4o",  # 改为你的模型名
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    base_url: str = field(
        default="https://api.openai-proxy.org/v1",
        metadata={
            "description": "The base URL for the OpenAI proxy API."
        },
    )
    api_key: str = field(
        default="sk-Pi81m0dUmZvpFOXJwa0erWEjybri0Yqq6Ay8U4H7xBSPhB8O",  # 你自己的 key
        metadata={
            "description": "The API key for the OpenAI proxy API."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    search_prompt: str = field(
        default=prompts.SEARCH_PROMPT,
        metadata={
            "description": "The prompt to use for the research agent's interactions. "
            "This prompt sets the context and behavior for the research agent."
        },
    )

    math_prompt: str = field(
        default=prompts.MATH_PROMPT,
        metadata={
            "description": "The prompt to use for the math agent's interactions. "
            "This prompt sets the context and behavior for the math agent."
        },
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
