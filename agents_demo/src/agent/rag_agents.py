import json
from typing import TypedDict, List, Dict, Annotated, Optional

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import E2BDataAnalysisTool
from langchain_openai import ChatOpenAI

from src.agent.tools import search
from src.agent.configuration import Configuration

import asyncio

configuration = Configuration.from_context()

llm = ChatOpenAI(
    model=configuration.model,
    base_url=configuration.base_url,
    api_key=configuration.api_key,
)



