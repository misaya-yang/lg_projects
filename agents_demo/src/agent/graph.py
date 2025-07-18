import json
from typing import TypedDict, List, Dict, Annotated, Optional

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command, interrupt
from langgraph.prebuilt import create_react_agent

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.tools.tavily_search import TavilySearchResults
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


client = MultiServerMCPClient(
    {
        "context7": {
        "command": "bunx",
        "args": ["-y", "@upstash/context7-mcp"],
        "transport": "stdio",
        }
    }
)


async def main():
    global graph
    tools = await client.get_tools()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=configuration.system_prompt,
    )
    graph = (
        StateGraph(MessagesState)
        .add_node("agent", agent)
        .add_edge(START, "agent")
        .add_edge("agent", END)
    ).compile()
    # 你可以在这里继续用 graph 做后续操作

asyncio.run(main())









