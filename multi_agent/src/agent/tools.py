


"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from src.agent.configuration import Configuration


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

async def add(a:int,b:int):
    '''
    Add two numbers
    '''
    return a+b

async def sub(a:int,b:int):
    '''
    Subtract two numbers
    '''
    return a-b

async def mul(a:int,b:int):
    '''
    Multiply two numbers
    '''
    return a*b

async def div(a:int,b:int):
    '''
    Divide two numbers
    '''
    return a/b


TOOLS: List[Callable[..., Any]] = [search,add,sub,mul]
