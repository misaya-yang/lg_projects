"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Annotated, Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph import graph
from langgraph.graph import END, StateGraph,START,MessagesState
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent,ToolNode,InjectedState
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool, InjectedToolCallId




from src.agent.configuration import Configuration
from src.agent.state import InputState, State
from src.agent.tools import TOOLS
from src.agent.tools import search
from src.agent.tools import add,sub,mul,div
from src.agent.utils import load_chat_model


#读取配置
configuration = Configuration.from_context()

# 定义一个网络搜索代理
research_agent = create_react_agent(
    model=load_chat_model(
        configuration.model,
        configuration.base_url,
        configuration.api_key,
    ),
    tools=[search],
    prompt=configuration.search_prompt,
    name ='research_agent'
)

#定义一个数学计算代理
math_agent = create_react_agent(
    model=load_chat_model(
        configuration.model,
        configuration.base_url,
        configuration.api_key,
    ),
    tools=[add,sub,mul,div],
    prompt=configuration.math_prompt,
    name ='math_agent'
)

# 定义一个supervisor节点，用于监控工具调用

supervisor = create_supervisor(
   model=load_chat_model(
        configuration.model,
        configuration.base_url,
        configuration.api_key,
    ),
    agents=[research_agent,math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="last_message",
)

def create_handoff_tool(*, agent_name:str, description:str|None=None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help"

    @tool(name,description=description)
    def handoff_tool(
        state:Annotated[MessagesState,InjectedState],
        tool_call_id:Annotated[str,InjectedToolCallId]
    ) -> Command:
        toole_message ={
            "role":"tool",
            "content":f"Successfully Transfer to {agent_name}",
            "name":name,
            "tool_call_id":tool_call_id
        }
        return Command(
            goto=agent_name,
            update={
                **state,
                "messages":state["messages"]+[toole_message]
            },
            graph=Command.PARENT
        )
    return handoff_tool

# Handoffs
assign_to_research_agent = create_handoff_tool(agent_name="research_agent",description="Assign a research task to the research agent")
assign_to_math_agent = create_handoff_tool(agent_name="math_agent",description="Assign a math task to the math agent")



supervisor_agent=create_react_agent(
    model=load_chat_model(
        configuration.model,
        configuration.base_url,
        configuration.api_key,
    ),
    tools=[assign_to_research_agent,assign_to_math_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor" 
)

graph=(
    StateGraph(MessagesState)
    .add_node(supervisor_agent,destinations=("research_agent","math_agent",END))
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START,"supervisor")
    .add_edge("research_agent","supervisor")
    .add_edge("math_agent","supervisor")
    .add_edge("supervisor",END)
    .compile()
)

# Define the function that calls the model


async def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(
        configuration.model,
        configuration.base_url,
        configuration.api_key,
    ).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


# Define a new graph

builder = StateGraph(State, config_schema=Configuration)
builder.set_entry_point("call_model")

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
#graph = builder.compile(name="ReAct Agent")
