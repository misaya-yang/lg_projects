from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from src.agent.prompt import system_prompt,generate_query_system_prompt,check_query_system_prompt


llm =  ChatOpenAI(
        model="gpt-4o",
        base_url="https://api.openai-proxy.org/v1",
        api_key="sk-Pi81m0dUmZvpFOXJwa0erWEjybri0Yqq6Ay8U4H7xBSPhB8O",
    )

#db = SQLDatabase.from_uri("sqlite:////Users/apple/Desktop/lg_projects/sql_agent_demo/Chinook.db")
db = SQLDatabase.from_uri("mysql+pymysql://root:123456dc@localhost:3306/resume")
toolkit = SQLDatabaseToolkit(db=db,llm=llm)
tools = toolkit.get_tools()

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool],name="get_schema")

run_query_tool= next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool],name="run_query")


def list_tables(state:MessagesState):
    tool_call ={
        "name":"sql_db_list_tables",
        "args":{},
        "id":"abc123",
        "type":"tool_call"
    }
    tool_call_message = AIMessage(content="",tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {
        "messages":[tool_call_message,tool_message,response]
    }

def call_get_schema(state:MessagesState):
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages":[response]
    }

def call_run_query(state:MessagesState):
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages":[response]
    }

def generate_query(state:MessagesState):
    system_message = {
        "role":"system",
        "content":generate_query_system_prompt.format(dialect="mysql",top_k=5)
    }
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages":[response]}

def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"


#构建图
builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

graph = builder.compile()


