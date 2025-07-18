from langchain_openai import ChatOpenAI

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph,START,END,MessagesState

from src.agent.prompt import system_prompt

llm =  ChatOpenAI(
        model="gpt-4o",
        base_url="https://api.openai-proxy.org/v1",
        api_key="sk-Pi81m0dUmZvpFOXJwa0erWEjybri0Yqq6Ay8U4H7xBSPhB8O",
    )

#db = SQLDatabase.from_uri("sqlite:////Users/apple/Desktop/lg_projects/sql_agent_demo/Chinook.db")
db = SQLDatabase.from_uri("mysql+pymysql://root:123456dc@localhost:3306/resume")
toolkit = SQLDatabaseToolkit(db=db,llm=llm)
tools = toolkit.get_tools()

sql_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt.format(dialect="mysql",top_k=5),
    name="sql_agent"
)

graph =(
    StateGraph(MessagesState)
    .add_node("sql_agent",sql_agent)
    .add_edge(START,"sql_agent")
    .add_edge("sql_agent",END)
    .compile()
)


