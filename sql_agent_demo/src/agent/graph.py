from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph,START,END,MessagesState

from src.agent.prompt import system_prompt

# 加载sql_agent_demo目录下的.env文件
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)



# 从环境变量获取API密钥
api_key = os.getenv("api_key")
if not api_key:
    raise ValueError("API密钥未在环境变量中设置")

llm =  ChatOpenAI(
        model="gpt-4o",
        base_url="https://api.openai-proxy.org/v1",
        api_key=SecretStr(api_key),
    )

#db = SQLDatabase.from_uri("sqlite:////Users/apple/Desktop/lg_projects/sql_agent_demo/Chinook.db")
db = SQLDatabase.from_uri("mysql+pymysql://root:123456dccs@localhost:3306/resume")
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


