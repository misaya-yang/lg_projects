from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr
import json
import pymysql
from typing import Dict, Any

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph,START,END,MessagesState

from src.agent.prompt import system_prompt,report_prompt
from src.agent.smart_agent_prompt import smart_agent_prompt
from src.agent.tools import get_tools

# 加载sql_agent_demo目录下的.env文件
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 从环境变量获取API密钥
api_key = os.getenv("api_key")
if not api_key:
    raise ValueError("API密钥未在环境变量中设置")

llm2 =  ChatOpenAI(
        model="gpt-4o",
        base_url="https://api.openai-proxy.org/v1",
        api_key=SecretStr(api_key),
    )

llm4 = ChatOpenAI(
            model="qwen2.5:32b",
            base_url="http://10.40.1.3:11434/v1",
            api_key=SecretStr("sk-1234567890")
        )
llm = ChatOpenAI(
            model="deepseek-ai/DeepSeek-V3",
            base_url="https://api.siliconflow.cn/v1",
            api_key=SecretStr("sk-qkqlzipgkhikealjjxtdhnycbjjbtwhiojbqtavtanzysgsm")
        )
llm3 = ChatOpenAI(
        model="XGenerationLab/XiYanSQL-QwenCoder-32B-2412",
        base_url="https://api-inference.modelscope.cn/v1/",
        api_key=SecretStr("ms-19b7a775-61c5-4a56-89d6-9c099652668d"),
    )

#db = SQLDatabase.from_uri("sqlite:////Users/apple/Desktop/lg_projects/sql_agent_demo/Chinook.db")
db = SQLDatabase.from_uri("mysql+pymysql://rdadmin:Ld%40513.@10.20.11.14:3306/test")

toolkit = SQLDatabaseToolkit(db=db,llm=llm)
sql_tools = toolkit.get_tools()

# 获取舆情分析工具
sentiment_tools = get_tools()

# 创建智能舆情分析agent
smart_sentiment_agent = create_react_agent(
    model=llm,
    tools=sentiment_tools,
    prompt=smart_agent_prompt,
    name="smart_sentiment_agent"
)

# 创建SQL查询agent
sql_agent = create_react_agent(
    model=llm,
    tools=sql_tools,
    prompt=system_prompt.format(dialect="mysql",top_k=5),
    name="sql_agent"
)

# 创建报告生成agent
report_agent = create_react_agent(
    model=llm,
    tools=sql_tools,
    prompt=report_prompt,
    name="report_agent"
)

# 创建主图，使用智能舆情分析agent + 报告生成agent
graph = (
    StateGraph(MessagesState)
    .add_node("smart_sentiment_agent", smart_sentiment_agent)
    .add_node("report_agent", report_agent)
    .add_edge(START, "smart_sentiment_agent")
    .add_edge("smart_sentiment_agent", "report_agent")
    .add_edge("report_agent", END)
    .compile()
)

# 如果需要使用原来的SQL agent和报告生成流程，可以使用以下图：
# graph = (
#     StateGraph(MessagesState)
#     .add_node("sql_agent", sql_agent)
#     .add_node("report_agent", report_agent)
#     .add_edge(START, "sql_agent")
#     .add_edge("sql_agent", "report_agent")
#     .add_edge("report_agent", END)
#     .compile()
# )


