from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent


from dotenv import load_dotenv
from pydantic import SecretStr
from datetime import datetime
import os

load_dotenv()


api_key = os.getenv("api_key")
if not api_key:
    raise ValueError("API密钥未在环境变量中设置")

llm =  ChatOpenAI(
        model="gpt-4o",
        base_url="https://api.openai-proxy.org/v1",
        api_key=SecretStr(api_key),
    )

@tool
def get_current_time(input: str) -> str:
    '''
    获取当前时间
    '''
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

search = TavilySearch(max_results=5)

agent = create_react_agent(
    model = llm,
    tools = [get_current_time, search],
    prompt = "你是一个助手，请根据用户的问题给出回答",
)

result = agent.invoke({"messages": [{"role": "user", "content": "深圳的天气"}]})

print(result)





