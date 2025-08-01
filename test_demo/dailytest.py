from langchain_core.prompts import prompt
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from pydantic import SecretStr
llm2 = ChatOpenAI(
            model="deepseek-v3:671b",
            base_url="http://10.40.0.100:8081/v1",
            api_key=SecretStr("sk-1234567890")
        )
llm = ChatOpenAI(
            model="qwen2.5:32b",
            base_url="http://10.40.1.3:11434/v1",
            api_key=SecretStr("sk-1234567890")
        )

@tool
def get_current_time(input: str) -> str:
    '''获取当前时间'''
    return "2025-07-24 10:00:00"

agent = create_react_agent(model = llm, tools=[get_current_time], prompt = "you are a helpful assistant")

res= agent.invoke({"messages": [{"role": "user", "content": "what is the current time?"}]})
print(res)


