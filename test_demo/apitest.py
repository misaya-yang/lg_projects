
import argparse
import sys
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from datetime import datetime

def create_llm(model_choice: int) -> ChatOpenAI:
    """根据选择创建对应的LLM实例"""
    if model_choice == 1:
        return ChatOpenAI(
            model="deepseek-v3:671b",
            base_url="http://10.40.0.100:8081/v1",
            api_key=SecretStr("sk-1234567890")
        )
    elif model_choice == 2:
        return ChatOpenAI(
            model="gpt-4o",
            base_url="https://api.openai-proxy.org/v1",
            api_key=SecretStr("sk-Pi81m0dUmZvpFOXJwa0erWEjybri0Yqq6Ay8U4H7xBSPhB8O"),
        )
    elif model_choice == 3:
        return ChatOpenAI(
            model="deepseek-v3:671b",
            base_url="http://10.40.0.100:8081/v1",
            api_key=SecretStr("sk-1234567890")
        )
    elif model_choice == 4:
        return ChatOpenAI(
            model="gpt-4o",
            base_url="https://api.openai-proxy.org/v1",
            api_key=SecretStr("sk-Pi81m0dUmZvpFOXJwa0erWEjybri0Yqq6Ay8U4H7xBSPhB8O"),
        )
    else:
        raise ValueError("模型选择必须是 1、2、3 或 4")

@tool
def get_current_time(input: str) -> str:
    '''获取当前时间'''
    return "2025-07-24 10:00:00"

def create_agent(llm: ChatOpenAI, use_tools: bool = False):

    prompt_template = """
You are a helpful assistant with access to tools.

Use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question

Begin!

Question: {input}
"""
    """创建agent，可选择是否使用工具"""
    if use_tools:
        return create_react_agent(
            model=llm,
            tools=[get_current_time],
            prompt=prompt_template.format(tool_names=get_current_time.__name__,input=input)
        )
    else:
        return create_react_agent(
            model=llm,
            tools=[],
            prompt="You are a helpful assistant."
        )

def extract_ai_response(res: dict) -> str:
    """从agent响应中提取AI回复"""
    if "messages" in res:
        last_message = res["messages"][-1]
        if hasattr(last_message, 'content'):
            return last_message.content
    return str(res)

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="AI助手工具 - 支持多种模型选择",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python apitest.py "写一首春天的诗" 1
  python apitest.py "解释什么是机器学习" 2
  python apitest.py "现在几点了" 3
  python apitest.py "获取当前时间" 4
        """
    )
    
    parser.add_argument(
        "prompt", 
        type=str, 
        help="要发送给AI的提示内容"
    )
    
    parser.add_argument(
        "model", 
        type=int, 
        choices=[1, 2, 3, 4], 
        help="选择模型: 1=deepseek-v3(无工具), 2=gpt-4o(无工具), 3=deepseek-v3(有工具), 4=gpt-4o(有工具)"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 创建LLM实例
        print(f"🚀 正在初始化模型 {args.model}...")
        llm = create_llm(args.model)
        
        # 创建agent
        use_tools = args.model in [3, 4]
        tool_status = "（带工具）" if use_tools else "（无工具）"
        print(f"🤖 正在创建AI助手{tool_status}...")
        agent = create_agent(llm, use_tools=use_tools)
        
        # 发送请求
        print(f"📤 正在发送请求: {args.prompt}")
        res = agent.invoke({
            "messages": [{"role": "user", "content": args.prompt}]
        })
        
        # 提取并显示结果
        print("\n📤 AI回复：")
        print("=" * 50)
        ai_response = extract_ai_response(res)
        print(ai_response)
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

