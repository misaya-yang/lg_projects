import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from dotenv import load_dotenv
from pydantic import SecretStr
from e2b_code_interpreter import Sandbox

# 加载环境变量
load_dotenv()

# 从环境变量获取API密钥
api_key = os.getenv("api_key")
if not api_key:
    raise ValueError("API密钥未在环境变量中设置")

# ------------ 1. 定义代码执行工具 ------------
@tool
def execute_python(code: str) -> str:
    """执行任意 Python 代码并返回结果
    
    Args:
        code: 要执行的Python代码，应该是纯Python代码，不要包含markdown格式
        
    Returns:
        代码执行的文本结果
    """
    try:
        with Sandbox() as sandbox:
            execution = sandbox.run_code(code)
            return execution.text or "执行完成，无输出"
    except Exception as e:
        return f"执行错误: {str(e)}"

# ------------ 2. 定义 Agent ------------
llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.openai-proxy.org/v1",
    api_key=SecretStr(api_key),
    temperature=0
)

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

# 创建简单的prompt模板
prompt = PromptTemplate.from_template(
    """你是一个AI助手，可以执行Python代码来解决用户问题。

可用工具：
{tools}

用户问题: {input}

{agent_scratchpad}

请根据用户需求编写并执行相应的Python代码。
"""
)

# 创建agent和executor
agent = create_react_agent(llm, [execute_python], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[execute_python], verbose=True)

# ------------ 3. 运行 ------------
if __name__ == "__main__":
    print("🚀 代码执行Agent已启动！")
    print("💡 示例任务：")
    print("   - 计算1到100的和")
    print("   - 创建一个简单的图表")
    print("   - 读取和分析数据")
    print("   - 进行数学计算")
    print()
    
    try:
        while True:
            task = input("\n💬 输入需求（或 q 退出）：")
            if task.lower() == "q":
                break
                
            print("\n🔄 正在处理...")
            try:
                reply = agent_executor.invoke({"input": task})
                print("📤 Agent 结果：")
                print(reply["output"])
            except Exception as e:
                print(f"❌ 执行出错: {str(e)}")
                
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    finally:
        print("✅ 程序已退出")