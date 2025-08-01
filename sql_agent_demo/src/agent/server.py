from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Dict, Any, List
import logging

from src.agent.smart_agent_prompt import smart_agent_prompt
from src.agent.tools import get_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr
from fastapi.responses import StreamingResponse

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 从环境变量获取API密钥
api_key = os.getenv("api_key")
if not api_key:
    raise ValueError("API密钥未在环境变量中设置")

# 创建LLM实例
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url="https://api.siliconflow.cn/v1",
    api_key=SecretStr("sk-qkqlzipgkhikealjjxtdhnycbjjbtwhiojbqtavtanzysgsm")
)

# 获取舆情分析工具
sentiment_tools = get_tools()

# 创建智能舆情分析agent
smart_sentiment_agent = create_react_agent(
    model=llm,
    tools=sentiment_tools,
    prompt=smart_agent_prompt,
    name="smart_sentiment_agent"
)

# 创建图
graph = (
    StateGraph(MessagesState)
    .add_node("smart_sentiment_agent", smart_sentiment_agent)
    .add_edge(START, "smart_sentiment_agent")
    .add_edge("smart_sentiment_agent", END)
    .compile()
)

# 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str

class MessagesRequest(BaseModel):
    messages: List[Dict[str, str]]

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

class StreamResponse(BaseModel):
    chunk: Dict[str, Any]

# 创建FastAPI应用
app = FastAPI(
    title="智能舆情分析API",
    description="基于LangGraph的智能舆情分析系统，支持关键词提取和报告生成",
    version="1.0.0",
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 根路径 - 返回前端页面
@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# 健康检查端点
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "智能舆情分析API运行正常"}

# 配置端点
@app.get("/config")
def get_config():
    return {
        "name": "智能舆情分析系统",
        "description": "基于关键词生成社交网络舆情报告",
        "features": [
            "智能关键词提取",
            "舆情数据查询",
            "专业报告生成"
        ],
        "endpoints": {
            "/chat": "POST - 聊天对话",
            "/chat/stream": "POST - 流式聊天",
            "/tools/test": "POST - 测试工具"
        }
    }

# 聊天端点
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"收到请求: {request.message}")
        
        # 创建消息
        messages = [HumanMessage(content=request.message)]
        
        # 调用图
        result = graph.invoke({"messages": messages})
        
        # 获取最终回复
        final_message = result["messages"][-1]
        response_content = final_message.content
        
        logger.info("请求处理完成")
        
        return ChatResponse(response=response_content)
        
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 流式聊天端点
@app.post("/chat/stream")
async def chat_stream(request: MessagesRequest):
    return await _chat_stream_impl(request)

@app.get("/chat/stream")
async def chat_stream_get(message: str):
    # 支持GET请求，从query参数获取消息
    request = MessagesRequest(messages=[{"role": "user", "content": message}])
    return await _chat_stream_impl(request)

async def _chat_stream_impl(request: MessagesRequest):
    logger.info(f"收到流式请求: {request.messages}")
    messages = []
    for msg in request.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))

    def event_generator():
        for chunk in graph.stream({"messages": messages}):
            # 只提取AI回复的content并且非空
            try:
                ai_messages = chunk.get("smart_sentiment_agent", {}).get("messages", [])
                for m in ai_messages:
                    # 检查是否是AI消息且有内容
                    if hasattr(m, 'type') and m.type == "ai" and hasattr(m, 'content') and m.content:
                        yield f"data: {m.content}\n\n"
            except Exception as e:
                yield f"data: [error]{str(e)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 工具测试端点
@app.post("/tools/test")
async def test_tools():
    """测试工具功能"""
    try:
        from src.agent.tools import extract_keyword_from_user_input, query_sentiment_data
        
        # 测试关键词提取
        test_input = "根据玉溪关键字帮我生成舆情报告"
        keyword = extract_keyword_from_user_input.invoke(test_input)
        
        result = {
            "keyword_extraction": {
                "input": test_input,
                "output": keyword
            }
        }
        
        # 如果关键词提取成功，测试数据查询
        if keyword and keyword != "无法识别关键词":
            data = query_sentiment_data.invoke(keyword)
            result["data_query"] = {
                "keyword": keyword,
                "posts_count": data.get("posts_count", 0),
                "replies_count": data.get("replies_count", 0)
            }
        
        return result
        
    except Exception as e:
        logger.error(f"测试工具时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 直接调用图端点
@app.post("/graph/invoke")
async def invoke_graph(request: Dict[str, Any]):
    """直接调用图，接受完整的消息格式"""
    try:
        logger.info("收到图调用请求")
        result = graph.invoke(request)
        return result
    except Exception as e:
        logger.error(f"调用图时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)