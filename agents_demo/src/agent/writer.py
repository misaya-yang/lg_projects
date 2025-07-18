from typing import List, TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.agent.configuration import Configuration

# 定义 State
class State(TypedDict):
    user_input: dict              # 用户输入，包括标题、构思、章节数等
    outline: Optional[List[dict]] # 生成的大纲
    outline_feedback: Optional[str]  # 用户对大纲的反馈（用于调整大纲）
    chapter_index: int            # 当前章节的索引
    chapter_text: Optional[str]   # 当前章节生成的内容
    human_feedback: Optional[str] # 人类对章节内容的反馈

# 初始化配置
configuration = Configuration.from_context()
llm = ChatOpenAI(
    model=configuration.model,
    base_url=configuration.base_url,
    api_key=configuration.api_key,
)

# 大纲生成节点
def generate_outline(state: State):
    user_input = state["user_input"]
    outline_feedback = state.get("outline_feedback", None)

    if outline_feedback:
        print("根据用户反馈调整大纲中...")
        outline_prompt = (
            "以下是用户提出的大纲修改建议，请基于建议重新生成小说大纲，严格只输出JSON格式：\n"
            f"{outline_feedback}\n\n"
            "输出格式：[{\"name\": \"第1章\", \"summary\": \"本章大纲...\"}]\n"
        )
    else:
        outline_prompt = (
            "请根据以下要求生成小说大纲，严格只输出JSON格式（不要有任何解释、代码块标记等）：\n"
            "输出格式：[{\"name\": \"第1章\", \"summary\": \"本章大纲...\"}]\n"
            f"标题：{user_input['title']}\n"
            f"构思：{user_input['idea']}\n"
            f"章节数：{user_input['chapter_cnt']}\n"
            "注意：只输出JSON数组，不要有任何其他文字。"
        )

    response = llm.invoke([HumanMessage(content=outline_prompt)])
    
    # 健壮的 JSON 解析
    import re
    import json
    
    content = str(response.content).strip()
    print("LLM返回内容：", repr(content))
    
    # 尝试直接解析
    try:
        outline = json.loads(content)
    except json.JSONDecodeError:
        # 尝试提取 JSON 部分
        match = re.search(r'(\[.*\])', content, re.DOTALL)
        if match:
            try:
                outline = json.loads(match.group(1))
            except json.JSONDecodeError:
                raise ValueError(f"无法解析JSON: {content}")
        else:
            raise ValueError(f"未找到JSON格式内容: {content}")
    
    # 验证格式
    if not isinstance(outline, list):
        raise ValueError("大纲必须是数组格式")
    
    return {
        "outline": outline, 
        "chapter_index": 0, 
        "outline_feedback": None
    }

# 单章节生成节点
def generate_chapter(state: State):
    outline = state.get("outline")
    if not outline:
        raise ValueError("大纲未生成，请先生成大纲")
    
    chapter_index = state["chapter_index"]
    
    if chapter_index >= len(outline):
        raise ValueError("章节索引超出范围，可能大纲已全部生成")

    chapter_data = outline[chapter_index]
    feedback = state.get("human_feedback", "")
    if feedback:
        feedback = str(feedback).strip()
    
    chapter_prompt = (
        f"根据以下大纲生成小说正文：\n"
        f"章节名：{chapter_data['name']}\n"
        f"章节大纲：{chapter_data['summary']}\n"
    )
    if feedback:
        chapter_prompt += f"人类修订建议：{feedback}\n"
    chapter_prompt += "请生成流畅且逻辑连贯的章节内容。"

    response = llm.invoke([HumanMessage(content=chapter_prompt)])
    return {
        "chapter_text": str(response.content).strip(),
        "human_feedback": None
    }

# 提取用户输入的函数
def extract_user_input(interrupt_data):
    if isinstance(interrupt_data, dict) and interrupt_data:
        return str(list(interrupt_data.values())[0]).strip()
    return str(interrupt_data).strip()

# 人类审阅节点（处理章节反馈）
def human_node(state: State):
    chapter_text = state.get("chapter_text", "")
    if not chapter_text:
        print("当前没有章节内容")
        return {}
    
    print(f"当前章节内容：\n{chapter_text[:500]}...\n")
    print("请选择操作：")
    print("1. 输入 `accept`：接受此章节内容，进入下一章节；")
    print("2. 输入 `revise`：重新生成当前章节内容；")
    print("3. 输入 `outline`：修改大纲；")
    print("4. 或直接输入具体修改建议。")

    # 获取用户输入
    feedback_data = interrupt(
        {
            "text_to_review": chapter_text[:100],
            "feedback": "请审阅并提供修改建议，或按 Enter 跳过。"
        }
    )
    
    # 解析 interrupt 返回的内容
    feedback = ""
    print("--------------------------------")
    print(f"feedback_data: {feedback_data}")
    
    if isinstance(feedback_data, dict):
        # 提取字典中的值（用户输入）
        values = list(feedback_data.values())
        if values:
            feedback = str(values[0]).strip()
    else:
        feedback = str(feedback_data).strip()

    print(f"用户输入：{feedback}")

    # 根据用户输入进行判断和处理
    if feedback.lower() == "accept":
        return {"chapter_index": state["chapter_index"] + 1}
    elif feedback.lower() == "revise":
        return {"human_feedback": "请重新生成当前章节。"}
    elif feedback.lower() == "outline":
        # 获取大纲修改建议
        outline_feedback_data = interrupt({
            "prompt": "请输入对大纲的调整建议："
        })
        
        outline_feedback = ""
        if isinstance(outline_feedback_data, dict):
            values = list(outline_feedback_data.values())
            if values:
                outline_feedback = str(values[0]).strip()
        else:
            outline_feedback = str(outline_feedback_data).strip()
            
        return {"outline_feedback": outline_feedback}
    else:
        # 用户输入了具体的修改建议
        return {"human_feedback": feedback}

# 判断流程是否继续
def should_continue(state: State):
    if state.get("outline_feedback"):
        return "generate_outline"
    elif state.get("human_feedback"):
        return "generate_chapter"
    
    outline = state.get("outline")
    if outline and state["chapter_index"] < len(outline):
        return "generate_chapter"
    return END


# 定义工作流
workflow_builder = StateGraph(State)

# 添加节点
workflow_builder.add_node("generate_outline", generate_outline)  # 大纲生成节点
workflow_builder.add_node("generate_chapter", generate_chapter)  # 章节生成节点
workflow_builder.add_node("review_chapter", human_node)          # 章节审阅（人机互动）节点

# 添加固定边
workflow_builder.add_edge(START, "generate_outline")  # 起始点 -> 生成大纲
workflow_builder.add_edge("generate_outline", "generate_chapter")  # 生成大纲 -> 生成章节
workflow_builder.add_edge("generate_chapter", "review_chapter")  # 生成章节 -> 人机审阅

# 添加条件边（review_chapter 分支处理逻辑）
workflow_builder.add_conditional_edges(
    "review_chapter", should_continue, {
        "generate_chapter": "generate_chapter",  # 用户要求重写章节 -> 回到章节生成节点
        "generate_outline": "generate_outline",  # 用户要求重写大纲 -> 回到生成大纲节点
        END: END  # 流程结束
    }
)

# 编译完整工作流
novel_workflow = workflow_builder.compile()
