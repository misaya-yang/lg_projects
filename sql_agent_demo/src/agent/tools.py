from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr
import json
import pymysql
from typing import Dict, Any, List
import re

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 从环境变量获取API密钥
api_key = os.getenv("api_key")
if not api_key:
    raise ValueError("API密钥未在环境变量中设置")

# llm = ChatOpenAI(
#     model="gpt-4o",
#     base_url="https://api.openai-proxy.org/v1",
#     api_key=SecretStr(api_key),
# )
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url="https://api.siliconflow.cn/v1",
    api_key=SecretStr("sk-qkqlzipgkhikealjjxtdhnycbjjbtwhiojbqtavtanzysgsm")
)

@tool
def extract_keyword_from_user_input(user_input: str) -> str:
    """
    从用户输入中智能提取关键词
    
    Args:
        user_input: 用户的原始输入，例如"根据玉溪关键字帮我生成舆情报告"
    
    Returns:
        提取的关键词，例如"玉溪"
    """
    print(f"=== 关键词提取工具开始 ===")
    print(f"用户输入: {user_input}")
    
    # 使用大模型智能提取关键词
    prompt = f"""
请从以下用户输入中提取最相关的关键词，用于舆情分析：

用户输入: {user_input}

请分析用户意图，提取出需要进行舆情分析的关键词。关键词应该是：
1. 具体的实体名称（如地名、品牌名、人名等）
2. 事件名称
3. 产品名称
4. 其他需要监控的关键词

请只返回关键词本身，不要包含其他解释。如果无法提取到明确的关键词，请返回"无法识别关键词"。

关键词:"""

    try:
        response = llm.invoke(prompt)
        keyword = response.content.strip()
        
        # 清理关键词，移除可能的标点符号和多余空格
        keyword = re.sub(r'[^\w\s\u4e00-\u9fff]', '', keyword).strip()
        
        print(f"提取的关键词: '{keyword}'")
        print(f"=== 关键词提取工具完成 ===")
        
        return keyword
    except Exception as e:
        print(f"关键词提取失败: {e}")
        return "无法识别关键词"

@tool
def query_sentiment_data(keyword: str) -> Dict[str, Any]:
    """
    根据关键词查询舆情数据
    
    Args:
        keyword: 要查询的关键词，例如"玉溪"
    
    Returns:
        包含帖子和回复数据的字典
    """
    print(f"=== 舆情数据查询工具开始 ===")
    print(f"查询关键词: '{keyword}'")
    
    if not keyword or keyword == "无法识别关键词":
        return {
            "error": "无法识别有效的关键词",
            "keyword": keyword,
            "posts_count": 0,
            "replies_count": 0,
            "posts": [],
            "replies": []
        }
    
    # 连接数据库
    print(f"正在连接数据库: 10.20.11.14:3306/test")
    connection = pymysql.connect(
        host='10.20.11.14',
        port=3306,
        user='rdadmin',
        password='Ld@513.',
        database='test',
        charset='utf8mb4'
    )
    
    try:
        with connection.cursor() as cursor:
            # 查询帖子数据
            posts_query = """
            SELECT dp.search_url, dp.content, dp.account_name, dp.datetime, 
                   dp.replies_count, dp.reposts_count, dp.likes_count, dp.views_count,
                   dp.platform_name, dp.url
            FROM data_posts dp
            WHERE dp.search_url LIKE %s OR dp.content LIKE %s
            ORDER BY dp.datetime DESC
            LIMIT 20
            """
            posts_params = (f"%{keyword}%", f"%{keyword}%")
            
            print(f"执行帖子查询SQL: {posts_query}")
            cursor.execute(posts_query, posts_params)
            posts_data = cursor.fetchall()
            
            print(f"查询到帖子数量: {len(posts_data)}")
            
            # 查询对应的回复数据
            replies_data = []
            if posts_data:
                post_ids = [str(row[0]) for row in posts_data if row[0]]
                
                if post_ids:
                    placeholders = ','.join(['%s'] * len(post_ids))
                    replies_query = f"""
                    SELECT dr.content, dr.account_name, dr.datetime, dr.post_url
                    FROM data_replies dr
                    WHERE dr.post_url IN ({placeholders})
                    ORDER BY dr.datetime DESC
                    LIMIT 50
                    """
                    
                    print(f"执行回复查询SQL")
                    cursor.execute(replies_query, post_ids)
                    replies_data = cursor.fetchall()
                    
                    print(f"查询到回复数量: {len(replies_data)}")
            
            # 清理和格式化数据
            cleaned_data = {
                "keyword": keyword,
                "posts_count": len(posts_data),
                "replies_count": len(replies_data),
                "posts": [],
                "replies": []
            }
            
            # 处理帖子数据
            for post in posts_data:
                try:
                    content_json = json.loads(post[1]) if post[1] else {}
                    post_content = content_json.get('content', '') if isinstance(content_json, dict) else str(post[1])
                    
                    cleaned_data["posts"].append({
                        "search_url": post[0],
                        "content": post_content,
                        "account_name": post[2],
                        "datetime": str(post[3]) if post[3] else "",
                        "replies_count": post[4] or 0,
                        "reposts_count": post[5] or 0,
                        "likes_count": post[6] or 0,
                        "views_count": post[7] or 0,
                        "platform_name": post[8],
                        "url": post[9]
                    })
                except Exception as e:
                    print(f"帖子处理失败: {e}")
                    continue
            
            # 处理回复数据
            for reply in replies_data:
                try:
                    content_json = json.loads(reply[0]) if reply[0] else {}
                    reply_content = content_json.get('content', '') if isinstance(content_json, dict) else str(reply[0])
                    
                    cleaned_data["replies"].append({
                        "content": reply_content,
                        "account_name": reply[1],
                        "datetime": str(reply[2]) if reply[2] else "",
                        "post_url": reply[3]
                    })
                except Exception as e:
                    print(f"回复处理失败: {e}")
                    continue
                    
    finally:
        connection.close()
        print(f"数据库连接已关闭")
    
    print(f"=== 舆情数据查询工具完成 ===")
    print(f"最终数据统计: 帖子{len(cleaned_data['posts'])}条, 回复{len(cleaned_data['replies'])}条")
    
    return cleaned_data

def get_tools() -> List:
    """获取所有可用的工具"""
    return [extract_keyword_from_user_input, query_sentiment_data] 