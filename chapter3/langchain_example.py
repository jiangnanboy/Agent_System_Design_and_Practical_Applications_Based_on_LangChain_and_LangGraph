import asyncio
import os
os.environ["USER_AGENT"] = "chapter3/langchain_example"  # 可自定义名称和联系方式

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 初始化llm ---
from init_client import init_llm

llm = init_llm(0.3)


# --- 文档加载和预处理 ---
def load_and_process_document(url: str) -> str:
    """加载并处理文档，返回文本内容"""
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    # 合并所有文本块为一个完整文档
    full_text = "\n\n".join([doc.page_content for doc in texts])
    return full_text


# --- 定义并行分析链 ---
# 情感分析链
sentiment_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "分析以下文本的整体情感倾向（积极、消极或中性），并提供简短解释："),
            ("user", "{text}")
        ])
        | llm
        | StrOutputParser()
)

# 关键实体提取链
entities_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "从以下文本中提取并列出所有重要实体（人物、组织、地点等），以JSON格式返回，包含实体类型和实体名称："),
            ("user", "{text}")
        ])
        | llm
        | JsonOutputParser()
)

# 主题提取链
topics_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "从以下文本中识别3-5个主要主题或概念，每个主题不超过5个词："),
            ("user", "{text}")
        ])
        | llm
        | StrOutputParser()
)

# 摘要生成链
summary_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "为以下文本生成一个简洁的摘要（不超过200字）："),
            ("user", "{text}")
        ])
        | llm
        | StrOutputParser()
)

# --- 构建并行 + 综合链 ---
# 1. 定义要并行运行的任务块
parallel_analysis = RunnableParallel(
    {
        "sentiment": sentiment_chain,
        "entities": entities_chain,
        "topics": topics_chain,
        "summary": summary_chain,
        "text": RunnablePassthrough(),  # 传递原始文本
    }
)

# 2. 定义将组合并行结果的最终综合提示词
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """基于以下分析结果：
    情感分析：{sentiment}
    关键实体：{entities}
    主要主题：{topics}
    内容摘要：{summary}

    生成一个全面的分析报告，包含以上所有方面的洞察，并突出最重要的发现。"""),
    ("user", "原始文本片段：{text}...")
])

# 3. 构建完整链
full_analysis_chain = parallel_analysis | synthesis_prompt | llm | StrOutputParser()


# --- 运行链 ---
async def analyze_document(url: str) -> None:
    """
    异步分析文档，并行执行多种分析任务
    参数：
        url: 要分析的文档URL
    """
    if not llm:
        print("LLM未初始化。无法运行示例。")
        return

    # print(f"\n--- 开始分析文档：{url} ---")
    try:
        # 加载并处理文档
        document_text = load_and_process_document(url)
        document_text = document_text[:500]
        print(f"文档加载完成，长度：{len(document_text)} 字符")

        # 并行执行分析
        print("\n--- 执行并行分析 ---")
        response = await full_analysis_chain.ainvoke(document_text)

        print("\n--- 分析结果 ---")
        print(response)
    except Exception as e:
        print(f"\n分析过程中发生错误：{e}")


if __name__ == "__main__":
    article_url = "https://hub.baai.ac.cn/view/50604"
    asyncio.run(analyze_document(article_url))

