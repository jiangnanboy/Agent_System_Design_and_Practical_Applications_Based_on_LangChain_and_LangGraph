import asyncio
import os
os.environ["USER_AGENT"] = "chapter3/langgraph_example"  # 可自定义名称和联系方式

from typing import Optional, Dict, Any, List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# --- 定义状态结构 ---
class AnalysisState(TypedDict):
    """定义图中的状态结构，包含所有需要在节点间传递的数据"""
    url: str
    document_text: str
    sentiment: Optional[str]
    entities: Optional[Dict[str, List[str]]]
    topics: Optional[str]
    summary: Optional[str]
    final_report: Optional[str]


# --- 初始化llm ---
from init_client import init_llm

llm = init_llm(0.3)


# --- 文档加载和预处理 ---
def load_and_process_document(state: AnalysisState) -> AnalysisState:
    """加载并处理文档，返回更新后的状态"""
    loader = WebBaseLoader(state["url"])
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    full_text = "\n\n".join([doc.page_content for doc in texts])

    # 返回一个新的状态字典，只包含需要更新的键
    return {"document_text": full_text}


# --- 定义并行分析节点 ---
def analyze_sentiment(state: AnalysisState) -> Dict[str, str]:
    """分析文档的情感倾向，只返回sentiment"""
    if not llm:
        return {"sentiment": "语言模型未初始化"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "分析以下文本的整体情感倾向（积极、消极或中性），并提供简短解释："),
        ("user", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": state["document_text"]})

    # 只返回本节点新增的数据
    return {"sentiment": result}


def extract_entities(state: AnalysisState) -> Dict[str, Any]:
    """提取文档中的关键实体，只返回entities"""
    if not llm:
        return {"entities": {"error": "语言模型未初始化"}}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "从以下文本中提取并列出所有重要实体（人物、组织、地点等），以JSON格式返回，包含实体类型和实体名称："),
        ("user", "{text}")
    ])
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"text": state["document_text"]})

    # 只返回本节点新增的数据
    return {"entities": result}


def extract_topics(state: AnalysisState) -> Dict[str, str]:
    """提取文档的主要主题，只返回topics"""
    if not llm:
        return {"topics": "语言模型未初始化"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "从以下文本中识别3-5个主要主题或概念，每个主题不超过5个词："),
        ("user", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": state["document_text"]})

    # 只返回本节点新增的数据
    return {"topics": result}


def generate_summary(state: AnalysisState) -> Dict[str, str]:
    """生成文档的摘要，只返回summary"""
    if not llm:
        return {"summary": "语言模型未初始化"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "为以下文本生成一个简洁的摘要（不超过200字）："),
        ("user", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": state["document_text"]})

    # 只返回本节点新增的数据
    return {"summary": result}


def synthesize_report(state: AnalysisState) -> Dict[str, str]:
    """综合所有分析结果生成最终报告，只返回final_report"""
    if not llm:
        return {"final_report": "语言模型未初始化"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", """基于以下分析结果：
        情感分析：{sentiment}
        关键实体：{entities}
        主要主题：{topics}
        内容摘要：{summary}

        生成一个全面的分析报告，包含以上所有方面的洞察，并突出最重要的发现。"""),
        ("user", "原始文本片段：{text}...")
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "sentiment": state["sentiment"],
        "entities": state["entities"],
        "topics": state["topics"],
        "summary": state["summary"],
        "text": state["document_text"][:500]
    })

    # 只返回本节点新增的数据
    return {"final_report": result}


# --- 构建图 ---
def build_analysis_graph():
    """构建并返回分析图"""
    workflow = StateGraph(AnalysisState)

    # 添加节点
    workflow.add_node("load_document", load_and_process_document)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("extract_topics", extract_topics)
    workflow.add_node("generate_summary", generate_summary)
    workflow.add_node("synthesize_report", synthesize_report)

    # 设置入口点
    workflow.set_entry_point("load_document")

    # 添加边 - 从文档加载到并行分析节点
    workflow.add_edge("load_document", "analyze_sentiment")
    workflow.add_edge("load_document", "extract_entities")
    workflow.add_edge("load_document", "extract_topics")
    workflow.add_edge("load_document", "generate_summary")

    # 添加边 - 从并行分析节点到综合节点
    workflow.add_edge("analyze_sentiment", "synthesize_report")
    workflow.add_edge("extract_entities", "synthesize_report")
    workflow.add_edge("extract_topics", "synthesize_report")
    workflow.add_edge("generate_summary", "synthesize_report")

    # 设置结束点
    workflow.add_edge("synthesize_report", END)

    # 编译图
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# --- 运行图 ---
async def analyze_document_with_graph(url: str) -> Dict[str, Any]:
    if not llm:
        print("LLM未初始化。无法运行示例。")
        return {"error": "LLM未初始化"}

    app = build_analysis_graph()
    # 打印图的结构（可选，非常直观！）
    try:
        print("--- 图结构 ---")
        app.get_graph().print_ascii()
        print("\n" + "=" * 20 + "\n")
    except Exception as e:
        print(f"无法打印图结构: {e}")

    print(f"\n--- 开始使用LangGraph分析文档：{url} ---")

    try:
        # 初始状态
        initial_state = {"url": url}

        # 运行图
        config = {"configurable": {"thread_id": "1"}}
        final_state = await app.ainvoke(initial_state, config=config)

        print("\n--- 分析结果 ---")
        print(final_state["final_report"])

        return final_state
    except Exception as e:
        print(f"\n分析过程中发生错误：{e}")
        return {"error": str(e)}


if __name__ == "__main__":
    article_url = "https://hub.baai.ac.cn/view/50604"
    asyncio.run(analyze_document_with_graph(article_url))