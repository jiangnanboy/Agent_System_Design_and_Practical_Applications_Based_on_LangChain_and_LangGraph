import os
import json
import re
from typing import List, Dict, Any, Optional, TypedDict, Literal
from pathlib import Path

# PDF处理库
import pdfplumber

# LangChain & LangGraph 组件
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser

# LangGraph 核心组件
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from init_client import init_llm

llm = init_llm(0.1)
# --- 1. 定义 Agent 的状态 ---
class AgentState(TypedDict):
    """定义在整个图中流转的状态"""
    pdf_path: str
    paper_text: str
    goals: List[str]
    analysis: Dict[str, Any]
    current_summary: str
    evaluation: Dict[str, Any]
    iterations: int
    max_iterations: int
    final_result: Optional[Dict[str, Any]]
    error_message: Optional[str]


# --- 2. 定义图的节点 (每个节点是一个执行步骤) ---

def parse_pdf_node(state: AgentState) -> AgentState:
    """节点1: 解析PDF文件"""
    print("🔍 节点: 解析PDF文件...")
    pdf_path = state["pdf_path"]
    if not os.path.exists(pdf_path):
        return {"error_message": f"PDF文件不存在: {pdf_path}"}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []
            max_pages = min(len(pdf.pages), 10)
            for i in range(max_pages):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    text_content.append(text)
            paper_text = "\n\n".join(text_content)
            paper_text = re.sub(r'\s+', ' ', paper_text)
            print(f"✅ PDF解析完成，共提取 {len(paper_text)} 字符")
            return {"paper_text": paper_text}
    except Exception as e:
        return {"error_message": f"PDF解析失败: {e}"}


def analyze_paper_node(state: AgentState) -> AgentState:
    """节点2: 分析论文内容"""
    print("🔍 节点: 分析论文内容...")
    paper_text = state["paper_text"]

    analysis_schemas = [
        ResponseSchema(name="title", description="论文标题"),
        ResponseSchema(name="authors", description="论文作者列表"),
        ResponseSchema(name="abstract", description="论文摘要"),
        ResponseSchema(name="key_findings", description="主要发现，以列表形式呈现"),
        ResponseSchema(name="methodology", description="研究方法简述"),
    ]
    analysis_parser = StructuredOutputParser.from_response_schemas(analysis_schemas)

    prompt = PromptTemplate(
        input_variables=["paper_text", "format_instructions"],
        template="""
你是一个专业的研究论文分析助手。请仔细阅读以下研究论文，并提取关键信息。
{format_instructions}

论文内容:
{paper_text}

请确保提取的信息准确完整。
        """
    )
    messages = [
        SystemMessage(content="你是一个专业的学术研究分析专家。"),
        HumanMessage(content=prompt.format(
            paper_text=paper_text[:8000],  # 限制长度
            format_instructions=analysis_parser.get_format_instructions()
        ))
    ]
    response = llm.invoke(messages)
    analysis = analysis_parser.parse(response.content)

    print("✅ 论文分析完成")
    return {"analysis": analysis}


def generate_summary_node(state: AgentState) -> AgentState:
    """节点3: 生成初始摘要"""
    print("🔍 节点: 生成初始摘要...")
    analysis = state["analysis"]
    goals = state["goals"]

    prompt = PromptTemplate(
        input_variables=["goals", "analysis"],
        template="""
基于以下研究论文分析，生成一个简洁明了的摘要，满足以下目标:
{goals}

论文分析:
{analysis}

摘要应该:
1. 简明扼要，不超过200字
2. 突出研究的主要贡献
3. 使用清晰易懂的语言
4. 避免技术术语过多

请直接返回摘要内容，不要包含其他解释。
        """
    )

    messages = [
        SystemMessage(content="你是一个专业的学术写作专家，擅长将复杂研究转化为简洁易懂的摘要。"),
        HumanMessage(content=prompt.format(
            goals=", ".join(goals),
            analysis=json.dumps(analysis, ensure_ascii=False, indent=2)
        ))
    ]
    response = llm.invoke(messages)
    summary = response.content.strip()

    print("✅ 初始摘要生成完成")
    return {"current_summary": summary, "iterations": 1}


def evaluate_summary_node(state: AgentState) -> AgentState:
    """节点4: 评估摘要质量"""
    print("🔍 节点: 评估摘要质量...")
    summary = state["current_summary"]
    analysis = state["analysis"]
    goals = state["goals"]

    eval_schemas = [
        ResponseSchema(name="meets_goals", description="摘要是否满足所有设定目标，回答'是'或'否'"),
        ResponseSchema(name="accuracy_score", description="摘要准确度评分，1-10"),
        ResponseSchema(name="clarity_score", description="摘要清晰度评分，1-10"),
        ResponseSchema(name="feedback", description="改进建议，如果不满足目标"),
    ]
    eval_parser = StructuredOutputParser.from_response_schemas(eval_schemas)

    prompt = PromptTemplate(
        input_variables=["goals", "analysis", "summary", "format_instructions"],
        template="""
评估以下研究论文摘要是否满足设定的目标:

目标: {goals}

论文分析:
{analysis}

摘要:
{summary}

{format_instructions}

请客观评估摘要质量，并提供具体的改进建议。
        """
    )

    messages = [
        SystemMessage(content="你是一个严谨的学术评估专家，擅长评估研究摘要的质量。"),
        HumanMessage(content=prompt.format(
            goals=", ".join(goals),
            analysis=json.dumps(analysis, ensure_ascii=False, indent=2),
            summary=summary,
            format_instructions=eval_parser.get_format_instructions()
        ))
    ]
    response = llm.invoke(messages)
    evaluation = eval_parser.parse(response.content)

    print(f"✅ 评估完成 - 满足目标: {evaluation.get('meets_goals', '未知')}")
    return {"evaluation": evaluation}


def improve_summary_node(state: AgentState) -> AgentState:
    """节点5: 改进摘要"""
    print("🔍 节点: 改进摘要...")
    summary = state["current_summary"]
    feedback = state["evaluation"].get("feedback", "需要改进")
    goals = state["goals"]

    prompt = PromptTemplate(
        input_variables=["summary", "feedback", "goals"],
        template="""
根据以下反馈改进研究论文摘要:

当前摘要:
{summary}

改进反馈:
{feedback}

目标要求:
{goals}

请提供改进后的摘要，要求:
1. 保持简洁明了，不超过200字
2. 充分考虑反馈意见
3. 确保满足所有目标要求

直接返回改进后的摘要，不要包含其他解释。
        """
    )

    messages = [
        SystemMessage(content="你是一个专业的学术写作改进专家，擅长根据反馈优化研究摘要。"),
        HumanMessage(content=prompt.format(
            summary=summary,
            feedback=feedback,
            goals=", ".join(goals)
        ))
    ]
    response = llm.invoke(messages)
    improved_summary = response.content.strip()

    print("✅ 摘要改进完成")
    # 增加迭代次数
    return {"current_summary": improved_summary, "iterations": state["iterations"] + 1}


def save_results_node(state: AgentState) -> AgentState:
    """节点6: 保存最终结果"""
    print("🔍 节点: 保存最终结果...")
    pdf_path = state["pdf_path"]
    analysis = state["analysis"]
    summary = state["current_summary"]
    evaluation = state["evaluation"]

    pdf_name = Path(pdf_path).stem
    output_dir = Path("analysis_results_langgraph")
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / f"{pdf_name}_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== 论文分析报告 (LangGraph版本) ===\n\n")
        f.write(f"论文标题: {analysis.get('title', '未知')}\n")
        f.write(f"作者: {analysis.get('authors', '未知')}\n\n")
        f.write("=== 分析结果 ===\n")
        f.write(json.dumps(analysis, ensure_ascii=False, indent=2))
        f.write("\n\n=== 最终摘要 ===\n")
        f.write(summary)
        f.write("\n\n=== 最终评估 ===\n")
        f.write(json.dumps(evaluation, ensure_ascii=False, indent=2))

    print(f"💾 分析报告已保存至: {report_path}")
    return {"final_result": {"analysis": analysis, "summary": summary, "evaluation": evaluation}}


# --- 3. 定义决策逻辑 (决定下一步走向哪个节点) ---

def should_continue(state: AgentState) -> Literal["improve_summary", "save_results", "end"]:
    """决策函数：根据评估结果和迭代次数决定下一步"""
    evaluation = state["evaluation"]
    iterations = state["iterations"]
    max_iterations = state["max_iterations"]

    if evaluation.get("meets_goals", "").lower() == "是":
        print("✅ 决策: 目标已满足，准备保存结果。")
        return "save_results"
    elif iterations < max_iterations:
        print(f"🔄 决策: 目标未满足，但未达最大迭代次数({iterations}/{max_iterations})，继续改进。")
        return "improve_summary"
    else:
        print(f"⚠️ 决策: 已达到最大迭代次数({max_iterations})，结束流程。")
        return "end"


# --- 4. 构建和编译图 ---

def build_graph():
    """构建LangGraph工作流图"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("parse_pdf", parse_pdf_node)
    workflow.add_node("analyze_paper", analyze_paper_node)
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("evaluate_summary", evaluate_summary_node)
    workflow.add_node("improve_summary", improve_summary_node)
    workflow.add_node("save_results", save_results_node)

    # 设置入口点
    workflow.set_entry_point("parse_pdf")

    # 添加线性边
    workflow.add_edge("parse_pdf", "analyze_paper")
    workflow.add_edge("analyze_paper", "generate_summary")
    workflow.add_edge("generate_summary", "evaluate_summary")

    # 添加条件边：从评估节点到决策
    workflow.add_conditional_edges(
        "evaluate_summary",
        should_continue,
        {
            "improve_summary": "improve_summary",
            "save_results": "save_results",
            "end": END
        }
    )

    # 添加循环边：从改进节点回到评估节点
    workflow.add_edge("improve_summary", "evaluate_summary")

    # 添加结束边
    workflow.add_edge("save_results", END)

    # 使用内存检查点来保存状态（可选，但对于持久化和调试很有用）
    memory = MemorySaver()

    # 编译图
    app = workflow.compile(checkpointer=memory)
    return app


# --- 5. 封装成主类 ---

class LangGraphPaperAnalyzer:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.app = build_graph()
        # 可选：可视化图的结构
        self.app.get_graph().print_ascii()

    def analyze(self, pdf_path: str, goals: List[str]) -> Dict[str, Any]:
        """启动分析流程"""
        print(f"\n🚀 启动 LangGraph 论文分析器...")
        print("=" * 60)

        initial_state = {
            "pdf_path": pdf_path,
            "paper_text": "",
            "goals": goals,
            "analysis": {},
            "current_summary": "",
            "evaluation": {},
            "iterations": 0,
            "max_iterations": self.max_iterations,
            "final_result": None,
            "error_message": None
        }

        # 使用 thread_id 来跟踪特定的对话/运行
        config = {"configurable": {"thread_id": "paper-analysis-1"}}

        # 运行图直到结束
        final_state = self.app.invoke(initial_state, config=config)

        if final_state.get("error_message"):
            print(f"\n❌ 流程出错: {final_state['error_message']}")
            return {"success": False, "error": final_state["error_message"]}

        if final_state.get("final_result"):
            print("\n✅ 分析流程成功完成！")
            return {"success": True, "result": final_state["final_result"]}
        else:
            print("\n⚠️ 流程结束，但未达到目标。")
            return {"success": False, "result": final_state, "message": "未在最大迭代次数内达成目标"}


# --- 6. 使用示例 ---
if __name__ == "__main__":
    # 创建分析器
    analyzer = LangGraphPaperAnalyzer(max_iterations=3)

    # 设定目标
    goals = [
        "简洁明了",
        "突出研究贡献",
        "适合非专业读者理解",
        "包含关键发现",
        "不超过200字"
    ]

    # 分析PDF论文
    pdf_file_path = "基于关系驱动多模态嵌入塑形的图像描述生成.pdf"  # 替换为你的PDF文件路径

    if os.path.exists(pdf_file_path):
        result = analyzer.analyze(pdf_file_path, goals)

        if result["success"]:
            print("\n" + "=" * 60)
            print("📊 最终分析结果:")
            print("=" * 60)
            final_data = result["result"]
            print(f"\n📝 最终摘要:\n{final_data['summary']}")

            eval_data = final_data['evaluation']
            print(f"\n📈 最终评估:")
            print(f"   满足目标: {eval_data.get('meets_goals', '未知')}")
            print(f"   准确度: {eval_data.get('accuracy_score', 'N/A')}/10")
            print(f"   清晰度: {eval_data.get('clarity_score', 'N/A')}/10")
        else:
            print(f"\n❌ 分析失败或未完成: {result.get('error', result.get('message'))}")
    else:
        print(f"⚠️ PDF文件不存在: {pdf_file_path}")