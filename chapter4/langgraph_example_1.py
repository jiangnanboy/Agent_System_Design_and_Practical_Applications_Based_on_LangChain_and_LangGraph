from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 初始化模型
from init_client import init_llm

llm = init_llm(0.1)

# 定义状态结构
class GraphState(TypedDict):
    """定义图中节点之间共享的状态"""
    task: str  # 原始任务描述
    report: str  # 当前生成的报告
    feedback: str  # 评审反馈
    iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数
    is_perfect: bool  # 报告是否完美


# 定义生产者节点
def producer_node(state: GraphState):
    """生产者节点：根据任务和反馈生成或优化报告"""
    print(f"\n>>> 阶段1：生成/优化报告 (迭代 {state['iteration']})...")

    # 生产者提示模板
    producer_template = """
    你是一名数据分析师，需要根据任务要求和反馈生成或优化数据分析报告。

    任务要求：
    {task}

    反馈：
    {feedback}

    请根据以上要求生成数据分析报告：
    """

    producer_prompt = PromptTemplate(
        template=producer_template,
        input_variables=["task", "feedback"]
    )

    producer_chain = producer_prompt | llm | StrOutputParser()

    # 生成报告
    report = producer_chain.invoke({
        "task": state["task"],
        "feedback": state["feedback"]
    })

    print(f"\n--- 生成的报告 (v{state['iteration']}) ---\n{report}")

    # 更新状态
    return {
        **state,
        "report": report,
        "iteration": state["iteration"] + 1
    }


# 定义评审者节点
def reviewer_node(state: GraphState):
    """评审者节点：评估报告质量并提供反馈"""
    print("\n>>> 阶段2：评估报告质量...")

    # 评审者提示模板
    reviewer_template = """
    你是一名资深数据分析专家，负责评估数据分析报告的质量。

    原始任务要求：
    {task}

    待评估报告：
    {report}

    请评估报告是否满足以下标准：
    1. 数据准确性
    2. 分析深度
    3. 洞察价值
    4. 结构清晰度
    5. 语言专业性

    如果报告完美无缺，请回复"REPORT_IS_PERFECT"。
    否则，请提供具体改进建议，以项目符号形式列出：
    """

    reviewer_prompt = PromptTemplate(
        template=reviewer_template,
        input_variables=["task", "report"]
    )

    reviewer_chain = reviewer_prompt | llm | StrOutputParser()

    # 评估报告
    review = reviewer_chain.invoke({
        "task": state["task"],
        "report": state["report"]
    })

    # 检查报告是否完美
    is_perfect = "REPORT_IS_PERFECT" in review

    if is_perfect:
        print("\n--- 评估结果 ---\n报告质量满意，无需进一步改进。")
    else:
        print(f"\n--- 评估结果 ---\n{review}")

    # 更新状态
    return {
        **state,
        "feedback": f"请根据以下评估结果优化报告：\n{review}",
        "is_perfect": is_perfect
    }


# 定义条件边函数
def should_continue(state: GraphState):
    """决定是否继续迭代"""
    # 如果报告完美或达到最大迭代次数，则结束
    if state["is_perfect"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    # 否则继续生产者节点
    return "producer"


# 创建工作流图
def create_reflection_graph():
    """创建反思模式的工作流图"""
    # 初始化状态图
    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("producer", producer_node)
    workflow.add_node("reviewer", reviewer_node)

    # 设置入口点
    workflow.set_entry_point("producer")

    # 添加边
    workflow.add_edge("producer", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "producer": "producer",  # 如果继续，回到生产者
            "end": END  # 如果结束，终止流程
        }
    )

    # 设置内存保存器（用于状态持久化）
    memory = MemorySaver()

    # 编译图
    app = workflow.compile(checkpointer=memory)

    return app


# 主函数
def generate_data_analysis_report_with_langgraph():
    """使用LangGraph实现反思模式生成数据分析报告"""

    # 定义核心任务
    task_prompt = """
    你是一名数据分析师，需要基于以下销售数据生成一份简洁而全面的分析报告：

    数据：
    - 产品A：Q1销售额120万，Q2销售额150万，Q3销售额180万，Q4销售额210万
    - 产品B：Q1销售额80万，Q2销售额75万，Q3销售额90万，Q4销售额85万
    - 产品C：Q1销售额200万，Q2销售额220万，Q3销售额195万，Q4销售额240万

    报告应包括：
    1. 整体销售趋势分析
    2. 各产品表现对比
    3. 季度增长/下降率
    4. 关键洞察和建议
    """

    # 创建工作流图
    app = create_reflection_graph()
    # 打印图的结构（可选，非常直观！）
    try:
        print("--- 图结构 ---")
        app.get_graph().print_ascii()
        print("\n" + "=" * 20 + "\n")
    except Exception as e:
        print(f"无法打印图结构: {e}")

    # 初始状态
    initial_state = {
        "task": task_prompt,
        "report": "",
        "feedback": "请生成初始报告。",
        "iteration": 1,
        "max_iterations": 3,
        "is_perfect": False
    }

    # 运行工作流
    print(f"\n{'=' * 30} 开始反思模式工作流 {'=' * 30}")

    # 使用线程ID和配置运行图
    config = {"configurable": {"thread_id": "1"}}
    final_state = app.invoke(initial_state, config)

    print(f"\n{'=' * 35} 最终结果 {'=' * 35}")
    print("\n经过反思过程优化的最终数据分析报告：\n")
    print(final_state["report"])


if __name__ == "__main__":
    generate_data_analysis_report_with_langgraph()