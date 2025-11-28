from typing import TypedDict, Annotated, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import Tool
from langgraph.graph import StateGraph, END

# --- 步骤 1: 定义模拟知识库和工具 ---
from init_client import init_llm

legal_knowledge_base = {
    "合同违约": "合同违约是指合同当事人一方或双方不履行合同义务或履行合同义务不符合约定的行为。违约方应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
    "根本违约": "根本违约是指一方的违约行为严重影响了另一方订立合同时所期望的经济利益。在此情况下，守约方有权解除合同，并要求违约方承担违约责任。",
    "合同变更": "合同变更需经合同双方当事人协商一致。如果一方主张合同内容发生变更，该方负有举证责任，需提供证据（如书面协议、邮件等）证明双方已就变更内容达成合意。",
    "损害赔偿": "违约损害赔偿额应当相当于因违约所造成的损失，包括合同履行后可以获得的利益；但是，不得超过违约一方订立合同时预见到或者应当预见到的因违约可能造成的损失。",
    "合同变更举证": "主张合同变更的一方负有举证责任。如果无法提供证据证明变更内容，则法律上推定为合同未变更。",
    "预期损失": "损失赔偿额不得超过违约一方订立合同时预见到或者应当预见到的因违约可能造成的损失。"
}


def query_legal_database(query: str) -> str:
    """根据关键词查询内部法律知识库。"""
    for key in legal_knowledge_base:
        if key in query:
            return f"在知识库中找到关于 '{key}' 的信息：\n{legal_knowledge_base[key]}"
    return f"在知识库中未找到与 '{query}' 相关的信息。请尝试其他关键词。"


legal_research_tool = Tool(
    name="LegalKB",
    description="一个包含合同法条文的内部数据库。使用关键词进行查询，例如 '合同违约'。",
    func=query_legal_database,
)

# 2. 初始化模型
llm = init_llm(temperature=0.1)


# --- 步骤 2: 定义图的状态 ---
# 确保状态中包含所有必要的键
class AgentState(TypedDict):
    initial_query: str
    search_query: str
    search_results: Annotated[List[str], "The list of search results"]
    final_answer: str
    loop_count: int


# --- 步骤 3: 定义图的节点 ---

def generate_query(state: AgentState):
    """节点1: 分析初始问题，生成第一个搜索查询"""
    print("--- 节点: 生成查询 ---")
    prompt = ChatPromptTemplate.from_template("""
    你是一个法律专家。根据案情摘要，生成一个用于查询法律知识库的关键词。
    案情摘要: {initial_query}
    只返回关键词，不要任何解释。
    """)
    chain = prompt | llm
    ai_message = chain.invoke({"initial_query": state["initial_query"]})
    search_query_text = ai_message.content.strip()
    return {"search_query": search_query_text, "loop_count": 1}


def web_research(state: AgentState):
    """节点2: 执行搜索并存储结果"""
    print(f"--- 节点: 执行研究 (查询: {state['search_query']}) ---")
    result = legal_research_tool.run(state["search_query"])
    return {"search_results": state["search_results"] + [result], "loop_count": state["loop_count"] + 1}


# reflection 节点
def reflect_and_decide(state: AgentState):
    """节点3: 反思研究结果，并决定下一步行动"""
    print("--- 节点: 反思与决策 ---")
    all_results = "\n\n".join(state["search_results"])
    last_query = state.get("search_query", "")

    # 安全网：防止无限循环
    if state.get("loop_count", 0) > 5:
        print("达到最大循环次数，强制结束。")
        return {"decision": "finalize"}

    prompt = ChatPromptTemplate.from_template("""
    你是一个法律专家。你正在分析一个案件。
    案情摘要: {initial_query}
    你已经进行了 {loop_count} 次研究。上一次的查询是 "{last_query}"。
    到目前为止，你获得了以下研究结果:
    {all_results}

    现在，请判断：
    - 如果信息已足够撰写一份完整的法律备忘录，请只返回 "FINALIZE"。
    - 如果信息不足，请返回一个与之前不同的、新的搜索关键词。

    你的回答必须严格遵循以上要求。
    """)
    chain = prompt | llm
    ai_message = chain.invoke({
        "initial_query": state["initial_query"],
        "all_results": all_results,
        "loop_count": state["loop_count"],
        "last_query": last_query
    })

    response_text = ai_message.content.strip()
    print(f"LLM 响应: '{response_text}'")

    # 决策逻辑
    if response_text == "FINALIZE":
        return {"decision": "finalize"}
    elif response_text and response_text != last_query:
        print(f"决定继续研究，新查询为: {response_text}")
        return {"decision": "continue", "search_query": response_text}
    else:
        print(f"LLM 未提供有效的新查询（可能是空字符串或重复查询），强制结束。")
        return {"decision": "finalize"}


def finalize_answer(state: AgentState):
    """节点4: 生成最终的法律备忘录"""
    print("--- 节点: 生成最终答案 ---")
    all_results = "\n\n".join(state["search_results"])
    prompt = ChatPromptTemplate.from_template("""
    你是一个顶级的法律分析AI助手。请根据以下案情摘要和你的研究结果，生成一份结构清晰、逻辑严谨的法律分析备忘录。

    案情摘要: {initial_query}

    研究结果:
    {all_results}

    备忘录应包含以下部分：
    1. 案件概述
    2. 核心法律问题
    3. 法律分析与论点构建
    4. 结论与建议
    """)
    chain = prompt | llm
    ai_message = chain.invoke({"initial_query": state["initial_query"], "all_results": all_results})
    final_answer_text = ai_message.content
    return {"final_answer": final_answer_text}


# --- 步骤 4: 构建图 ---
workflow = StateGraph(AgentState)

workflow.add_node("generate_query", generate_query)
workflow.add_node("web_research", web_research)
workflow.add_node("reflect_and_decide", reflect_and_decide)  # 节点名也改了
workflow.add_node("finalize_answer", finalize_answer)

workflow.set_entry_point("generate_query")

workflow.add_edge("generate_query", "web_research")
workflow.add_edge("web_research", "reflect_and_decide")

workflow.add_conditional_edges(
    "reflect_and_decide",
    lambda state: state["decision"],
    {
        "continue": "web_research",
        "finalize": "finalize_answer"
    }
)

workflow.add_edge("finalize_answer", END)

app = workflow.compile()

# 可选：可视化图的结构
app.get_graph().print_ascii()

# --- 步骤 5: 可视化图 ---
try:
    print("\n--- 工作流图结构 ---")
    mermaid_code = app.get_graph().draw_mermaid()
    print(mermaid_code)
except Exception as e:
    print(f"无法可视化图: {e}")

# --- 步骤 6: 运行图 ---
case_summary = """
2023年1月，一家初创公司"创新科技"与一家软件开发工作室"卓越代码"签订了一份合同。
合同规定，"卓越代码"应在2023年6月30日前，为"创新科技"开发一款具有特定功能的移动应用，总费用为10万美元。
合同明确列出五项核心功能。截止到2023年7月15日，"卓越代码"交付的应用仅实现了三项核心功能，且存在多处bug，导致无法正常使用。
"创新科技"拒绝支付尾款，并要求"卓越代码"赔偿因其延迟交付和质量问题造成的商业损失。
"卓越代码"辩称，延迟是由于"创新科技"在开发中期提出了额外的需求变更，导致了工作量的增加。
"""

initial_state = {
    "initial_query": case_summary,
    "search_query": "",
    "search_results": [],
    "final_answer": "",
    "loop_count": 0
}

print("\n--- 开始法律分析 ---")
final_state = app.invoke(initial_state, config={"recursion_limit": 15})

print("\n--- 分析完成 ---")
print("最终法律备忘录:")
print(final_state["final_answer"])