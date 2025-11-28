from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from init_client import init_llm

# --- 定义模拟知识库和工具---
legal_knowledge_base = {
    "合同违约": "合同违约是指合同当事人一方或双方不履行合同义务或履行合同义务不符合约定的行为。违约方应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
    "根本违约": "根本违约是指一方的违约行为严重影响了另一方订立合同时所期望的经济利益。在此情况下，守约方有权解除合同，并要求违约方承担违约责任。",
    "合同变更": "合同变更需经合同双方当事人协商一致。如果一方主张合同内容发生变更，该方负有举证责任，需提供证据（如书面协议、邮件等）证明双方已就变更内容达成合意。",
    "损害赔偿": "违约损害赔偿额应当相当于因违约所造成的损失，包括合同履行后可以获得的利益；但是，不得超过违约一方订立合同时预见到或者应当预见到的因违约可能造成的损失。"
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

# 初始化模型
llm = init_llm(temperature=0.1)

legal_analysis_prompt = PromptTemplate.from_template("""
你是一个顶级的法律分析AI助手。你必须严格遵循 ReAct (Reasoning and Acting) 格式来回答问题。

**可用工具:**
{tools}

**ReAct 格式要求:**
你必须严格按照以下格式进行思考和行动，使用英文关键词：
Thought: [你在此处进行推理，分析问题，决定下一步做什么]
Action: [你必须选择的工具名称，必须是以下之一: {tool_names}]
Action Input: [你传递给工具的输入]
Observation: [工具的输出结果]
... (这个 Thought/Action/Action Input/Observation 循环可以重复多次)
当你认为已经收集了足够的信息，可以给出最终答案时，请直接输出最终的法律备忘录。

**案情摘要:**
{input}

**思考过程:**
{agent_scratchpad}
""")

# 创建Agent
tools = [legal_research_tool]
agent = create_react_agent(llm, tools, legal_analysis_prompt)

# 在 AgentExecutor 中启用错误处理
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True
)

# 提供一个案例并运行Agent
case_summary = """
2023年1月，一家初创公司"创新科技"与一家软件开发工作室"卓越代码"签订了一份合同。
合同规定，"卓越代码"应在2023年6月30日前，为"创新科技"开发一款具有特定功能的移动应用，总费用为10万美元。
合同明确列出五项核心功能。截止到2023年7月15日，"卓越代码"交付的应用仅实现了三项核心功能，且存在多处bug，导致无法正常使用。
"创新科技"拒绝支付尾款，并要求"卓越代码"赔偿因其延迟交付和质量问题造成的商业损失。
"卓越代码"辩称，延迟是由于"创新科技"在开发中期提出了额外的需求变更，导致了工作量的增加。
"""

print("--- 开始法律分析 (使用内部知识库) ---")
result = agent_executor.invoke({"input": case_summary})
print("\n--- 分析完成 ---")
print(result['output'])