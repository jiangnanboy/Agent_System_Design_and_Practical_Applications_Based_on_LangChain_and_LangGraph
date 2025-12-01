from typing import Optional, List

from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from init_client import init_llm

# --- 1. 初始化核心模型 ---
llm = init_llm(temperature=0.2)

# --- 2. 定义“认知状态” ---
class SecurityAnalysisState(TypedDict):
    target: str
    config_or_code: Optional[str]
    # messages 列表记录了整个探索过程的“思考轨迹”
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    scan_results: Optional[str]
    vulnerability_analysis: Optional[str]
    threat_assessment: Optional[str]
    final_report: Optional[str]


# --- 3. 定义“工具”函数（模拟外部知识获取） ---

def local_search(query: str) -> str:
    """模拟本地搜索，提供上下文知识。"""
    if "CVE" in query or "vulnerability" in query.lower():
        return "常见漏洞信息：CVE-2023-1234 (Apache RCE), CVE-2023-5678 (OpenSSL DoS)"
    elif "apache" in query.lower():
        return "Apache常见安全问题：默认配置泄露、旧版本漏洞、不安全的HTTP方法。"
    else:
        return "未找到相关信息。"


def local_vulnerability_db(component: str, version: str = "") -> str:
    """查询本地漏洞数据库。"""
    db = {
        "apache": {"2.4.49": "存在路径遍历漏洞 (CVE-2021-41773) 和 mod_proxy 漏洞 (CVE-2021-40438)。"},
        "openssh": {"8.2p1": "存在scp客户端信息泄露漏洞 (CVE-2021-28041)。"}
    }
    if component.lower() in db:
        return db[component.lower()].get(version, f"找到 {component} 的相关漏洞信息。")
    return f"未找到 {component} 的漏洞信息。"


# --- 4. 定义“认知节点” ---

def initial_scan_node(state: SecurityAnalysisState):
    """节点1：信息收集与初步扫描"""
    print("--- 🔍 执行节点1: 初始扫描与信息收集 ---")
    # 模拟扫描结果，以体现探索的起点
    simulated_scan_result = f"对 {state['target']} 的扫描结果：\n- 开放端口: 22(SSH), 80(HTTP)\n- Web服务器: Apache/2.4.49\n- SSH版本: OpenSSH_8.2p1"

    # 将发现作为消息添加到状态中，形成“思考链”
    new_message = AIMessage(content=f"初步扫描完成。发现:\n{simulated_scan_result}")

    return {"scan_results": simulated_scan_result, "messages": [new_message]}


def vulnerability_analysis_node(state: SecurityAnalysisState):
    """节点2：深度漏洞分析（核心探索与发现）"""
    print("\n--- 🧠 执行节点2: 深度漏洞分析 ---")

    # 构建一个复杂的提示，引导 LLM 进行主动推理和探索
    analysis_prompt = f"""
    作为一名资深的网络安全专家，请基于以下信息进行深度分析。
    你的任务不仅仅是匹配已知漏洞，更是要**推理和发现潜在的、组合式的安全风险**。

    **扫描结果:**
    {state['scan_results']}

    **本地知识库查询:**
    - Apache 2.4.49: {local_vulnerability_db('apache', '2.4.49')}
    - OpenSSH 8.2p1: {local_vulnerability_db('openssh', '8.2p1')}

    **分析要求:**
    1.  **关联分析**: 结合扫描结果和本地知识，评估这些服务同时运行可能带来的复合风险。
    2.  **推理未知**: 基于这些版本，推测可能存在的、尚未被广泛记录的配置错误或逻辑漏洞。
    3.  **生成新知**: 提出一个你的独特见解，例如一个非标准的攻击向量或一个容易被忽视的安全隐患。

    请提供结构化的分析报告。
    """

    # 使用 llm 进行深度分析，这是从“数据”到“洞察”的关键一步
    analysis = llm.invoke(analysis_prompt)

    new_message = AIMessage(content=f"深度漏洞分析完成。分析结果:\n{analysis.content}")

    return {"vulnerability_analysis": analysis.content, "messages": [new_message]}


def threat_assessment_node(state: SecurityAnalysisState):
    """节点3：威胁评估与优先级排序"""
    print("\n--- ⚖️ 执行节点3: 威胁评估 ---")

    assessment_prompt = f"""
    作为一名安全主管，请对以下漏洞分析结果进行威胁评估。

    **漏洞分析报告:**
    {state['vulnerability_analysis']}

    **评估标准:**
    - 可利用性: 攻击者利用该漏洞的难易程度。
    - 影响范围: 成功攻击后可能对业务造成的损害。
    - 修复成本: 修复该漏洞所需的时间和资源。

    请为每个发现的漏洞评定一个威胁等级（高、中、低），并解释原因。最后，给出一个修复优先级的建议。
    """

    assessment = llm.invoke(assessment_prompt)

    new_message = AIMessage(content=f"威胁评估完成。评估结果:\n{assessment.content}")

    return {"threat_assessment": assessment.content, "messages": [new_message]}


def report_generation_node(state: SecurityAnalysisState):
    """节点4：综合报告生成"""
    print("\n--- 📄 执行节点4: 生成最终报告 ---")

    report_prompt = f"""
    请将以下所有分析过程和结果，整合成一份专业的安全评估报告。

    **目标:** {state['target']}
    **扫描结果:** {state['scan_results']}
    **漏洞分析:** {state['vulnerability_analysis']}
    **威胁评估:** {state['threat_assessment']}

    报告应包含以下部分：
    1. 执行摘要
    2. 详细发现
    3. 风险评估与优先级
    4. 可操作的修复建议
    5. 结论
    """

    final_report = llm.invoke(report_prompt)

    new_message = AIMessage(content=f"最终报告已生成。")

    return {"final_report": final_report.content, "messages": [new_message]}

# --- 5. 构建并编译“认知工作流图” ---

def create_security_analysis_workflow():
    workflow = StateGraph(SecurityAnalysisState)

    # 添加节点
    workflow.add_node("initial_scan", initial_scan_node)
    workflow.add_node("vulnerability_analysis", vulnerability_analysis_node)
    workflow.add_node("threat_assessment", threat_assessment_node)
    workflow.add_node("report_generation", report_generation_node)

    # 定义入口点
    workflow.set_entry_point("initial_scan")

    # 定义边（流程）-> 这定义了认知的线性推进路径
    workflow.add_edge("initial_scan", "vulnerability_analysis")
    workflow.add_edge("vulnerability_analysis", "threat_assessment")
    workflow.add_edge("threat_assessment", "report_generation")
    workflow.add_edge("report_generation", END)

    # 编译图，添加内存以保存状态
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# --- 6. 执行探索与发现流程 ---

def run_security_analysis(target: str, config_or_code: Optional[str] = None):
    """启动并运行整个安全分析工作流。"""

    # 创建工作流实例
    app = create_security_analysis_workflow()

    # 初始化状态
    initial_state = {
        "target": target,
        "config_or_code": config_or_code,
        "messages": [HumanMessage(content=f"开始对 {target} 进行全面的安全分析。")],
        "scan_results": None,
        "vulnerability_analysis": None,
        "threat_assessment": None,
        "final_report": None
    }

    # 执行工作流，thread_id 用于在内存中跟踪特定会话
    config = {"configurable": {"thread_id": "security-analysis-001"}}
    final_state = app.invoke(initial_state, config)

    # 打印完整的思考轨迹
    print("\n\n===== 完整的认知探索轨迹 =====")
    for message in final_state["messages"]:
        print(f"{message.type}: {message.content[:100]}...")

    # 返回最终发现
    return final_state["final_report"]

if __name__ == "__main__":
    # 模拟网站
    target_system = "internal-web-server.example.com"
    final_security_report = run_security_analysis(target_system)

    print("\n\n===============================")
    print("   最终安全分析报告")
    print("===============================")
    print(final_security_report)

