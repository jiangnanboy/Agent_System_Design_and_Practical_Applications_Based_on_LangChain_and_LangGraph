# -*- coding: utf-8 -*-
"""
使用 LangGraph 和 DeepSeek 实现的合同审核AI Agent评估与监控案例

工作流步骤:
1. Agent Node: 执行合同审核。
2. Evaluator Node: 评估Agent的输出质量。
3. Conditional Edge: 根据评估分数决定是重试还是继续。
4. Increment Retry Node: 增加重试计数器。
5. Monitor Node: 记录最终性能指标。
"""
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Literal, TypedDict, Annotated, Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from init_client import init_llm

llm = init_llm(temperature=0.1)

# --- 1. 数据模型与状态定义 ---

class ContractClause(BaseModel):
    clause_id: str = Field(description="条款唯一标识符")
    text: str = Field(description="条款原文")
    type: str = Field(description="条款类型")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(description="风险等级")
    concerns: List[str] = Field(description="识别出的风险点")
    suggestions: List[str] = Field(description="修改建议")


class ContractAnalysisReport(BaseModel):
    contract_id: str = Field(description="合同唯一标识符")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    overall_risk_score: float = Field(description="整体风险评分(0-10)")
    critical_clauses: List[ContractClause] = Field(description="高风险条款列表")
    missing_clauses: List[str] = Field(description="缺失的重要条款类型")
    compliance_issues: List[str] = Field(description="合规性问题")
    recommendations: List[str] = Field(description="整体建议")
    processing_time_seconds: float = Field(description="处理耗时(秒)")


class GraphState(TypedDict):
    task_id: str
    contract_text: str
    review_standards: Dict[str, str]
    compliance_frameworks: List[str]
    company_policies: List[str]

    agent_report: Optional[ContractAnalysisReport]
    evaluation_result: Optional[Dict[str, Any]]

    retry_count: Annotated[int, "当前重试次数"]
    max_retries: Annotated[int, "最大重试次数"]
    final_decision: Annotated[str, "最终决策: 'accepted' 或 'failed_after_retries'"]


# --- 2. 节点逻辑实现 ---

def run_agent_review(state: GraphState):
    """节点1: 运行合同审核Agent"""
    print(f"\n--- [Node 1: Agent Review] ---")
    print(f"正在执行任务: {state['task_id']} (尝试次数: {state['retry_count'] + 1})")

    # 为了简化，我们在这里直接使用之前的 Agent 类
    class ContractReviewAgent:
        def __init__(self):
            self.parser = PydanticOutputParser(pydantic_object=ContractAnalysisReport)
            self.review_prompt = PromptTemplate(
                template="""
            作为一名专业的法律顾问，请仔细审核以下合同，并识别其中的风险条款。

            合同内容:
            {contract_text}

            审核标准:
            {review_standards}

            需要检查的合规框架:
            {compliance_frameworks}

            公司政策:
            {company_policies}

            请按照以下格式提供分析报告:
            {format_instructions}
            """,
                input_variables=["contract_text", "review_standards", "compliance_frameworks", "company_policies"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            self.chain = self.review_prompt | llm | self.parser

        def review(self, contract_details: dict) -> ContractAnalysisReport:
            start_time = time.time()
            try:
                report = self.chain.invoke(contract_details)
                report.contract_id = contract_details['task_id']
                report.processing_time_seconds = time.time() - start_time
                return report
            except Exception as e:
                return ContractAnalysisReport(
                    contract_id=contract_details['task_id'], overall_risk_score=0.0, critical_clauses=[],
                    missing_clauses=[], compliance_issues=[f"错误: {str(e)}"], recommendations=["请重新提交"],
                    processing_time_seconds=time.time() - start_time
                )

    agent = ContractReviewAgent()
    contract_details = {k: state[k] for k in
                        ["task_id", "contract_text", "review_standards", "compliance_frameworks", "company_policies"]}
    report = agent.review(contract_details)
    print(f"Agent 审核完成，风险评分: {report.overall_risk_score}/10")
    return {"agent_report": report}


def run_evaluation(state: GraphState):
    """节点2: 评估Agent的输出质量"""
    print("\n--- [Node 2: Evaluator] ---")
    print("正在评估Agent报告质量...")

    class ContractReviewEvaluator:
        def __init__(self):
            self.evaluation_prompt = PromptTemplate(template="""
            作为一名资深法律专家，请评估以下AI Agent对合同的审核质量。

            原始合同:
            {contract_text}

            AI Agent生成的审核报告:
            {agent_report}

            专家参考报告(黄金标准):
            {expert_report}

            请从以下维度评估AI Agent的审核质量(1-10分):
            1. 风险识别准确性: 是否准确识别了所有风险条款
            2. 风险评估准确性: 是否正确评估了风险的严重程度
            3. 建议实用性: 提出的修改建议是否具体可行
            4. 合规性检查: 是否正确识别了合规性问题
            5. 报告完整性: 报告是否包含了所有必要信息

            请以JSON格式返回评估结果:
            {{
                "overall_score": 总体质量评分(1-10),
                "risk_identification_score": 风险识别准确性评分(1-10),
                "risk_assessment_score": 风险评估准确性评分(1-10),
                "suggestion_usefulness_score": 建议实用性评分(1-10),
                "compliance_check_score": 合规性检查评分(1-10),
                "report_completeness_score": 报告完整性评分(1-10),
                "missed_risks": ["AI Agent遗漏的风险1", "AI Agent遗漏的风险2"],
                "false_positives": ["AI Agent误报的风险1", "AI Agent误报的风险2"],
                "improvement_suggestions": ["改进建议1", "改进建议2"]
            }}
            """,
            input_variables=["contract_text", "agent_report", "expert_report"])
            self.chain = self.evaluation_prompt | llm

        def evaluate(self, contract_text: str, agent_report: ContractAnalysisReport, expert_report: dict) -> dict:
            agent_report_str = f"..."
            expert_report_str = f"..."
            inputs = {"contract_text": contract_text, "agent_report": agent_report_str,
                      "expert_report": expert_report_str}
            response = self.chain.invoke(inputs)
            response = response.content.replace('```json', '').replace('```', '').strip()
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {"overall_score": 5, "error": "评估结果解析失败"}

    evaluator = ContractReviewEvaluator()
    expert_report = {"overall_risk_score": 8.5, "critical_clauses": [...], "compliance_issues": [...]}  # 模拟专家报告
    result = evaluator.evaluate(state["contract_text"], state["agent_report"], expert_report)
    print(f"评估完成，总体得分: {result.get('overall_score', 'N/A')}/10")
    return {"evaluation_result": result}


# --- 新增：用于增加重试次数的节点 ---
def increment_retry(state: GraphState):
    """节点3: 增加重试计数器"""
    print("\n--- [Node 3: Increment Retry] ---")
    print(f"重试计数器从 {state['retry_count']} 增加到 {state['retry_count'] + 1}")
    return {"retry_count": state['retry_count'] + 1}


def run_monitoring(state: GraphState):
    """节点4: 记录最终性能指标"""
    print("\n--- [Node 4: Monitor] ---")
    print("正在记录最终性能指标...")

    class ContractReviewMonitor:
        def __init__(self, metrics_file: str = "langgraph_metrics.json"):
            self.metrics_file = metrics_file
            self.metrics_history = []
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self.metrics_history = []

        def record(self, state: GraphState):
            report, evaluation = state['agent_report'], state['evaluation_result']
            metrics = {
                "timestamp": datetime.now().isoformat(), "task_id": state['task_id'],
                "final_decision": state['final_decision'],
                "retry_count": state['retry_count'], "processing_time_seconds": report.processing_time_seconds,
                "final_evaluation_score": evaluation.get("overall_score", 0),
            }
            self.metrics_history.append(metrics)
            with open(self.metrics_file, "w") as f: json.dump(self.metrics_history, f, indent=2)
            print(f"指标已记录到 {self.metrics_file}")

    monitor = ContractReviewMonitor()
    monitor.record(state)
    return state


# --- 3. 条件路由逻辑 (更新后) ---

def should_retry(state: GraphState) -> str:
    """条件边: 根据评估结果决定下一步"""
    print("\n--- [Conditional Edge: Should Retry?] ---")
    evaluation = state['evaluation_result']
    if not evaluation: return "monitor"

    score = evaluation.get("overall_score", 10)
    print(f"当前评估分数: {score}, 最大重试次数: {state['max_retries']}, 当前重试次数: {state['retry_count']}")

    if score < 7 and state['retry_count'] < state['max_retries']:
        print("分数低于7且未达到最大重试次数，决定重试。")
        return "increment_retry"  # 路由到新的节点
    else:
        decision = "accepted" if score >= 7 else "failed_after_retries"
        print(f"决定进入监控阶段。最终决策: {decision}。")
        return "monitor"


# --- 4. 构建和编译图 (更新后) ---

def create_contract_review_graph():
    memory = MemorySaver()
    workflow = StateGraph(GraphState)

    # 添加所有节点
    workflow.add_node("agent", run_agent_review)
    workflow.add_node("evaluator", run_evaluation)
    workflow.add_node("increment_retry", increment_retry)  # 新增节点
    workflow.add_node("monitor", run_monitoring)

    workflow.set_entry_point("agent")

    # 添加边
    workflow.add_edge("agent", "evaluator")

    # 更新条件边
    workflow.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "increment_retry": "increment_retry",  # 新增路由
            "monitor": "monitor"
        }
    )

    # 新增：从 increment_retry 节点返回到 agent 节点
    workflow.add_edge("increment_retry", "agent")

    workflow.add_edge("monitor", END)

    app = workflow.compile(checkpointer=memory)
    return app


# --- 5. 主执行流程 ---

if __name__ == "__main__":

    sample_contract_text = """
    软件开发服务协议
    甲方：ABC科技有限公司
    乙方：XYZ软件开发公司
    第二条 知识产权
    乙方保留所有开发过程中产生的代码和文档的知识产权。甲方获得软件的永久使用权，但不得进行修改或二次开发。
    第三条 保密义务
    双方应对在合作过程中获知的对方商业秘密保密，保密期限为合同终止后一年。
    """

    app = create_contract_review_graph()

    initial_state = {
        "task_id": "contract_langgraph_001",
        "contract_text": sample_contract_text,
        "review_standards": {"知识产权": "甲方应拥有全部知识产权"},
        "compliance_frameworks": ["GDPR"],
        "company_policies": ["必须包含源代码交付"],
        "agent_report": None,
        "evaluation_result": None,
        "retry_count": 0,
        "max_retries": 2,
        "final_decision": ""
    }

    print("=" * 50)
    print("开始执行 LangGraph 合同审核工作流...")

    final_state = app.invoke(initial_state, config={"configurable": {"thread_id": "contract-review-001"}})

    print("\n" + "=" * 50)
    print("工作流执行完毕！")
    print(f"最终决策: {final_state['final_decision']}")
    print(f"最终Agent报告风险评分: {final_state['agent_report'].overall_risk_score}/10")
    print(f"最终评估得分: {final_state['evaluation_result'].get('overall_score', 'N/A')}/10")
    print(f"总重试次数: {final_state['retry_count']}")
