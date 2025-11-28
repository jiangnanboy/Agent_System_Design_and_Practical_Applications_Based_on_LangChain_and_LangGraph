# -*- coding: utf-8 -*-
"""
一个完整的合同审核AI Agent评估与监控案例

此脚本演示了以下核心概念，并使用 LangChain LCEL (链式调用) 实现：
1. 使用Pydantic定义结构化数据模型（任务合约、分析报告）。
2. 使用LangChain和DeepSeek构建一个合同审核Agent。
3. 使用LLM-as-a-Judge的方法来评估Agent的输出质量。
4. 实施一个性能监控器来跟踪关键指标并分析趋势。

"""

import json
import time
from datetime import datetime
from typing import List, Dict, Literal, Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field



# --- 1. 数据模型定义 ---
from init_client import init_llm
llm = init_llm(temperature=0.1)

class ContractClause(BaseModel):
    """表示单个合同条款及其分析结果"""
    clause_id: str = Field(description="条款唯一标识符")
    text: str = Field(description="条款原文")
    type: str = Field(description="条款类型，如赔偿、责任限制、保密等")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(description="风险等级")
    concerns: List[str] = Field(description="识别出的风险点")
    suggestions: List[str] = Field(description="修改建议")


class ContractAnalysisReport(BaseModel):
    """合同审核Agent生成的分析报告"""
    contract_id: str = Field(description="合同唯一标识符")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    overall_risk_score: float = Field(description="整体风险评分(0-10)")
    critical_clauses: List[ContractClause] = Field(description="高风险条款列表")
    missing_clauses: List[str] = Field(description="缺失的重要条款类型")
    compliance_issues: List[str] = Field(description="合规性问题")
    recommendations: List[str] = Field(description="整体建议")
    processing_time_seconds: float = Field(description="处理耗时(秒)")


class ContractReviewContract(BaseModel):
    """定义合同审核任务的'高级合约'"""
    task_id: str = Field(description="任务唯一标识符")
    contract_text: str = Field(description="待审核的合同全文")
    review_standards: Dict[str, str] = Field(description="审核标准，定义各类条款的判断依据")
    compliance_frameworks: List[str] = Field(description="需遵循的合规框架，如GDPR、SOX等")
    company_policies: List[str] = Field(description="公司特定政策要求")
    acceptance_criteria: List[str] = Field(description="验收标准")
    quality_metrics: Dict[str, float] = Field(description="质量指标阈值")


# --- 2. 核心Agent实现 (使用LCEL) ---

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

        # 使用 LCEL 创建链: prompt -> llm -> parser
        self.chain = self.review_prompt | llm | self.parser

    def review_contract(self, contract: ContractReviewContract) -> ContractAnalysisReport:
        """审核合同并生成分析报告"""
        print(f"\n[Agent] 开始审核合同任务: {contract.task_id}...")
        start_time = time.time()

        # 准备链的输入
        inputs = {
            "contract_text": contract.contract_text,
            "review_standards": contract.review_standards,
            "compliance_frameworks": contract.compliance_frameworks,
            "company_policies": contract.company_policies
        }

        try:
            # 调用链，它会自动解析输出为 Pydantic 对象
            report = self.chain.invoke(inputs)
            report.contract_id = contract.task_id
        except Exception as e:
            print(f"[Agent] 解析LLM输出失败，生成默认错误报告: {e}")
            report = ContractAnalysisReport(
                contract_id=contract.task_id,
                overall_risk_score=0.0,
                critical_clauses=[],
                missing_clauses=[],
                compliance_issues=[f"解析错误: {str(e)}"],
                recommendations=["请重新提交合同进行审核"],
                processing_time_seconds=time.time() - start_time
            )

        report.processing_time_seconds = time.time() - start_time
        print(f"[Agent] 审核完成，耗时: {report.processing_time_seconds:.2f}秒。")
        return report


# --- 3. 评估框架实现 (使用LCEL) ---

class ContractReviewEvaluator:
    """使用LLM-as-a-Judge评估合同审核Agent的性能 (使用LCEL实现)"""

    def __init__(self):
        self.evaluation_prompt = PromptTemplate(
            template="""
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
            input_variables=["contract_text", "agent_report", "expert_report"]
        )

        # 使用 LCEL 创建链: prompt -> llm
        self.chain = self.evaluation_prompt | llm

    def evaluate_review(self, contract_text: str, agent_report: ContractAnalysisReport, expert_report: dict) -> dict:
        """评估合同审核报告的质量"""
        print("\n[Evaluator] 开始评估AI Agent的审核质量...")

        agent_report_str = f"""
        整体风险评分: {agent_report.overall_risk_score}/10
        高风险条款: {len(agent_report.critical_clauses)}条
        合规性问题: {', '.join(agent_report.compliance_issues)}
        建议: {', '.join(agent_report.recommendations)}
        """

        expert_report_str = f"""
        整体风险评分: {expert_report.get('overall_risk_score', 'N/A')}/10
        高风险条款: {len(expert_report.get('critical_clauses', []))}条
        合规性问题: {', '.join(expert_report.get('compliance_issues', []))}
        建议: {', '.join(expert_report.get('recommendations', []))}
        """

        # 准备链的输入
        inputs = {
            "contract_text": contract_text,
            "agent_report": agent_report_str,
            "expert_report": expert_report_str
        }

        # 调用链
        response = self.chain.invoke(inputs)
        response = response.content.replace('```json', '').replace('```', '').strip()

        try:
            result = json.loads(response)
            print(f"[Evaluator] 评估完成，总体得分: {result.get('overall_score', 'N/A')}/10")
            return result
        except json.JSONDecodeError:
            print("[Evaluator] 评估结果JSON解析失败。")
            return {
                "overall_score": 5,
                "error": "评估结果解析失败",
                "raw_response": response
            }


# --- 4. 性能监控实现 ---

class ContractReviewMonitor:
    """合同审核Agent性能监控器"""

    def __init__(self, metrics_file: str = "contract_review_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics_history: List[Dict[str, Any]] = []
        try:
            with open(self.metrics_file, 'r') as f:
                self.metrics_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.metrics_history = []

    def record_review_metrics(self, task_id: str, contract_length: int, report: ContractAnalysisReport,
                              evaluation: Dict[str, Any]):
        """记录单次审核的指标"""
        print(f"\n[Monitor] 记录任务 {task_id} 的性能指标...")

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "contract_length_chars": contract_length,
            "processing_time_seconds": report.processing_time_seconds,
            "overall_risk_score": report.overall_risk_score,
            "critical_clauses_count": len(report.critical_clauses),
            "compliance_issues_count": len(report.compliance_issues),
            "evaluation_score": evaluation.get("overall_score", 0),
            "risk_identification_score": evaluation.get("risk_identification_score", 0),
            "missed_risks_count": len(evaluation.get("missed_risks", [])),
            "false_positives_count": len(evaluation.get("false_positives", []))
        }

        self.metrics_history.append(metrics)

        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"[Monitor] 指标已记录到 {self.metrics_file}")
        return metrics

    def get_performance_summary(self, last_n_records: int = 10) -> Dict[str, Any]:
        """获取最近n次审核的性能摘要"""
        if not self.metrics_history:
            return {"message": "暂无性能数据"}

        recent_metrics = self.metrics_history[-last_n_records:]

        avg_processing_time = sum(m["processing_time_seconds"] for m in recent_metrics) / len(recent_metrics)
        avg_evaluation_score = sum(m["evaluation_score"] for m in recent_metrics) / len(recent_metrics)
        avg_risk_identification = sum(m["risk_identification_score"] for m in recent_metrics) / len(recent_metrics)
        avg_missed_risks = sum(m["missed_risks_count"] for m in recent_metrics) / len(recent_metrics)

        return {
            "evaluations_count": len(recent_metrics),
            "average_processing_time_seconds": round(avg_processing_time, 2),
            "average_evaluation_score": round(avg_evaluation_score, 2),
            "average_risk_identification_score": round(avg_risk_identification, 2),
            "average_missed_risks_count": round(avg_missed_risks, 2),
            "performance_trend": self._calculate_trend("evaluation_score", recent_metrics)
        }

    def _calculate_trend(self, metric_name: str, metrics: List[Dict[str, Any]]) -> str:
        """计算指标趋势"""
        if len(metrics) < 2:
            return "insufficient_data"

        x = list(range(len(metrics)))
        y = [m[metric_name] for m in metrics]

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        if (n * sum_x2 - sum_x ** 2) == 0: return "stable"

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"


# --- 5. 主执行流程 ---

if __name__ == "__main__":
    # --- 示例数据准备 ---
    sample_contract_text = """
    软件开发服务协议

    甲方：ABC科技有限公司
    乙方：XYZ软件开发公司

    第一条 服务内容
    乙方为甲方开发一套客户关系管理系统，包括需求分析、系统设计、编码实现、测试和部署。

    第二条 知识产权
    乙方保留所有开发过程中产生的代码和文档的知识产权。甲方获得软件的永久使用权，但不得进行修改或二次开发。

    第三条 保密义务
    双方应对在合作过程中获知的对方商业秘密保密，保密期限为合同终止后一年。

    第四条 责任限制
    乙方对因软件缺陷导致的甲方损失不承担任何责任，最高赔偿金额不超过合同总金额的10%。
    """

    # 模拟专家报告，作为评估的黄金标准
    expert_reference_report = {
        "overall_risk_score": 8.5,
        "critical_clauses": [
            {"text": "乙方保留所有开发过程中产生的代码和文档的知识产权", "type": "知识产权", "risk_level": "high",
             "concerns": ["甲方应拥有定制开发软件的全部知识产权"], "suggestions": ["修改为甲方拥有全部知识产权"]},
            {"text": "保密期限为合同终止后一年", "type": "保密义务", "risk_level": "medium", "concerns": ["保密期限过短，通常应为3-5年"],
             "suggestions": ["延长保密期限至3年"]}
        ],
        "compliance_issues": ["未包含GDPR要求的数据处理条款"],
        "recommendations": ["修改知识产权条款", "延长保密期限", "添加数据处理条款"]
    }

    # --- 步骤 1: 定义任务合约 ---
    task_contract = ContractReviewContract(
        task_id="contract_2023_001",
        contract_text=sample_contract_text,
        review_standards={
            "知识产权": "甲方应拥有定制开发软件的全部知识产权",
            "保密义务": "保密期限通常应为合同终止后3-5年",
            "责任限制": "责任限制条款应合理，不应完全免除乙方责任"
        },
        compliance_frameworks=["GDPR"],
        company_policies=["所有软件项目必须包含源代码交付"],
        acceptance_criteria=["识别所有高风险条款", "提供具体修改建议"],
        quality_metrics={"min_risk_detection_rate": 0.95}
    )
    print("=" * 50)
    print("步骤 1: 任务合约已定义")
    print(f"任务ID: {task_contract.task_id}")

    # --- 步骤 2: Agent执行合同审核 ---
    agent = ContractReviewAgent()
    agent_report = agent.review_contract(task_contract)
    print("\n" + "=" * 50)
    print("步骤 2: Agent审核报告")
    print(f"整体风险评分: {agent_report.overall_risk_score}/10")
    print(f"高风险条款数量: {len(agent_report.critical_clauses)}")
    print(f"合规性问题: {', '.join(agent_report.compliance_issues)}")
    print(f"建议: {', '.join(agent_report.recommendations)}")

    # --- 步骤 3: 评估Agent的审核质量 ---
    evaluator = ContractReviewEvaluator()
    evaluation_result = evaluator.evaluate_review(
        task_contract.contract_text,
        agent_report,
        expert_reference_report
    )
    print("\n" + "=" * 50)
    print("步骤 3: LLM-as-a-Judge 评估结果")
    print(json.dumps(evaluation_result, indent=2, ensure_ascii=False))

    # --- 步骤 4: 记录和监控性能 ---
    monitor = ContractReviewMonitor()
    monitor.record_review_metrics(
        task_contract.task_id,
        len(task_contract.contract_text),
        agent_report,
        evaluation_result
    )

    print("\n" + "=" * 50)
    print("步骤 4: 性能监控摘要")
    performance_summary = monitor.get_performance_summary()
    print(json.dumps(performance_summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 50)
    print("工作流程演示完成！")