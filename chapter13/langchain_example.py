from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from typing import Dict, List
from datetime import datetime
from enum import Enum

from init_client import init_llm

llm = init_llm(
    temperature=0.1
)


class InteractionMode(Enum):
    """人机交互模式"""
    AI_GUIDED = "ai_guided"  # AI主导，人类监督
    HUMAN_GUIDED = "human_guided"  # 人类主导，AI辅助
    COLLABORATIVE = "collaborative"  # 平等协作


class ConfidenceLevel(Enum):
    """AI置信度级别"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


# 医疗知识库（模拟）
class MedicalKnowledgeBase:
    def __init__(self):
        self.diseases = {
            "肺炎": {
                "symptoms": ["咳嗽", "发热", "胸痛", "呼吸困难"],
                "exams": ["胸部X光", "血常规", "C反应蛋白"],
                "treatments": ["抗生素治疗", "对症支持"],
                "confidence_factors": {
                    "咳嗽+发热": 0.7,
                    "X光斑片影": 0.9,
                    "白细胞升高": 0.8
                }
            },
            "心肌梗死": {
                "symptoms": ["胸痛", "呼吸困难", "出汗", "恶心"],
                "exams": ["心电图", "心肌酶谱", "冠脉造影"],
                "treatments": ["急诊PCI", "溶栓治疗", "抗凝治疗"],
                "confidence_factors": {
                    "剧烈胸痛": 0.8,
                    "心电图ST段抬高": 0.95,
                    "心肌酶升高": 0.9
                }
            }
        }

        self.learning_history = []  # 存储人工纠正的学习记录

    def analyze_symptoms(self, symptoms: List[str]) -> Dict:
        """AI分析症状"""
        results = []

        for disease, info in self.diseases.items():
            match_count = sum(1 for symptom in symptoms if symptom in info["symptoms"])
            total_symptoms = len(info["symptoms"])
            match_ratio = match_count / total_symptoms

            # 基于症状匹配计算初始置信度
            confidence = match_ratio * 0.7

            # 检查是否有高置信度因子
            for factor, factor_confidence in info["confidence_factors"].items():
                if any(keyword in " ".join(symptoms) for keyword in factor.split("+")):
                    confidence = max(confidence, factor_confidence)

            if confidence > ConfidenceLevel.LOW.value:
                results.append({
                    "disease": disease,
                    "confidence": confidence,
                    "matched_symptoms": [s for s in symptoms if s in info["symptoms"]],
                    "recommended_exams": info["exams"],
                    "treatments": info["treatments"]
                })

        return sorted(results, key=lambda x: x["confidence"], reverse=True)

    def learn_from_feedback(self, case_id: str, ai_diagnosis: str, human_diagnosis: str,
                            confidence: float, feedback_notes: str):
        """从人工反馈中学习"""
        learning_record = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "ai_diagnosis": ai_diagnosis,
            "human_diagnosis": human_diagnosis,
            "ai_confidence": confidence,
            "feedback": feedback_notes,
            "correct": ai_diagnosis == human_diagnosis
        }

        self.learning_history.append(learning_record)

        # 模拟学习过程：调整置信度计算
        if not learning_record["correct"]:
            # 如果AI诊断错误，降低相似病例的置信度
            for disease in self.diseases:
                if disease == ai_diagnosis:
                    for factor in self.diseases[disease]["confidence_factors"]:
                        current_conf = self.diseases[disease]["confidence_factors"][factor]
                        self.diseases[disease]["confidence_factors"][factor] = max(0.3, current_conf - 0.1)


# 人机协同交互界面
class HumanAIInteractionInterface:
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase()
        self.current_session = None
        self.interaction_history = []

    def start_diagnostic_session(self, patient_info: Dict, mode: InteractionMode) -> Dict:
        """开始诊断会话"""
        session_id = f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.current_session = {
            "session_id": session_id,
            "patient_info": patient_info,
            "mode": mode,
            "start_time": datetime.now(),
            "symptoms": [],
            "ai_suggestions": [],
            "human_feedback": [],
            "final_diagnosis": None,
            "confidence_evolution": []
        }

        return {
            "session_id": session_id,
            "message": f"诊断会话已开始，模式：{mode.value}",
            "next_step": "请输入患者症状"
        }

    def add_symptoms(self, symptoms: List[str]) -> Dict:
        """添加症状并获取AI分析"""
        if not self.current_session:
            return {"error": "未激活的诊断会话"}

        self.current_session["symptoms"].extend(symptoms)

        # AI分析症状
        ai_analysis = self.knowledge_base.analyze_symptoms(self.current_session["symptoms"])

        # 记录AI建议
        self.current_session["ai_suggestions"] = ai_analysis

        # 计算整体置信度
        overall_confidence = ai_analysis[0]["confidence"] if ai_analysis else 0.0
        self.current_session["confidence_evolution"].append({
            "timestamp": datetime.now().isoformat(),
            "symptoms": symptoms.copy(),
            "ai_suggestions": ai_analysis.copy(),
            "confidence": overall_confidence
        })

        # 根据交互模式决定下一步
        if self.current_session["mode"] == InteractionMode.AI_GUIDED:
            if overall_confidence >= ConfidenceLevel.HIGH.value:
                return {
                    "ai_analysis": ai_analysis,
                    "confidence": overall_confidence,
                    "recommendation": "AI置信度高，建议进行相关检查",
                    "human_action_needed": "确认检查项目或提供额外信息"
                }
            else:
                return {
                    "ai_analysis": ai_analysis,
                    "confidence": overall_confidence,
                    "recommendation": "AI置信度不足，需要人工指导",
                    "human_action_needed": "请提供诊断方向或排除某些疾病"
                }

        elif self.current_session["mode"] == InteractionMode.HUMAN_GUIDED:
            return {
                "ai_analysis": ai_analysis,
                "confidence": overall_confidence,
                "message": "AI分析完成，等待人工决策",
                "human_action_needed": "请选择诊断方向或要求AI进一步分析"
            }

        else:  # COLLABORATIVE
            return {
                "ai_analysis": ai_analysis,
                "confidence": overall_confidence,
                "message": "人机协作模式，请共同讨论诊断方案",
                "collaboration_points": [
                    "AI建议的疾病可能性",
                    "需要补充的检查",
                    "诊断的不确定因素"
                ]
            }

    def provide_human_feedback(self, feedback: Dict) -> Dict:
        """提供人工反馈，AI据此调整分析"""
        if not self.current_session:
            return {"error": "未激活的诊断会话"}

        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        }
        self.current_session["human_feedback"].append(feedback_record)

        # 根据人工反馈调整AI分析
        if "confirm_disease" in feedback:
            confirmed_disease = feedback["confirm_disease"]
            # AI学习这次确认
            if self.current_session["ai_suggestions"]:
                top_suggestion = self.current_session["ai_suggestions"][0]
                self.knowledge_base.learn_from_feedback(
                    self.current_session["session_id"],
                    top_suggestion["disease"],
                    confirmed_disease,
                    top_suggestion["confidence"],
                    feedback.get("notes", "")
                )

            self.current_session["final_diagnosis"] = confirmed_disease

            return {
                "message": f"诊断已确认为：{confirmed_disease}",
                "ai_learning": "AI已从此次反馈中学习",
                "treatment_suggestions": self.knowledge_base.diseases.get(confirmed_disease, {}).get("treatments", [])
            }

        elif "exclude_disease" in feedback:
            excluded = feedback["exclude_disease"]
            # 重新分析，排除指定疾病
            filtered_suggestions = [
                s for s in self.current_session["ai_suggestions"]
                if s["disease"] != excluded
            ]

            self.current_session["ai_suggestions"] = filtered_suggestions

            return {
                "message": f"已排除{excluded}，重新分析结果",
                "updated_suggestions": filtered_suggestions,
                "next_step": "请确认新的诊断方向"
            }

        elif "request_more_info" in feedback:
            info_type = feedback["request_more_info"]

            if info_type == "differential_diagnosis":
                # 提供鉴别诊断
                return {
                    "differential_diagnosis": [
                        s["disease"] for s in self.current_session["ai_suggestions"][:3]
                    ],
                    "comparison": self._compare_diseases(self.current_session["ai_suggestions"][:3]),
                    "human_action_needed": "请选择最可能的诊断"
                }

            elif info_type == "exam_recommendations":
                # 提供检查建议
                all_exams = set()
                for suggestion in self.current_session["ai_suggestions"]:
                    all_exams.update(suggestion["recommended_exams"])

                return {
                    "recommended_exams": list(all_exams),
                    "priority": self._prioritize_exams(self.current_session["ai_suggestions"]),
                    "human_action_needed": "请选择要进行的检查"
                }

        return {"message": "反馈已记录，AI正在调整分析"}

    def _compare_diseases(self, suggestions: List[Dict]) -> Dict:
        """比较不同疾病的特征"""
        comparison = {}
        for suggestion in suggestions:
            disease = suggestion["disease"]
            disease_info = self.knowledge_base.diseases.get(disease, {})
            comparison[disease] = {
                "key_symptoms": disease_info.get("symptoms", [])[:3],
                "definitive_exams": disease_info.get("exams", [])[:2],
                "confidence": suggestion["confidence"]
            }
        return comparison

    def _prioritize_exams(self, suggestions: List[Dict]) -> List[Dict]:
        """根据诊断可能性优先级排序检查"""
        exam_scores = {}

        for suggestion in suggestions:
            weight = suggestion["confidence"]
            for exam in suggestion["recommended_exams"]:
                exam_scores[exam] = exam_scores.get(exam, 0) + weight

        prioritized = sorted(exam_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"exam": exam, "score": score} for exam, score in prioritized]

    def end_session(self, final_diagnosis: str, summary: str) -> Dict:
        """结束诊断会话"""
        if not self.current_session:
            return {"error": "未激活的诊断会话"}

        session_summary = {
            "session_id": self.current_session["session_id"],
            "duration": str(datetime.now() - self.current_session["start_time"]),
            "final_diagnosis": final_diagnosis,
            "ai_confidence_evolution": self.current_session["confidence_evolution"],
            "human_feedback_count": len(self.current_session["human_feedback"]),
            "summary": summary,
            "learning_outcome": "AI已从本次人机协作中学习"
        }

        self.interaction_history.append(session_summary)
        self.current_session = None

        return session_summary


# LangChain工具（用于更复杂的分析）
@tool
def analyze_medical_report(report_text: str) -> Dict:
    """深度分析医疗报告"""
    # 模拟深度分析
    analysis = {
        "key_findings": ["发现异常密度影", "边界不清", "大小约3cm"],
        "impressions": ["不能排除恶性肿瘤", "建议进一步检查"],
        "confidence": 0.65,
        "recommendations": ["增强CT", "活检"]
    }
    return analysis


@tool
def get_treatment_guidelines(diagnosis: str, patient_profile: Dict) -> Dict:
    """获取个性化治疗指南"""
    guidelines = {
        "standard_treatment": ["药物治疗", "定期随访"],
        "alternative_options": ["手术治疗", "放射治疗"],
        "contraindications": ["患者过敏史", "合并症"],
        "lifestyle_recommendations": ["饮食调整", "适量运动"]
    }
    return guidelines


# 创建医疗代理
medical_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个医疗诊断AI助手，与医生协作进行诊断。请记住：

    1. 你是医生的助手，不是决策者
    2. 明确表达你的置信度
    3. 主动指出不确定性
    4. 从医生的反馈中学习
    5. 提供鉴别诊断和检查建议

    在人机协作中，保持专业、谦逊和学习的态度。"""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

medical_agent = create_openai_tools_agent(llm, [
    analyze_medical_report,
    get_treatment_guidelines
], medical_prompt)

medical_executor = AgentExecutor(
    agent=medical_agent,
    tools=[analyze_medical_report, get_treatment_guidelines],
    verbose=True
)


# 人机协同医疗诊断系统
class CollaborativeMedicalDiagnosisSystem:
    def __init__(self):
        self.interface = HumanAIInteractionInterface()
        self.agent_executor = medical_executor

    def demonstrate_collaborative_diagnosis(self):
        """演示完整的人机协同诊断流程"""
        print("=" * 60)
        print("人机协同医疗诊断系统演示")
        print("=" * 60)

        # 步骤1：开始诊断会话
        print("\n步骤1：开始诊断会话")
        patient_info = {
            "patient_id": "P001",
            "age": 45,
            "gender": "男",
            "medical_history": ["高血压", "吸烟史"]
        }

        session = self.interface.start_diagnostic_session(
            patient_info,
            InteractionMode.COLLABORATIVE
        )
        print(f"会话ID: {session['session_id']}")
        print(f"交互模式: {session['message']}")

        # 步骤2：输入初步症状
        print("\n步骤2：输入初步症状")
        initial_symptoms = ["胸痛", "呼吸困难"]
        result = self.interface.add_symptoms(initial_symptoms)

        print("AI分析结果:")
        for suggestion in result["ai_analysis"]:
            print(f"  - {suggestion['disease']}: 置信度 {suggestion['confidence']:.2f}")
            print(f"    匹配症状: {suggestion['matched_symptoms']}")

        print(f"\n协作要点: {result['collaboration_points']}")

        # 步骤3：人工反馈 - 请求更多信息
        print("\n步骤3：医生请求鉴别诊断")
        feedback = {
            "request_more_info": "differential_diagnosis"
        }
        more_info = self.interface.provide_human_feedback(feedback)

        print("\n鉴别诊断比较:")
        for disease, info in more_info["comparison"].items():
            print(f"\n{disease}:")
            print(f"  关键症状: {info['key_symptoms']}")
            print(f"  确诊检查: {info['definitive_exams']}")
            print(f"  AI置信度: {info['confidence']:.2f}")

        # 步骤4：医生提供额外症状
        print("\n步骤4：医生补充症状信息")
        additional_symptoms = ["出汗", "恶心"]
        result = self.interface.add_symptoms(additional_symptoms)

        print("\n更新后的AI分析:")
        for suggestion in result["ai_analysis"]:
            print(f"  - {suggestion['disease']}: 置信度 {suggestion['confidence']:.2f}")

        # 步骤5：医生确认诊断
        print("\n步骤5：医生确认诊断")
        final_feedback = {
            "confirm_disease": "心肌梗死",
            "notes": "患者症状典型，心电图支持诊断"
        }
        confirmation = self.interface.provide_human_feedback(final_feedback)

        print(f"\n{confirmation['message']}")
        print(f"治疗建议: {confirmation['treatment_suggestions']}")
        print(f"AI学习状态: {confirmation['ai_learning']}")

        # 步骤6：使用LangChain代理进行深度分析
        print("\n步骤6：使用AI代理进行深度分析")
        deep_analysis = self.agent_executor.invoke({
            "input": "患者确诊为心肌梗死，45岁男性，有高血压病史，请提供个性化治疗建议",
            "patient_profile": patient_info
        })

        print("深度分析结果:")
        print(deep_analysis["output"])

        # 步骤7：结束会话
        print("\n步骤7：结束诊断会话")
        summary = self.interface.end_session(
            "心肌梗死",
            "通过人机协作，快速准确诊断。AI初步分析，医生确认，AI学习改进。"
        )

        print(f"\n会话总结:")
        print(f"  诊断: {summary['final_diagnosis']}")
        print(f"  时长: {summary['duration']}")
        print(f"  人工反馈次数: {summary['human_feedback_count']}")
        print(f"  学习成果: {summary['learning_outcome']}")

        # 步骤8：展示AI学习效果
        print("\n步骤8：展示AI学习效果")
        print("AI学习历史记录:")
        for record in self.interface.knowledge_base.learning_history[-1:]:
            print(f"  案例: {record['case_id']}")
            print(f"  AI诊断: {record['ai_diagnosis']}")
            print(f"  人工诊断: {record['human_diagnosis']}")
            print(f"  正确性: {'✓' if record['correct'] else '✗'}")
            print(f"  反馈: {record['feedback']}")


# 运行演示
if __name__ == "__main__":
    system = CollaborativeMedicalDiagnosisSystem()
    system.demonstrate_collaborative_diagnosis()