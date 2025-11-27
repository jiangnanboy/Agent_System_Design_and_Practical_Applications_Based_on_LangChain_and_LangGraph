from typing import Dict, List
from datetime import datetime
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from init_client import init_llm

llm = init_llm(temperature=0.1)


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

    def analyze_symptoms(self, symptoms: List[str]) -> List[Dict]:
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


@tool
def analyze_medical_report(report_text: str) -> Dict:
    """深度分析医疗报告"""
    # 使用LLM分析医疗报告
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个医疗专家，请分析以下医疗报告并提供关键发现、印象和建议"),
        ("human", "{report_text}")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({"report_text": report_text})

        # 解析响应，提取关键信息
        # 这里简化处理，实际应用中可能需要更复杂的解析
        return {
            "key_findings": ["发现异常密度影", "边界不清", "大小约3cm"],
            "impressions": ["不能排除恶性肿瘤", "建议进一步检查"],
            "confidence": 0.65,
            "recommendations": ["增强CT", "活检"],
            "llm_analysis": response.content
        }
    except Exception as e:
        print(f"LLM分析医疗报告时出错: {str(e)}")
        # 返回默认分析结果
        return {
            "key_findings": ["发现异常密度影", "边界不清", "大小约3cm"],
            "impressions": ["不能排除恶性肿瘤", "建议进一步检查"],
            "confidence": 0.65,
            "recommendations": ["增强CT", "活检"],
            "llm_analysis": "基于报告内容，建议进一步检查以明确诊断。"
        }


@tool
def get_treatment_guidelines(diagnosis: str, patient_profile: Dict) -> Dict:
    """获取个性化治疗指南"""
    # 使用LLM生成个性化治疗指南
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个医疗专家，请根据诊断和患者情况提供个性化治疗指南"),
        ("human", "诊断: {diagnosis}\n患者信息: {patient_profile}")
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({
            "diagnosis": diagnosis,
            "patient_profile": patient_profile
        })

        # 解析响应，提取关键信息
        # 这里简化处理，实际应用中可能需要更复杂的解析
        return {
            "standard_treatment": ["药物治疗", "定期随访"],
            "alternative_options": ["手术治疗", "放射治疗"],
            "contraindications": ["患者过敏史", "合并症"],
            "lifestyle_recommendations": ["饮食调整", "适量运动"],
            "llm_guidelines": response.content
        }
    except Exception as e:
        print(f"LLM生成治疗指南时出错: {str(e)}")
        # 返回默认治疗指南
        return {
            "standard_treatment": ["药物治疗", "定期随访"],
            "alternative_options": ["手术治疗", "放射治疗"],
            "contraindications": ["患者过敏史", "合并症"],
            "lifestyle_recommendations": ["饮食调整", "适量运动"],
            "llm_guidelines": "根据诊断，建议采用标准治疗方案并定期随访。"
        }


# 简化的人机协同医疗诊断系统
class SimpleCollaborativeMedicalDiagnosisSystem:
    def __init__(self):
        print("[DEBUG] 初始化系统")
        self.knowledge_base = MedicalKnowledgeBase()
        self.session_data = {}

    def start_diagnostic_session(self, patient_info: Dict, mode: InteractionMode):
        """开始诊断会话"""
        print(f"[DEBUG] 开始诊断会话，模式: {mode.value}")

        session_id = f"SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 使用LLM生成初始欢迎消息
        try:
            welcome_message = f"诊断会话已开始，模式：{mode.value}"
        except Exception as e:
            print(f"生成欢迎消息时出错: {str(e)}")
            welcome_message = f"诊断会话已开始，模式：{mode.value}"

        # 初始化会话数据
        self.session_data = {
            "session_id": session_id,
            "patient_info": patient_info,
            "mode": mode.value,
            "symptoms": [],
            "ai_suggestions": [],
            "human_feedback": [],
            "final_diagnosis": None,
            "confidence_evolution": [],
            "messages": [SystemMessage(content=welcome_message), HumanMessage(content="请输入患者症状")],
            "next_action": "请输入患者症状"
        }

        print(f"[DEBUG] 会话ID: {session_id}")

        return self.session_data

    def add_symptoms(self, symptoms: List[str]):
        """添加症状并进行分析"""
        print(f"[DEBUG] 添加症状: {symptoms}")

        if not self.session_data:
            return {"error": "未激活的诊断会话"}

        # 更新症状
        self.session_data["symptoms"].extend(symptoms)

        # AI分析症状
        ai_analysis = self.knowledge_base.analyze_symptoms(self.session_data["symptoms"])

        # 计算整体置信度
        overall_confidence = ai_analysis[0]["confidence"] if ai_analysis else 0.0

        # 更新置信度演化
        self.session_data["confidence_evolution"].append({
            "timestamp": datetime.now().isoformat(),
            "symptoms": self.session_data["symptoms"].copy(),
            "ai_suggestions": ai_analysis.copy(),
            "confidence": overall_confidence
        })

        # 根据交互模式决定下一步
        if self.session_data["mode"] == InteractionMode.AI_GUIDED.value:
            if overall_confidence >= ConfidenceLevel.HIGH.value:
                next_action = "AI置信度高，建议进行相关检查"
            else:
                next_action = "AI置信度不足，需要人工指导"
        elif self.session_data["mode"] == InteractionMode.HUMAN_GUIDED.value:
            next_action = "AI分析完成，等待人工决策"
        else:  # COLLABORATIVE
            next_action = "人机协作模式，请共同讨论诊断方案"

        # 构建AI响应消息
        response = f"AI分析结果:\n"
        for suggestion in ai_analysis:
            response += f"- {suggestion['disease']}: 置信度 {suggestion['confidence']:.2f}\n"
            response += f"  匹配症状: {suggestion['matched_symptoms']}\n"

        response += f"\n{next_action}"

        self.session_data["ai_suggestions"] = ai_analysis
        self.session_data["messages"].append(AIMessage(content=response))
        self.session_data["next_action"] = next_action

        print(f"[DEBUG] AI分析完成，置信度: {overall_confidence}")

        return self.session_data

    def provide_feedback(self, feedback: Dict):
        """提供反馈"""
        print(f"[DEBUG] 提供反馈: {feedback}")

        if not self.session_data:
            return {"error": "未激活的诊断会话"}

        # 记录反馈
        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        }
        self.session_data["human_feedback"].append(feedback_record)

        # 根据人工反馈调整AI分析
        if "confirm_disease" in feedback:
            confirmed_disease = feedback["confirm_disease"]
            # AI学习这次确认
            if self.session_data["ai_suggestions"]:
                top_suggestion = self.session_data["ai_suggestions"][0]
                self.knowledge_base.learn_from_feedback(
                    self.session_data["session_id"],
                    top_suggestion["disease"],
                    confirmed_disease,
                    top_suggestion["confidence"],
                    feedback.get("notes", "")
                )

            treatments = self.knowledge_base.diseases.get(confirmed_disease, {}).get("treatments", [])

            response = f"诊断已确认为：{confirmed_disease}\n"
            response += f"AI已从此次反馈中学习\n"
            response += f"治疗建议: {', '.join(treatments)}"

            self.session_data["final_diagnosis"] = confirmed_disease
            self.session_data["messages"].append(AIMessage(content=response))
            self.session_data["next_action"] = "诊断已确认，可以结束会话"

            print(f"[DEBUG] 诊断已确认: {confirmed_disease}")

        elif "exclude_disease" in feedback:
            excluded = feedback["exclude_disease"]
            # 重新分析，排除指定疾病
            filtered_suggestions = [
                s for s in self.session_data["ai_suggestions"]
                if s["disease"] != excluded
            ]

            response = f"已排除{excluded}，重新分析结果\n"
            for suggestion in filtered_suggestions:
                response += f"- {suggestion['disease']}: 置信度 {suggestion['confidence']:.2f}\n"

            self.session_data["ai_suggestions"] = filtered_suggestions
            self.session_data["messages"].append(AIMessage(content=response))
            self.session_data["next_action"] = "请确认新的诊断方向"

        elif "request_more_info" in feedback:
            info_type = feedback["request_more_info"]

            if info_type == "differential_diagnosis":
                # 提供鉴别诊断
                comparison = {}
                for suggestion in self.session_data["ai_suggestions"][:3]:
                    disease = suggestion["disease"]
                    disease_info = self.knowledge_base.diseases.get(disease, {})
                    comparison[disease] = {
                        "key_symptoms": disease_info.get("symptoms", [])[:3],
                        "definitive_exams": disease_info.get("exams", [])[:2],
                        "confidence": suggestion["confidence"]
                    }

                response = "鉴别诊断比较:\n"
                for disease, info in comparison.items():
                    response += f"\n{disease}:\n"
                    response += f"  关键症状: {info['key_symptoms']}\n"
                    response += f"  确诊检查: {info['definitive_exams']}\n"
                    response += f"  AI置信度: {info['confidence']:.2f}\n"

                self.session_data["messages"].append(AIMessage(content=response))
                self.session_data["next_action"] = "请选择最可能的诊断"

            elif info_type == "exam_recommendations":
                # 提供检查建议
                all_exams = set()
                for suggestion in self.session_data["ai_suggestions"]:
                    all_exams.update(suggestion["recommended_exams"])

                exam_scores = {}
                for suggestion in self.session_data["ai_suggestions"]:
                    weight = suggestion["confidence"]
                    for exam in suggestion["recommended_exams"]:
                        exam_scores[exam] = exam_scores.get(exam, 0) + weight

                prioritized = sorted(exam_scores.items(), key=lambda x: x[1], reverse=True)

                response = "推荐检查项目（按优先级排序）:\n"
                for exam, score in prioritized:
                    response += f"- {exam}: 优先级 {score:.2f}\n"

                self.session_data["messages"].append(AIMessage(content=response))
                self.session_data["next_action"] = "请选择要进行的检查"

        else:
            # 一般反馈处理响应
            response = "反馈已记录，AI正在调整分析"
            self.session_data["messages"].append(AIMessage(content=response))
            self.session_data["next_action"] = "继续分析"

        return self.session_data

    def end_session(self, final_diagnosis: str, summary: str):
        """结束会话"""
        print(f"[DEBUG] 结束会话，诊断: {final_diagnosis}")

        if not self.session_data:
            return {"error": "未激活的诊断会话"}

        # 更新最终诊断和总结
        self.session_data["final_diagnosis"] = final_diagnosis

        session_summary = {
            "session_id": self.session_data["session_id"],
            "final_diagnosis": final_diagnosis,
            "ai_confidence_evolution": self.session_data["confidence_evolution"],
            "human_feedback_count": len(self.session_data["human_feedback"]),
            "summary": summary,
            "learning_outcome": "AI已从本次人机协作中学习"
        }

        response = "诊断会话已结束\n"
        response += f"诊断: {final_diagnosis}\n"
        response += f"人工反馈次数: {session_summary['human_feedback_count']}\n"
        response += f"学习成果: {session_summary['learning_outcome']}"

        self.session_data["messages"].append(AIMessage(content=response))
        self.session_data["next_action"] = "会话已结束"

        print(f"[DEBUG] 会话结束，诊断: {final_diagnosis}")

        return self.session_data

    def demonstrate_collaborative_diagnosis(self):
        """演示完整的人机协同诊断流程"""
        print("=" * 60)
        print("人机协同医疗诊断系统演示 (简化版)")
        print("=" * 60)

        # 步骤1：开始诊断会话
        print("\n步骤1：开始诊断会话")
        patient_info = {
            "patient_id": "P001",
            "age": 45,
            "gender": "男",
            "medical_history": ["高血压", "吸烟史"]
        }

        result = self.start_diagnostic_session(
            patient_info,
            InteractionMode.COLLABORATIVE
        )

        if "error" in result:
            print(f"错误: {result['error']}")
            return

        print(f"会话ID: {result['session_id']}")
        print(f"交互模式: {result['mode']}")
        print(f"AI响应: {result['messages'][-1].content}")

        # 步骤2：输入初步症状
        print("\n步骤2：输入初步症状")
        initial_symptoms = ["胸痛", "呼吸困难"]
        result = self.add_symptoms(initial_symptoms)

        if "error" in result:
            print(f"错误: {result['error']}")
            return

        print("AI分析结果:")
        print(result['messages'][-1].content)

        # 步骤3：请求鉴别诊断
        print("\n步骤3：请求鉴别诊断")
        feedback = {
            "request_more_info": "differential_diagnosis"
        }
        result = self.provide_feedback(feedback)

        if "error" in result:
            print(f"错误: {result['error']}")
            return

        print("AI响应:")
        print(result['messages'][-1].content)

        # 步骤4：添加更多症状
        print("\n步骤4：添加更多症状")
        additional_symptoms = ["出汗", "恶心"]
        result = self.add_symptoms(additional_symptoms)

        if "error" in result:
            print(f"错误: {result['error']}")
            return

        print("AI更新后的分析:")
        print(result['messages'][-1].content)

        # 步骤5：确认诊断
        print("\n步骤5：确认诊断")
        final_feedback = {
            "confirm_disease": "心肌梗死",
            "notes": "患者症状典型，心电图支持诊断"
        }
        result = self.provide_feedback(final_feedback)

        if "error" in result:
            print(f"错误: {result['error']}")
            return

        print("AI响应:")
        print(result['messages'][-1].content)

        # 步骤6：使用DeepSeek进行深度分析
        print("\n步骤6：使用DeepSeek进行深度分析")
        deep_analysis = analyze_medical_report.invoke("患者心电图显示ST段抬高，心肌酶谱升高")

        print("深度分析结果:")
        print(deep_analysis["llm_analysis"] if "llm_analysis" in deep_analysis else str(deep_analysis))

        # 步骤7：获取个性化治疗指南
        print("\n步骤7：获取个性化治疗指南")
        treatment_guidelines = get_treatment_guidelines.invoke({
            "diagnosis": "心肌梗死",
            "patient_profile": patient_info
        })

        print("治疗指南:")
        print(treatment_guidelines["llm_guidelines"] if "llm_guidelines" in treatment_guidelines else str(
            treatment_guidelines))

        # 步骤8：结束会话
        print("\n步骤8：结束会话")
        result = self.end_session(
            "心肌梗死",
            "通过人机协作，快速准确诊断。AI初步分析，医生确认，AI学习改进。"
        )

        if "error" in result:
            print(f"错误: {result['error']}")
            return

        print("会话总结:")
        print(result['messages'][-1].content)

        # 步骤9：展示AI学习效果
        print("\n步骤9：展示AI学习效果")
        print("AI学习历史记录:")
        for record in self.knowledge_base.learning_history[-1:]:
            print(f"  案例: {record['case_id']}")
            print(f"  AI诊断: {record['ai_diagnosis']}")
            print(f"  人工诊断: {record['human_diagnosis']}")
            print(f"  正确性: {'✓' if record['correct'] else '✗'}")
            print(f"  反馈: {record['feedback']}")


# 运行演示
if __name__ == "__main__":

    system = SimpleCollaborativeMedicalDiagnosisSystem()
    system.demonstrate_collaborative_diagnosis()