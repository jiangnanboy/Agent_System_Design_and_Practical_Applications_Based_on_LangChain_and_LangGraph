import time
import random
from typing import Dict, Any

from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from init_client import init_llm

from langchain_core.output_parsers import JsonOutputParser

# --- 1. 定义资源约束 ---
class ManufacturingConstraints:
    def __init__(self, time_limit: float, cost_budget: float, quality_threshold: float):
        """
        初始化制造约束条件

        参数:
            time_limit: 检测时间限制 (分钟)
            cost_budget: 检测成本预算 (美元)
            quality_threshold: 最低可接受的质量分数 (0-100)
        """
        self.time_limit = time_limit
        self.cost_budget = cost_budget
        self.quality_threshold = quality_threshold


# --- 2. 定义不同资源消耗的检测工具 ---
class InspectionMethods:
    @staticmethod
    def basic_visual_inspection(batch_id: str) -> Dict[str, Any]:
        """基础人工目检：快速、便宜、低精度"""
        time.sleep(1)  # 模拟1分钟的检测时间
        defects_found = random.randint(0, 5)
        quality_score = max(80, 100 - defects_found * 4)
        result = {
            "method_used": "basic_visual_inspection",
            "batch_id": batch_id,
            "defects_found": defects_found,
            "quality_score": quality_score,
            "time_taken": 1.0,
            "cost_incurred": 20.0
        }
        return result

    @staticmethod
    def standard_sensor_inspection(batch_id: str) -> Dict[str, Any]:
        """标准传感器检测：中等速度、中等成本、中等精度"""
        time.sleep(3)  # 模拟3分钟的检测时间
        defects_found = random.randint(0, 3)
        quality_score = max(90, 100 - defects_found * 3)
        result = {
            "method_used": "standard_sensor_inspection",
            "batch_id": batch_id,
            "defects_found": defects_found,
            "quality_score": quality_score,
            "time_taken": 3.0,
            "cost_incurred": 75.0
        }
        return result

    @staticmethod
    def precision_ai_inspection(batch_id: str) -> Dict[str, Any]:
        """高精度AI视觉检测：慢速、昂贵、高精度"""
        time.sleep(7)  # 模拟7分钟的检测时间
        defects_found = random.randint(0, 2)
        quality_score = max(98, 100 - defects_found * 1)
        result = {
            "method_used": "precision_ai_inspection",
            "batch_id": batch_id,
            "defects_found": defects_found,
            "quality_score": quality_score,
            "time_taken": 7.0,
            "cost_incurred": 250.0
        }
        return result


# --- 3. 资源感知优化器---
class ResourceAwareProductionOptimizer:
    def __init__(self):
        # 初始化LLM
        self.llm = init_llm(temperature=0.1)

        # 创建决策提示模板
        decision_prompt_template = """
        你是一个智能制造系统的决策核心。请根据以下生产订单的约束条件，选择最合适的质量检测方法。

        订单约束:
        - 时间限制: {time_limit} 分钟
        - 成本预算: ${cost_budget}
        - 质量要求: 最低质量分数 {quality_threshold}/100

        可选的检测方法及其规格:
        | 方法名 | 预估耗时 | 预估成本 | 预估质量分数范围 |
        |---|---|---|---|
        | basic_visual_inspection | 1.0 分钟 | $20 | 80-95 |
        | standard_sensor_inspection | 3.0 分钟 | $75 | 90-98 |
        | precision_ai_inspection | 7.0 分钟 | $250 | 98-100 |

        你的任务是选择一个**既能满足所有约束条件，又最具成本效益**的方法。
        如果没有方法能同时满足所有约束，请选择最接近要求的方法。

        请只返回一个JSON对象，格式如下，不要包含任何其他解释性文字：
        {{"method": "你选择的方法名"}}
        """

        self.decision_prompt = PromptTemplate(
            input_variables=["time_limit", "cost_budget", "quality_threshold"],
            template=decision_prompt_template
        )

        # --- 创建决策链 ---
        # 这个链的工作流程是：
        # 1. 接收一个字典作为输入
        # 2. PromptTemplate 使用字典中的值格式化提示词
        # 3. LLM 接收格式化后的提示词并生成文本响应
        # 4. JsonOutputParser 尝试将LLM的文本响应解析为Python字典
        self.decision_chain = self.decision_prompt | self.llm | JsonOutputParser()

    def select_inspection_method(self, constraints: ManufacturingConstraints) -> str:
        """根据约束条件选择检测方法"""
        try:
            decision_dict = self.decision_chain.invoke({
                "time_limit": constraints.time_limit,
                "cost_budget": constraints.cost_budget,
                "quality_threshold": constraints.quality_threshold
            })
            # JsonOutputParser已经帮我们解析好了，直接取值
            return decision_dict["method"]
        except Exception as e:
            # 如果LLM返回的不是有效JSON或发生其他错误，执行回退逻辑
            print(f"⚠️ LLM决策链解析失败: {e}，使用默认方法 'standard_sensor_inspection'")
            return "standard_sensor_inspection"

    def run_qc_process(self, batch_id: str, constraints: ManufacturingConstraints) -> Dict[str, Any]:
        """执行完整的资源感知质量控制流程"""
        print(f"🏭 开始为批次 '{batch_id}' 进行质量控制...")
        print(
            f"📋 约束条件: 时间限制={constraints.time_limit}分钟, 预算=${constraints.cost_budget}, 质量要求>={constraints.quality_threshold}\n")

        # 1. LLM决策选择方法
        chosen_method = self.select_inspection_method(constraints)
        print(f"✅ 初始决策: 使用 '{chosen_method}' 方法。\n")

        # 2. 执行选定的方法
        if chosen_method == "basic_visual_inspection":
            result = InspectionMethods.basic_visual_inspection(batch_id)
        elif chosen_method == "standard_sensor_inspection":
            result = InspectionMethods.standard_sensor_inspection(batch_id)
        else:  # precision_ai_inspection
            result = InspectionMethods.precision_ai_inspection(batch_id)

        # 3. 验证结果是否满足约束
        time_ok = result['time_taken'] <= constraints.time_limit
        cost_ok = result['cost_incurred'] <= constraints.cost_budget
        quality_ok = result['quality_score'] >= constraints.quality_threshold

        result['within_constraints'] = time_ok and cost_ok and quality_ok
        result['initial_choice'] = chosen_method

        # 4. 回退机制
        if not result['within_constraints']:
            print(f"⚠️ 警告: 初始选择 '{chosen_method}' 的结果不满足约束条件！")
            print(f"   - 耗时: {result['time_taken']} (限制: {constraints.time_limit})")
            print(f"   - 成本: ${result['cost_incurred']} (预算: ${constraints.cost_budget})")
            print(f"   - 质量: {result['quality_score']} (要求: {constraints.quality_threshold})")

            # 实施回退策略
            fallback_method = None
            if chosen_method == "precision_ai_inspection":
                fallback_method = "standard_sensor_inspection"
            elif chosen_method == "standard_sensor_inspection":
                fallback_method = "basic_visual_inspection"

            if fallback_method:
                print(f"🔄 正在回退到更经济的方法: '{fallback_method}'...")
                if fallback_method == "standard_sensor_inspection":
                    result = InspectionMethods.standard_sensor_inspection(batch_id)
                else:
                    result = InspectionMethods.basic_visual_inspection(batch_id)
                result['fallback_used'] = fallback_method
                result['within_constraints'] = (result['time_taken'] <= constraints.time_limit and
                                                result['cost_incurred'] <= constraints.cost_budget and
                                                result['quality_score'] >= constraints.quality_threshold)

        return result


# --- 4. 创建LangChain工具 ---
def create_langchain_tools(optimizer: ResourceAwareProductionOptimizer) -> list:
    """创建LangChain工具列表"""

    def run_quality_control(batch_id: str, time_limit: float, cost_budget: float, quality_threshold: float) -> str:
        """运行质量控制流程"""
        constraints = ManufacturingConstraints(time_limit, cost_budget, quality_threshold)
        result = optimizer.run_qc_process(batch_id, constraints)

        report = f"""
        批次ID: {result['batch_id']}
        最终使用方法: {result['method_used']}
        发现缺陷数: {result['defects_found']}
        最终质量分数: {result['quality_score']}/100
        实际耗时: {result['time_taken']} 分钟
        实际成本: ${result['cost_incurred']}
        是否满足约束: {'是' if result['within_constraints'] else '否'}
        """

        if 'fallback_used' in result:
            report += f"\n⚠️ 已从 '{result['initial_choice']}' 回退到 '{result['fallback_used']}'"

        return report

    return [
        Tool(
            name="QualityControl",
            description="根据时间、成本和质量约束运行质量控制流程。输入参数：batch_id (字符串), time_limit (浮点数), cost_budget (浮点数), quality_threshold (浮点数)。",
            func=run_quality_control
        )
    ]

# --- 5. 主程序与测试案例 ---
if __name__ == "__main__":
    # 初始化资源感知优化器
    optimizer = ResourceAwareProductionOptimizer()

    # 定义三个不同的生产订单场景
    scenarios = {
        "紧急订单": ManufacturingConstraints(time_limit=2.0, cost_budget=50.0, quality_threshold=85.0),
        "标准订单": ManufacturingConstraints(time_limit=5.0, cost_budget=100.0, quality_threshold=92.0),
        "高价值客户订单": ManufacturingConstraints(time_limit=10.0, cost_budget=300.0, quality_threshold=98.0)
    }

    # 直接运行优化器
    print("=" * 60)
    print("直接运行资源感知优化器")
    print("=" * 60)

    for name, constraints in scenarios.items():
        print(f"\n🚀 场景: {name}")
        print("-" * 40)

        batch_id = f"BATCH-{name.replace(' ', '_').upper()}"
        final_report = optimizer.run_qc_process(batch_id, constraints)

        print("\n--- 最终报告 ---")
        print(f"批次ID: {final_report['batch_id']}")
        print(f"最终使用方法: {final_report['method_used']}")
        if 'fallback_used' in final_report:
            print(f"⚠️ 已从 '{final_report['initial_choice']}' 回退到 '{final_report['fallback_used']}'")
        print(f"发现缺陷数: {final_report['defects_found']}")
        print(f"最终质量分数: {final_report['quality_score']}/100")
        print(f"实际耗时: {final_report['time_taken']} 分钟")
        print(f"实际成本: ${final_report['cost_incurred']}")
        print(f"是否满足约束: {'✅ 是' if final_report['within_constraints'] else '❌ 否'}")
        print("-" * 40)

    # 使用LangChain Agent
    print("\n\n" + "=" * 60)
    print("使用LangChain Agent进行资源感知优化")
    print("=" * 60)

    # 创建工具
    tools = create_langchain_tools(optimizer)

    # 创建内存
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 初始化Agent
    agent = initialize_agent(
        tools=tools,
        llm=optimizer.llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    # 测试Agent
    print("\n请输入您的要求（例如：对批次URGENT_001进行质量控制，时间限制2分钟，预算50美元，质量要求85分）：")
    user_input = input("> ")

    response = agent.run(user_input)
    print("\nAgent响应:")
    print(response)


