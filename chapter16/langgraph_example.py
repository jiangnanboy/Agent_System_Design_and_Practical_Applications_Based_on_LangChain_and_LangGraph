import time
import random
from typing import Dict, Any, TypedDict, Annotated, List

# 导入LangGraph相关库
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from init_client import init_llm

# --- 1. 定义状态 ---
# 使用 TypedDict 定义图中节点之间传递的状态
class InspectionState(TypedDict):
    batch_id: str
    constraints: Dict[str, float]
    initial_decision: str
    final_decision: str
    inspection_result: Dict[str, Any]
    is_within_constraints: bool
    messages: Annotated[List[HumanMessage | AIMessage], "Messages"]


# --- 2. 定义资源约束和检测工具 ---
class ManufacturingConstraints:
    def __init__(self, time_limit: float, cost_budget: float, quality_threshold: float):
        self.time_limit = time_limit
        self.cost_budget = cost_budget
        self.quality_threshold = quality_threshold


class InspectionMethods:
    @staticmethod
    def basic_visual_inspection(batch_id: str) -> Dict[str, Any]:
        time.sleep(1)
        defects_found = random.randint(0, 5)
        quality_score = max(80, 100 - defects_found * 4)
        return {
            "method_used": "basic_visual_inspection",
            "batch_id": batch_id, "defects_found": defects_found, "quality_score": quality_score,
            "time_taken": 1.0, "cost_incurred": 20.0
        }

    @staticmethod
    def standard_sensor_inspection(batch_id: str) -> Dict[str, Any]:
        time.sleep(3)
        defects_found = random.randint(0, 3)
        quality_score = max(90, 100 - defects_found * 3)
        return {
            "method_used": "standard_sensor_inspection",
            "batch_id": batch_id, "defects_found": defects_found, "quality_score": quality_score,
            "time_taken": 3.0, "cost_incurred": 75.0
        }

    @staticmethod
    def precision_ai_inspection(batch_id: str) -> Dict[str, Any]:
        time.sleep(7)
        defects_found = random.randint(0, 2)
        quality_score = max(98, 100 - defects_found * 1)
        return {
            "method_used": "precision_ai_inspection",
            "batch_id": batch_id, "defects_found": defects_found, "quality_score": quality_score,
            "time_taken": 7.0, "cost_incurred": 250.0
        }


# --- 3. 定义图的节点 ---

# 节点1: 智能决策
def agent_node(state: InspectionState):
    llm = init_llm(temperature=0.1)

    decision_prompt = PromptTemplate(
        input_variables=["time_limit", "cost_budget", "quality_threshold"],
        template="""
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
    )

    decision_chain = decision_prompt | llm | JsonOutputParser()

    decision = decision_chain.invoke(state["constraints"])

    print(f"🤖 Agent决策: 选择 '{decision['method']}' 方法。")
    return {"initial_decision": decision['method'], "messages": [AIMessage(content=f"决策选择: {decision['method']}")]}


# 节点2: 执行检测
def execute_inspection_node(state: InspectionState):
    method_to_call = state["initial_decision"]
    batch_id = state["batch_id"]

    if method_to_call == "basic_visual_inspection":
        result = InspectionMethods.basic_visual_inspection(batch_id)
    elif method_to_call == "standard_sensor_inspection":
        result = InspectionMethods.standard_sensor_inspection(batch_id)
    else:  # precision_ai_inspection
        result = InspectionMethods.precision_ai_inspection(batch_id)

    print(f"🔧 执行检测: 使用 '{method_to_call}' 完成。")
    return {"inspection_result": result, "final_decision": method_to_call}


# 节点3: 验证约束
def check_constraints_node(state: InspectionState):
    result = state["inspection_result"]
    constraints = state["constraints"]

    time_ok = result['time_taken'] <= constraints['time_limit']
    cost_ok = result['cost_incurred'] <= constraints['cost_budget']
    quality_ok = result['quality_score'] >= constraints['quality_threshold']

    is_ok = time_ok and cost_ok and quality_ok

    print(f"🔍 验证约束: {'✅ 满足' if is_ok else '❌ 不满足'}所有约束。")
    return {"is_within_constraints": is_ok}


# 节点4: 回退决策
def fallback_node(state: InspectionState):
    initial_choice = state["initial_decision"]
    fallback_method = "basic_visual_inspection"  # 默认回退

    if initial_choice == "precision_ai_inspection":
        fallback_method = "standard_sensor_inspection"
    elif initial_choice == "standard_sensor_inspection":
        fallback_method = "basic_visual_inspection"

    print(f"⚠️ 触发回退: 从 '{initial_choice}' 回退到 '{fallback_method}'。")
    return {"final_decision": fallback_method, "messages": [AIMessage(content=f"回退到: {fallback_method}")]}


# 节点5: 执行回退后的检测
def execute_fallback_node(state: InspectionState):
    method_to_call = state["final_decision"]
    batch_id = state["batch_id"]

    if method_to_call == "basic_visual_inspection":
        result = InspectionMethods.basic_visual_inspection(batch_id)
    else:  # standard_sensor_inspection
        result = InspectionMethods.standard_sensor_inspection(batch_id)

    print(f"🔧 执行回退检测: 使用 '{method_to_call}' 完成。")
    return {"inspection_result": result}


# 节点6: 生成最终报告
def final_report_node(state: InspectionState):
    result = state["inspection_result"]
    constraints = state["constraints"]

    report = f"""
    ================== 最终报告 ==================
    批次ID: {result['batch_id']}
    初始决策方法: {state['initial_decision']}
    最终执行方法: {state['final_decision']}
    发现缺陷数: {result['defects_found']}
    最终质量分数: {result['quality_score']}/100
    实际耗时: {result['time_taken']} 分钟 (限制: {constraints['time_limit']})
    实际成本: ${result['cost_incurred']} (预算: ${constraints['cost_budget']})
    ============================================
    """
    print(report)
    return {"messages": [AIMessage(content=report)]}


# --- 4. 构建状态图 ---
def build_graph():
    # 创建一个新的状态图
    workflow = StateGraph(InspectionState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("execute_inspection", execute_inspection_node)
    workflow.add_node("check_constraints", check_constraints_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("execute_fallback", execute_fallback_node)
    workflow.add_node("final_report", final_report_node)

    # 定义边
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "execute_inspection")
    workflow.add_edge("execute_inspection", "check_constraints")

    # 添加条件边
    workflow.add_conditional_edges(
        "check_constraints",
        lambda state: "fallback" if not state["is_within_constraints"] else "final_report",
        {
            "fallback": "fallback",
            "final_report": "final_report"
        }
    )

    workflow.add_edge("fallback", "execute_fallback")
    workflow.add_edge("execute_fallback", "final_report")

    workflow.add_edge("final_report", END)

    # 编译图
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


# --- 5. 主程序与测试 ---
if __name__ == "__main__":
    # 构建图
    qc_graph = build_graph()
    # 可选：可视化图的结构
    qc_graph.get_graph().print_ascii()

    # 定义测试场景
    scenarios = {
        "紧急订单": ManufacturingConstraints(time_limit=2.0, cost_budget=50.0, quality_threshold=85.0),
        "标准订单": ManufacturingConstraints(time_limit=5.0, cost_budget=100.0, quality_threshold=92.0),
        "高价值客户订单": ManufacturingConstraints(time_limit=10.0, cost_budget=300.0, quality_threshold=98.0)
    }

    # 遍历并运行每个场景
    for name, constraints in scenarios.items():
        print("=" * 60)
        print(f"🚀 场景: {name}")
        print("=" * 60)

        # 初始化状态
        initial_state = {
            "batch_id": f"BATCH-{name.replace(' ', '_').upper()}",
            "constraints": {
                "time_limit": constraints.time_limit,
                "cost_budget": constraints.cost_budget,
                "quality_threshold": constraints.quality_threshold
            },
            "messages": []
        }

        # 调用图来执行流程
        # 使用 thread_id 来为每个对话创建独立的检查点
        final_state = qc_graph.invoke(initial_state, config={"configurable": {"thread_id": name}})

        print("\n--- 流程结束 ---")
        print("-" * 60 + "\n")

