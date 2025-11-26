from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 假设你的 init_llm 函数在 init_client.py 中
from init_client import init_llm


# --- 1. 定义图的状态 ---
# 使用 TypedDict 来定义在图的各个节点之间共享的状态信息
class AgentState(TypedDict):
    query: str
    category: str
    response: str


# --- 2. 初始化模型和工具链 ---
llm = init_llm(0)

# 路由决策提示模板
router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""分析用户查询并确定它属于哪个类别：
    - 如果查询涉及技术问题、产品功能或故障排除，返回"technical"
    - 如果查询涉及账单、付款、订阅或退款，返回"billing"
    - 如果查询是一般性问题、产品信息或公司政策，返回"general"
    - 如果查询不明确或不属于任何类别，返回"unknown"

    只返回一个词：technical, billing, general 或 unknown

    用户查询：{query}"""
)

# 创建路由决策链
router_chain = router_prompt | llm | StrOutputParser()


# --- 3. 定义图的节点 ---
# 每个节点都是一个函数，它接收当前状态，并返回状态更新

def route_query_node(state: AgentState):
    """节点1：分析查询并决定路由类别"""
    print("--- 正在进行路由决策 ---")
    query = state["query"]
    category = router_chain.invoke({"query": query})
    print(f"决策结果: {category}")
    return {"category": category}


def technical_handler_node(state: AgentState):
    """节点2：处理技术问题"""
    print("\n--- 路由到技术支持 ---")
    query = state["query"]
    response = f"技术支持处理了查询：'{query}'。结果：已生成技术解决方案。"
    return {"response": response}


def billing_handler_node(state: AgentState):
    """节点3：处理账单问题"""
    print("\n--- 路由到账单支持 ---")
    query = state["query"]
    response = f"账单支持处理了查询：'{query}'。结果：已提供账单信息。"
    return {"response": response}


def general_handler_node(state: AgentState):
    """节点4：处理一般问题"""
    print("\n--- 路由到一般支持 ---")
    query = state["query"]
    response = f"一般支持处理了查询：'{query}'。结果：已提供一般信息。"
    return {"response": response}


def unknown_handler_node(state: AgentState):
    """节点5：处理无法识别的问题"""
    print("\n--- 无法识别查询类型 ---")
    query = state["query"]
    response = f"系统无法处理查询：'{query}'。请重新表述您的问题。"
    return {"response": response}


# --- 4. 构建图 ---
from langgraph.graph import StateGraph, END, START

# 创建一个基于我们定义的状态的图
workflow = StateGraph(AgentState)

# 添加所有节点到图中
workflow.add_node("route_query", route_query_node)
workflow.add_node("technical_handler", technical_handler_node)
workflow.add_node("billing_handler", billing_handler_node)
workflow.add_node("general_handler", general_handler_node)
workflow.add_node("unknown_handler", unknown_handler_node)

# 设置图的入口点
workflow.set_entry_point("route_query")

# 添加条件边：这是路由的核心
# 根据路由决策节点的输出，决定下一步流向哪个处理节点
workflow.add_conditional_edges(
    "route_query",  # 起始节点
    lambda state: state["category"],  # 决策函数：从状态中获取类别
    {
        "technical": "technical_handler",
        "billing": "billing_handler",
        "general": "general_handler",
        "unknown": "unknown_handler",
    }
)

# 所有处理节点完成后，都流向终点
workflow.add_edge("technical_handler", END)
workflow.add_edge("billing_handler", END)
workflow.add_edge("general_handler", END)
workflow.add_edge("unknown_handler", END)

# 编译图，使其成为一个可运行的应用
app = workflow.compile()


# --- 5. 示例用法 ---
def main():
    if not llm:
        print("\n由于LLM初始化失败，跳过执行。")
        return

    # 打印图的结构（可选，非常直观！）
    try:
        print("--- 图结构 ---")
        app.get_graph().print_ascii()
        print("\n" + "=" * 20 + "\n")
    except Exception as e:
        print(f"无法打印图结构: {e}")

    # 测试查询
    queries = [
        "我的软件无法启动，显示错误代码0x80070005。",
        "我如何更改我的订阅计划？",
        "你们的营业时间是什么时候？",
        "blargh snorfle wump?"
    ]

    for query in queries:
        print(f"用户查询: {query}")

        # 通过调用 app.invoke 并传入初始状态来启动图
        final_state = app.invoke({"query": query})

        print(f"最终回复: {final_state['response']}")
        print("-" * 20)


if __name__ == "__main__":
    main()