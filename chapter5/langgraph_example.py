from typing import TypedDict, List, Annotated

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- 1.初始化模型 ---
from init_client import init_llm
llm = init_llm(0.1)

# --- 2. 模拟数据库 ---
# 模拟一个包含商品信息的数据库
product_database = {
    "prod_001": {
        "name": "UltraBook Pro 14",
        "brand": "FutureTech",
        "price": 12999,
        "specs": {
            "cpu": "Intel Core i7-13代",
            "ram": "16GB LPDDR5",
            "storage": "512GB NVMe SSD",
            "display": "14英寸 2.8K OLED"
        }
    },
    "prod_002": {
        "name": "MegaBook X",
        "brand": "Infinity",
        "price": 11999,
        "specs": {
            "cpu": "AMD Ryzen 7 7840HS",
            "ram": "32GB DDR5",
            "storage": "1TB NVMe SSD",
            "display": "15.6英寸 1080p IPS"
        }
    },
    "prod_003": {
        "name": "SlimAir Go",
        "brand": "CloudWalk",
        "price": 8999,
        "specs": {
            "cpu": "Apple M2",
            "ram": "8GB",
            "storage": "256GB SSD",
            "display": "13.6英寸 Liquid Retina"
        }
    }
}

# 模拟一个用户评论数据库
reviews_database = {
    "prod_001": [
        {"rating": 5, "comment": "屏幕素质惊人，性能强劲，非常满意！"},
        {"rating": 4, "comment": "一切都好，就是价格有点贵。"},
        {"rating": 5, "comment": "OLED屏幕看电影的体验无敌了。"}
    ],
    "prod_002": [
        {"rating": 5, "comment": "性价比之王，32GB内存太爽了，多任务无压力。"},
        {"rating": 3, "comment": "性能不错，但做工一般，而且有点重。"},
        {"rating": 4, "comment": "AMD处理器表现很好，续航比预想中长。"}
    ],
    "prod_003": [
        {"rating": 1, "comment": "不咋的，内存真小。"},
        {"rating": 2, "comment": "还行吧，一般办公用用。"},
        {"rating": 3, "comment": "参数一般，但苹果质量还是不错的。"}
    ]
}


# --- 3. 工具定义 ---
@tool
def search_product(query: str) -> str:
    """
    根据关键词搜索商品。返回匹配商品的ID和名称。
    例如，输入 'FutureTech' 或 'UltraBook'。
    """
    print(f"\n--- 🛠️ 工具调用: search_product, 查询: '{query}' ---")
    results = []
    for pid, product in product_database.items():
        if query.lower() in product["name"].lower() or query.lower() in product["brand"].lower():
            results.append(f"ID: {pid}, 名称: {product['name']}")
    if not results:
        return f"未找到与 '{query}' 相关的商品。"
    print(f"--- 工具结果: 找到 {len(results)} 个商品 ---")
    return "\n".join(results)


@tool
def get_product_details(product_id: str) -> str:
    """获取指定商品ID的详细信息，包括价格和规格。"""
    print(f"\n--- 🛠️ 工具调用: get_product_details, 商品ID: '{product_id}' ---")
    product = product_database.get(product_id)
    if not product:
        return f"错误：未找到ID为 '{product_id}' 的商品。"
    details = (
        f"名称: {product['name']}\n品牌: {product['brand']}\n价格: ¥{product['price']}\n规格:\n  - CPU: {product['specs']['cpu']}\n  - 内存: {product['specs']['ram']}\n  - 存储: {product['specs']['storage']}\n  - 屏幕: {product['specs']['display']}")
    print(f"--- 工具结果: 已获取商品详情 ---")
    return details


@tool
def analyze_reviews(product_id: str) -> str:
    """获取并总结指定商品ID的用户评论。"""
    print(f"\n--- 🛠️ 工具调用: analyze_reviews, 商品ID: '{product_id}' ---")
    reviews = reviews_database.get(product_id)
    if not reviews:
        return f"错误：未找到ID为 '{product_id}' 的商品评论。"
    summary = "用户评论总结:\n"
    positive_count = sum(1 for r in reviews if r['rating'] >= 4)
    negative_count = len(reviews) - positive_count
    summary += f"- 好评 ({positive_count}条): 用户普遍赞赏其性能和屏幕。\n"
    if negative_count > 0:
        summary += f"- 差评/中评 ({negative_count}条): 主要抱怨价格和重量。\n"
    summary += "\n代表性评论:\n" + "\n".join([f"- ({r['rating']}/5) {r['comment']}" for r in reviews])
    print(f"--- 工具结果: 已分析评论 ---")
    return summary


tools = [search_product, get_product_details, analyze_reviews]

# --- 4. LangGraph 工作流定义 ---

# 定义 Agent 的状态。状态是在图的节点之间传递的信息。
# 这里我们使用一个预定义的状态，它只包含一个 `messages` 键。
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 定义决定下一步路由的函数
def should_continue(state: State) -> str:
    """
    决定下一步是调用工具还是结束。
    """
    messages = state['messages']
    last_message = messages[-1]
    # 如果 LLM 决定调用工具，我们路由到 "tools" 节点
    if last_message.tool_calls:
        return "tools"
    # 否则，我们结束
    return END


# 定义调用模型的节点
def call_model(state: State):
    """
    调用模型并获取响应。
    """
    messages = state['messages']
    # 将工具绑定到模型上
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    # 返回一个包含新消息列表的状态
    return {"messages": [response]}


# --- 5. 构建图 ---
# 从我们的状态定义创建一个 StateGraph
workflow = StateGraph(State)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# 设置入口点
workflow.set_entry_point("agent")

# 添加条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["tools", END]
)

# 添加从 tools 回到 agent 的普通边
workflow.add_edge("tools", "agent")

# 编译图
app = workflow.compile()

# 打印图的结构（可选，非常直观！）
try:
    print("--- 图结构 ---")
    app.get_graph().print_ascii()
    print("\n" + "=" * 20 + "\n")
except Exception as e:
    print(f"无法打印图结构: {e}")

# --- 6. 执行与演示 ---
def product_research_agent():
    query = "我想比较一下 FutureTech 和 Infinity 这两个品牌的旗舰笔记本，帮我做个决定。"
    print(f"用户问题: {query}\n")

    # 初始状态，包含用户的第一个消息
    initial_state = {"messages": [HumanMessage(content=query)]}

    # 使用 stream 来实时查看每一步的输出
    for event in app.stream(initial_state):
        for key, value in event.items():
            print(f"--- 节点 '{key}' 的输出 ---")
            # pprint(value) # 可以打印完整的状态，但通常我们只关心最后一条消息
            if 'messages' in value and value['messages']:
                last_message = value['messages'][-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    print(f"AI 思考/回答: {last_message.content}")
                elif isinstance(last_message, AIMessage) and last_message.tool_calls:
                    print(f"AI 决定调用工具: {last_message.tool_calls}")
            print("-" * 20)

    # 获取最终状态
    final_state = app.invoke(initial_state)
    final_message = final_state['messages'][-1]
    print("\n--- ✅ 最终助理报告 ---")
    print(final_message.content)


# 运行测试
if __name__ == "__main__":

    product_research_agent()