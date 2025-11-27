import random
from typing import Literal
from init_client import init_llm

# --- 1. 导入 LangGraph 核心组件 ---
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool


# --- 2. 定义工业工具 ---

@tool
def pick_and_place_gear(part_id: str) -> str:
    """指令机器人拾取并安装一个齿轮。输入是零件ID。"""
    print(f"--- 工具调用: 尝试拾取并安装齿轮 '{part_id}' ---")
    if random.random() < 0.7:
        return "成功：齿轮已拾取并安装。"
    else:
        failure_type = random.choice(["拾取失败：料仓为空或齿轮位置偏移。", "安装失败：检测到碰撞或阻力过大。"])
        return f"错误：{failure_type}"


@tool
def verify_assembly_with_vision() -> str:
    """使用视觉系统验证齿轮是否安装正确。"""
    print("--- 工具调用: 使用视觉系统验证装配结果 ---")
    if random.random() < 0.8:
        return "验证成功：齿轮位置正确。"
    else:
        return "验证失败：视觉系统检测到齿轮安装位置存在显著偏移。"


@tool
def reset_robot_position() -> str:
    """将机器人臂复位到安全的初始位置。"""
    print("--- 工具调用: 复位机器人到安全位置 ---")
    return "机器人已成功复位。"


@tool
def alert_maintenance_team(message: str) -> str:
    """向维护团队发送详细的警报信息。"""
    print(f"--- 工具调用: 向维护团队发送警报 ---")
    print(f"警报内容: {message}")
    return "警报已成功发送给维护团队。"


tools = [pick_and_place_gear, verify_assembly_with_vision, reset_robot_position, alert_maintenance_team]


# --- 3. 定义 LangGraph 的状态 ---
# 继承 MessagesState，并添加一个自定义的 retry_count 字段来跟踪重试次数
class AgentState(MessagesState):
    retry_count: int


# --- 4. 创建图的核心组件 ---

# 初始化 LLM 和工具节点
llm = init_llm(temperature=0.1).bind_tools(tools)
tool_node = ToolNode(tools)


# 定义 "agent" 节点：这是 LLM 进行思考和决策的地方
def agent_node(state: AgentState):
    """调用 LLM 来决定下一步做什么。"""
    messages = state["messages"]
    # LLM 会根据历史消息和当前状态来生成响应（可能包含工具调用）
    response = llm.invoke(messages)
    return {"messages": [response]}


# 定义 "路由" 函数：决定在 agent 节点之后去哪里
def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
    """如果 LLM 决定调用工具，则进入 tools 节点，否则结束。"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


# 定义 "致命失败处理" 节点：当无法恢复的错误发生时，执行此节点
def handle_fatal_failure(state: AgentState):
    """
    处理无法恢复的失败，执行复位和报警。
    此函数现在会正确地从消息历史中查找导致失败的工具调用信息。
    """
    # 找到请求调用工具的 AIMessage，而不是 ToolMessage 响应。
    # 工具调用请求在工具结果之前，所以是倒数第二条消息。
    if len(state["messages"]) < 2:
        # 这是一个防御性检查，虽然在这个流程中不应该发生。
        return {"messages": [AIMessage(content="错误：无法确定失败的工具。")]}

    tool_call_message = state["messages"][-2]

    # 确保它确实是一个带有工具调用的 AIMessage
    if not isinstance(tool_call_message, AIMessage) or not tool_call_message.tool_calls:
         return {"messages": [AIMessage(content="错误：无法从消息历史中找到工具调用。")]}

    # 获取导致失败的工具调用信息
    tool_call = tool_call_message.tool_calls[0]

    # 根据失败原因构建报警信息
    failure_reason = ""
    if tool_call["name"] == "pick_and_place_gear":
        failure_reason = "拾取或安装齿轮失败"
    elif tool_call["name"] == "verify_assembly_with_vision":
        failure_reason = "装配验证多次失败，可能存在设备校准问题"

    # 构造一个包含工具调用的 AIMessage，以触发 ToolNode
    reset_call = {"name": "reset_robot_position", "args": {}, "id": "reset_call"}
    alert_call = {"name": "alert_maintenance_team", "args": {"message": f"致命错误: {failure_reason}，请立即介入。"}, "id": "alert_call"}

    # 返回一个新的 AIMessage，指示 ToolNode 执行这两个工具
    fatal_response = AIMessage(
        content="检测到致命错误，正在执行安全恢复程序。",
        tool_calls=[reset_call, alert_call]
    )
    return {"messages": [fatal_response]}


# 定义 "路由" 函数：在 tools 节点执行后，根据结果决定下一步
def route_after_tools(state: AgentState) -> Literal["agent", "handle_fatal_failure", "__end__"]:
    """
    检查工具执行的结果，决定是继续循环、处理致命失败还是结束任务。
    """
    messages = state["messages"]
    last_message = messages[-1]  # 这是 ToolMessage

    # 检查工具执行的结果
    if "错误：" in last_message.content:
        tool_call_that_failed = messages[-2].tool_calls[0]  # 同样，从 AIMessage 获取信息

        # 如果是拾取失败，这是致命的，需要立即处理
        if tool_call_that_failed["name"] == "pick_and_place_gear":
            return "handle_fatal_failure"

        # 如果是验证失败，检查重试次数
        if tool_call_that_failed["name"] == "verify_assembly_with_vision":
            # 从状态中获取并增加重试计数
            current_retry_count = state.get("retry_count", 0)
            if current_retry_count >= 2:
                return "handle_fatal_failure"  # 重试次数耗尽，处理致命失败
            # 如果还可以重试，不需要在这里做任何特殊操作，
            # 只需返回给 agent，让 agent 决定重试。
            # 需要在 agent 节点中管理重试计数。

    # 如果没有致命错误，或者任务成功，则返回给 agent 继续思考
    # Agent 会根据结果决定是重试还是结束
    return "agent"


def agent_node_with_retry(state: AgentState):
    """调用 LLM 来决定下一步做什么，并管理重试逻辑。"""
    messages = state["messages"]
    last_message = messages[-1]

    # 检查是否因为验证失败而需要重试
    if isinstance(last_message, ToolMessage) and "验证失败" in last_message.content:
        current_retry_count = state.get("retry_count", 0)
        if current_retry_count < 2:
            # 增加重试计数
            new_retry_count = current_retry_count + 1
            print(f"--- 验证失败，准备进行第 {new_retry_count} 次重试 ---")

            # 构造一个新的 AIMessage 来强制执行重试步骤
            # 这比让 LLM 自由重试更可控
            retry_call = {"name": "reset_robot_position", "args": {}, "id": "reset_for_retry"}
            response = AIMessage(
                content=f"验证失败，这是第 {new_retry_count} 次尝试。我将首先复位机器人，然后重新执行装配。",
                tool_calls=[retry_call]
            )
            return {"messages": [response], "retry_count": new_retry_count}

    # 如果不是重试场景，则正常调用 LLM
    response = llm.invoke(messages)
    return {"messages": [response]}

# --- 5. 构建状态图 ---

workflow = StateGraph(AgentState)

# 添加节点 (使用新的 agent_node)
workflow.add_node("agent", agent_node_with_retry)
workflow.add_node("tools", tool_node)
workflow.add_node("handle_fatal_failure", handle_fatal_failure)

# 设置入口点
workflow.add_edge(START, "agent")

# 添加条件边
workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_conditional_edges("tools", route_after_tools)

# 致命失败处理流程
workflow.add_edge("handle_fatal_failure", "tools")
workflow.add_edge("tools", END) # 致命处理执行完工具后，应该直接结束

# 编译图
app = workflow.compile()

# 可选：可视化图的结构
app.get_graph().print_ascii()

# --- 6. 运行并测试 ---

print("\n\n==================== 开始工业装配任务 ====================")
# 为了演示，设置随机种子以获得可预测的失败场景
# random.seed(1) # 可能会触发 pick_and_place 失败
random.seed(2)  # 可能会触发 verify_assembly 失败，从而进入重试逻辑

# 设置初始状态
initial_state = {
    "messages": [
        HumanMessage(content="请开始执行齿轮 G-1234 的装配任务。")
    ],
    "retry_count": 0
}

# 使用 stream 方法来观察每一步的执行过程
for event in app.stream(initial_state):
    # print(event) # 可以打印完整的事件信息
    for key, value in event.items():
        if key != "__end__":
            print(f"--- 节点 '{key}' 执行完成 ---")
            # 打印该节点产生的最后一条消息
            print(f"最新消息: {value['messages'][-1].content.strip()}")
    print("-" * 20)
