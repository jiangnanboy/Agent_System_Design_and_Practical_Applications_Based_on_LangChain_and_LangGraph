import random
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate

from init_client import init_llm
# --- 1. 定义工业工具 ---
@tool
def pick_and_place_gear(part_id: str) -> str:
    """指令机器人拾取并安装一个齿轮。输入是零件ID。"""
    print(f"--- 工具调用: 尝试拾取并安装齿轮 '{part_id}' ---")
    # 模拟 70% 的成功率
    if random.random() < 0.7:
        return "成功：齿轮已拾取并安装。"
    else:
        # 模拟不同类型的失败
        failure_type = random.choice(["拾取失败：料仓为空或齿轮位置偏移。", "安装失败：检测到碰撞或阻力过大。"])
        return f"错误：{failure_type}"

@tool
def verify_assembly_with_vision() -> str:
    """使用视觉系统验证齿轮是否安装正确。"""
    print("--- 工具调用: 使用视觉系统验证装配结果 ---")
    # 模拟 80% 的验证成功率
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
    """向维护团队发送详细的警报信息。输入是警报内容。"""
    print(f"--- 工具调用: 向维护团队发送警报 ---")
    print(f"警报内容: {message}")
    return "警报已成功发送给维护团队。"

tools = [pick_and_place_gear, verify_assembly_with_vision, reset_robot_position, alert_maintenance_team]

# --- 2. 创建具备异常处理能力的工业 Agent ---

llm = init_llm(temperature=0.1)

# 定义一个强调安全、重试和升级的工业级 Prompt
prompt_template = """
你是一个负责监督机器人装配线的AI班组长。你的首要目标是安全、高效地完成任务，并在出现问题时进行妥善处理。

当前任务：安装一个齿轮（ID: G-1234）。

**核心操作流程与异常处理协议：**

1.  **开始任务**: 首先，你必须使用 `pick_and_place_gear` 工具，并输入零件ID "G-1234"。
2.  **处理 `pick_and_place_gear` 的结果**:
    - 如果结果以 "错误：" 开头，说明机器人操作失败。
        - 你必须立即调用 `reset_robot_position` 工具。
        - 然后，调用 `alert_maintenance_team` 工具，将错误信息作为警报内容发送出去。
        - 任务结束，不要进行任何其他操作。
    - 如果结果是 "成功"，则继续执行下一步。
3.  **执行验证**: 在成功安装后，你**必须**调用 `verify_assembly_with_vision` 工具来检查质量。
4.  **处理 `verify_assembly_with_vision` 的结果**:
    - 如果结果是 "验证成功"，恭喜你，任务圆满完成。
    - 如果结果是 "验证失败"，说明装配质量有问题。你必须启动一个**最多2次的重试恢复流程**。
        - **重试步骤**:
            a. 调用 `reset_robot_position`。
            b. 再次调用 `pick_and_place_gear` 工具。
            c. 再次调用 `verify_assembly_with_vision` 工具。
        - **评估重试结果**:
            - 如果重试后验证成功，任务完成。
            - 如果重试后验证**仍然失败**，说明可能存在设备校准等系统性问题。你必须：
                i. 调用 `reset_robot_position`。
                ii. 调用 `alert_maintenance_team`，发送消息"警报：齿轮装配多次验证失败，请检查机器人校准。"
                iii. 任务结束。

严格遵守以上协议。你的每一步操作都必须基于上一步的观察结果。

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# 这个函数现在接受 llm, tools, 和 prompt 作为参数
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 Agent 执行器，设置 verbose=True 以观察其思考过程
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 3. 运行并测试 ---

print("\n\n==================== 开始工业装配任务 ====================")
# 为了演示，我们设置随机种子以获得可预测的失败场景
# random.seed(1) # 可能会触发 pick_and_place 失败
random.seed(2) # 可能会触发 verify_assembly 失败，从而进入重试逻辑

agent_executor.invoke({"input": "请开始执行齿轮 G-1234 的装配任务。"})