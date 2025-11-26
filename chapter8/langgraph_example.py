import os
import json
import ast
from typing import Dict, Any, List, TypedDict

from langgraph.graph import StateGraph, END

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 定义状态类型
class ConversationState(TypedDict):
    user_input: str
    chat_history: List[Any]
    user_profile: Dict[str, Any]
    response: str
    memory_updates: Dict[str, Any]


# 文件路径
USER_PROFILE_FILE = "user_profile.json"
CHAT_HISTORY_FILE = "chat_history.json"

# 初始化llm
from init_client import init_llm
llm = init_llm(0.7)


# 用户档案管理
def load_user_profile() -> Dict[str, Any]:
    """加载用户档案"""
    if os.path.exists(USER_PROFILE_FILE):
        with open(USER_PROFILE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_user_fact(fact: Dict[str, Any]):
    """保存新事实到用户档案"""
    profile = load_user_profile()
    profile.update(fact)
    with open(USER_PROFILE_FILE, 'w') as f:
        json.dump(profile, f, indent=4)
    print(f"[系统] 已记住新信息: {fact}")


# 对话历史管理
def load_chat_history(session_id: str) -> List[Any]:
    """加载对话历史"""
    histories = {}
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            histories = json.load(f)

    # 获取特定会话的历史
    session_history = histories.get(session_id, [])

    # 将字典转换为消息对象
    chat_history = []
    for msg in session_history:
        if msg["type"] == "human":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            chat_history.append(AIMessage(content=msg["content"]))

    return chat_history


def save_chat_history(session_id: str, chat_history: List[Any]):
    """保存对话历史"""
    histories = {}
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r') as f:
            histories = json.load(f)

    # 将消息对象转换为字典
    session_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            session_history.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            session_history.append({"type": "ai", "content": msg.content})

    histories[session_id] = session_history

    with open(CHAT_HISTORY_FILE, 'w') as f:
        json.dump(histories, f, indent=4)


# 提示词模板
def create_prompt_template():
    """创建提示词模板"""
    system_prompt = (
        "你是一个友好且健谈的个人助理。你的名字是 花花 助理。"
        "请根据下方的【用户档案】和【对话历史】来回答用户的问题。"
        "如果用户提供了新的个人信息，请在回答的末尾用一句话总结，并以 '[记忆更新]' 开头，例如：[记忆更新] {{'key': 'value'}}。"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
        ("system", "\n\n【用户档案】:\n{user_profile}")
    ])
    return prompt


# 定义LangGraph节点函数
def load_memory(state: ConversationState, session_id: str) -> ConversationState:
    """加载长期记忆到状态中"""
    state["user_profile"] = load_user_profile()
    state["chat_history"] = load_chat_history(session_id)
    return state


def generate_response(state: ConversationState, model, prompt) -> ConversationState:
    """生成AI响应"""
    chain = (
            RunnablePassthrough.assign(user_profile=lambda x: json.dumps(x["user_profile"], indent=2))
            | prompt
            | model
            | StrOutputParser()
    )

    response = chain.invoke({
        "user_input": state["user_input"],
        "chat_history": state["chat_history"],
        "user_profile": state["user_profile"]
    })

    state["response"] = response
    return state


def extract_memory_updates(state: ConversationState) -> ConversationState:
    """从响应中提取记忆更新"""
    response = state["response"]
    memory_updates = {}

    if "[记忆更新]" in response:
        try:
            fact_str = response.split("[记忆更新]")[1].strip()
            memory_updates = json.loads(fact_str)
        except (json.JSONDecodeError, IndexError):
            try:
                memory_updates = ast.literal_eval(fact_str)
            except Exception:
                print("[系统] 未能解析记忆更新。")

    state["memory_updates"] = memory_updates
    return state


def update_memory(state: ConversationState) -> ConversationState:
    """更新长期记忆"""
    if state["memory_updates"]:
        save_user_fact(state["memory_updates"])
    return state


def update_chat_history(state: ConversationState, session_id: str) -> ConversationState:
    """更新对话历史"""
    chat_history = state["chat_history"]
    chat_history.append(HumanMessage(content=state["user_input"]))
    chat_history.append(AIMessage(content=state["response"]))
    state["chat_history"] = chat_history

    # 保存到文件
    save_chat_history(session_id, chat_history)

    return state


# 创建LangGraph工作流
def create_conversation_graph():
    """创建对话工作流图"""
    prompt = create_prompt_template()

    # 创建工作流图
    workflow = StateGraph(ConversationState)

    # 添加节点
    workflow.add_node("load_memory", lambda state: load_memory(state, session_id="user_001_session"))
    workflow.add_node("generate_response", lambda state: generate_response(state, llm, prompt))
    workflow.add_node("extract_memory_updates", extract_memory_updates)
    workflow.add_node("update_memory", update_memory)
    workflow.add_node("update_chat_history", lambda state: update_chat_history(state, session_id="user_001_session"))

    # 设置入口点
    workflow.set_entry_point("load_memory")

    # 添加边
    workflow.add_edge("load_memory", "generate_response")
    workflow.add_edge("generate_response", "extract_memory_updates")
    workflow.add_edge("extract_memory_updates", "update_memory")
    workflow.add_edge("update_memory", "update_chat_history")
    workflow.add_edge("update_chat_history", END)

    # 编译工作流（不使用checkpointer）
    app = workflow.compile()

    return app


# 主交互循环
def chat_loop():
    """主交互循环"""
    print("花花 助理已启动！输入 'quit' 退出。")

    # 如果用户档案文件存在，删除它以开始新的会话
    if os.path.exists(USER_PROFILE_FILE):
        os.remove(USER_PROFILE_FILE)

    # 如果对话历史文件存在，删除它以开始新的会话
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

    # 创建对话图
    app = create_conversation_graph()
    # 可选：可视化图的结构
    app.get_graph().print_ascii()

    # 为本次会话设定一个固定的 session_id
    session_id = "user_001_session"

    while True:
        user_input = input("你: ")

        if user_input.lower() == 'quit':
            break

        # 初始化状态
        initial_state = {
            "user_input": user_input,
            "chat_history": [],
            "user_profile": {},
            "response": "",
            "memory_updates": {}
        }

        # 运行工作流
        result = app.invoke(initial_state)

        # 打印AI响应
        print(f"\n助理: {result['response']}\n")


if __name__ == "__main__":
    chat_loop()