import ast
import os
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. 初始化llm ---
from init_client import init_llm

llm = init_llm(0.7)

# --- 2. 模拟长期记忆（事实核心） ---
USER_PROFILE_FILE = "user_profile.json"


def load_user_profile(inputs: dict) -> str:
    """加载用户配置文件并格式化为字符串。"""
    if os.path.exists(USER_PROFILE_FILE):
        with open(USER_PROFILE_FILE, 'r') as f:
            profile = json.load(f)
            return json.dumps(profile, indent=2)
    return "暂无信息"


def save_user_fact(fact: dict):
    """保存一个新事实到用户配置文件"""
    profile = {}
    if os.path.exists(USER_PROFILE_FILE):
        with open(USER_PROFILE_FILE, 'r') as f:
            profile = json.load(f)
    profile.update(fact)
    with open(USER_PROFILE_FILE, 'w') as f:
        json.dump(profile, f, indent=4)
    print(f"[系统] 已记住新信息: {fact}")


# --- 3. 配置短期记忆（使用新范式） ---
# a. 创建一个存储会话历史的字典。在实际应用中，这可以是 Redis 或数据库。
store = {}
# b. 定义一个函数，根据 session_id 获取或创建会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 4. 创建提示词模板 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好且健谈的个人助理。你的名字是 花花 助理。"
               "请根据下方的【用户档案】和【对话历史】来回答用户的问题。"
               "如果用户提供了新的个人信息，请在回答的末尾用一句话总结，并以 '[记忆更新]' 开头，例如：[记忆更新] {{'key': 'value'}}。"
               "\n\n【用户档案】:\n{user_profile}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# --- 5. 创建一个带窗口限制的内部链 ---
# 这个链负责处理核心逻辑，包括加载用户档案和修剪历史
trimmer = trim_messages(
    max_tokens=4,  # 保留最近4条消息 (2轮对话)
    strategy="last",
    token_counter=len,
)

# base_chain 定义了核心处理流程，但它不知道如何获取历史
base_chain = (
        RunnablePassthrough.assign(user_profile=load_user_profile)
        | RunnablePassthrough.assign(chat_history=trimmer)  # 在这里修剪历史
        | prompt
        | llm
        | StrOutputParser()
)

# --- 6. 使用 RunnableWithMessageHistory 包装核心链 ---
# 这是最终暴露给用户的链，它负责管理历史记录
final_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# --- 7. 主交互循环 ---
def chat_loop():
    print("花花 助理已启动！输入 'quit' 退出。")
    if os.path.exists(USER_PROFILE_FILE):
        os.remove(USER_PROFILE_FILE)

    # 为本次会话设定一个固定的 session_id
    session_id = "user_001_session"

    while True:
        user_input = input("你: ")

        if user_input.lower() == 'quit':
            break

        # 调用 final_chain，将 session_id 放在 config 中
        ai_response = final_chain.invoke(
            {"question": user_input},  # 输入只包含问题
            config={"configurable": {"session_id": session_id}}  # session_id 在这里传递
        )
        print(f"\n助理: {ai_response}\n")

        # --- 调用后处理：手动管理长期记忆 ---
        if "[记忆更新]" in ai_response:
            try:
                fact_str = ai_response.split("[记忆更新]")[1].strip()
                new_fact = json.loads(fact_str)
                save_user_fact(new_fact)
            except (json.JSONDecodeError, IndexError):
                try:
                    new_fact = ast.literal_eval(fact_str)
                    save_user_fact(new_fact)
                except Exception as e2:
                    print("[系统] 未能解析记忆更新。")


if __name__ == "__main__":
    chat_loop()