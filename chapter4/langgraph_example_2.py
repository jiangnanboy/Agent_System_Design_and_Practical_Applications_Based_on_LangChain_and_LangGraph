from typing import Dict, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json

# 初始化模型
from init_client import init_llm

# 初始化两个DeepSeek实例
generator_llm = init_llm(temperature=0.7)
reviewer_llm = init_llm(temperature=0.1)


# 定义状态类
class AgentState(TypedDict):
    """Agent之间共享的状态"""
    user_input: str  # 用户输入的主题
    draft_text: str  # 生成者创建的草稿
    review_output: Dict[str, str]  # 审核者的评审结果


# 创建生成者Agent
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的内容创作者，擅长撰写关于各种主题的简短、信息丰富的段落。"),
    ("human", "撰写关于以下主题的简短、信息丰富的段落：\n{user_input}")
])

generator_chain = generator_prompt | generator_llm | StrOutputParser()

# 创建审核者Agent
reviewer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一个细致的事实核查员。
    1. 阅读提供的文本。
    2. 仔细验证所有声明的事实准确性。
    3. 你的最终输出必须是包含两个键的字典：
       - "status": 字符串，"ACCURATE" 或 "INACCURATE"。
       - "reasoning": 字符串，提供对你的状态的清楚解释，如果发现任何问题则引用具体问题。
    """),
    ("human", "请审核以下文本：\n{draft_text}")
])

# 使用JsonOutputParser确保输出为结构化JSON
reviewer_chain = reviewer_prompt | reviewer_llm | JsonOutputParser()


# 定义生成者节点
def generator_node(state: AgentState) -> AgentState:
    """生成者Agent：创建初始草稿"""
    print(">>> 生成者Agent正在创建草稿...")
    draft = generator_chain.invoke({"user_input": state["user_input"]})
    print(f"生成的草稿：\n{draft}\n")
    return {"draft_text": draft}


# 定义审核者节点
def reviewer_node(state: AgentState) -> AgentState:
    """审核者Agent：审核草稿内容"""
    print(">>> 审核者Agent正在审核草稿...")
    review = reviewer_chain.invoke({"draft_text": state["draft_text"]})
    json_str = json.dumps(review, ensure_ascii=False, indent=2)
    print(f"审核结果：\n{json_str}\n")
    return {"review_output": review}


# 创建工作流图
def create_sequential_graph():
    """创建顺序执行的Agent工作流图"""
    # 初始化状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("generator", generator_node)
    workflow.add_node("reviewer", reviewer_node)

    # 设置入口点
    workflow.set_entry_point("generator")

    # 添加边
    workflow.add_edge("generator", "reviewer")
    workflow.add_edge("reviewer", END)

    # 设置内存保存器（用于状态持久化）
    memory = MemorySaver()

    # 编译图
    app = workflow.compile(checkpointer=memory)

    return app


# 执行工作流的函数
def run_review_graph(user_topic: str) -> AgentState:
    """运行生成和审核工作流"""
    # 创建工作流图
    app = create_sequential_graph()

    # 打印图的结构（可选，非常直观！）
    try:
        print("--- 图结构 ---")
        app.get_graph().print_ascii()
        print("\n" + "=" * 20 + "\n")
    except Exception as e:
        print(f"无法打印图结构: {e}")

    # 初始状态
    initial_state = {
        "user_input": user_topic,
        "draft_text": "",
        "review_output": {}
    }

    # 执行工作流
    print(f"{'=' * 50}\n开始处理主题：{user_topic}\n{'=' * 50}")

    # 使用线程ID和配置运行图
    config = {"configurable": {"thread_id": "1"}}
    final_state = app.invoke(initial_state, config)

    # 输出最终结果
    print(f"{'=' * 50}\n最终结果\n{'=' * 50}")
    print(f"原始主题：{final_state['user_input']}")
    print(f"生成草稿：\n{final_state['draft_text']}")
    print(f"审核结果：\n状态：{final_state['review_output']['status']}")
    print(f"说明：{final_state['review_output']['reasoning']}")

    return final_state


# 测试函数
if __name__ == "__main__":
    # 测试主题
    test_topic = "量子计算的基本原理"

    # 运行工作流
    result = run_review_graph(test_topic)