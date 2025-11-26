from typing import List

from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, START, END

# --- 1. 模型初始化 ---
from init_client import init_llm

llm = init_llm(0.1)

# --- 2. 定义图的状态 ---
# 这是 LangGraph 的核心。状态是一个共享的字典，节点可以读取和写入它。
class State(TypedDict):
    feedback_text: str
    key_points: Annotated[List[str], "从反馈中提取的关键要点列表"]
    classified_points: Annotated[List[dict], "包含要点和情感分类的字典列表"]
    report: Annotated[str, "最终生成的摘要报告"]

# --- 3. 定义提示词模板 ---
# 与之前的提示词相同
prompt_extract = ChatPromptTemplate.from_template(
    "你是一位专业的客户反馈分析师。请仔细阅读以下客户反馈，并将其分解为独立的、简洁的要点。每个要点用一句话概括，不要添加任何额外的解释或评价。\n\n客户反馈：\n{feedback_text}\n\n关键要点："
)

prompt_classify = ChatPromptTemplate.from_template(
    "你是一位情感分析专家。请对下面列出的每一个关键要点进行分类，判断其属于'正面评价'、'负面评价'还是'功能建议'。请严格按照以下JSON数组格式输出，每个对象包含'point'（原始要点）和'sentiment'（情感类别）。\n\n例如：[\n  {{\"point\": \"界面很漂亮\", \"sentiment\": \"正面评价\"}},\n  {{\"point\": \"加载速度太慢\", \"sentiment\": \"负面评价\"}}\n]\n\n待分类的要点：\n{key_points}"
)

prompt_summarize = ChatPromptTemplate.from_template(
    "你是一位商业报告撰写专家。请根据下面已分类的客户反馈要点，生成一份结构清晰、专业的摘要报告。报告应包含三个部分：'优点与赞扬'、'问题与批评'和'改进建议'。请使用适当的标题和项目符号，使报告易于阅读。\n\n已分类的反馈要点（JSON格式）：\n{classified_points}\n\n客户反馈摘要报告："
)

# --- 4. 定义图的节点 ---
# 每个节点是一个函数，它接收当前状态，并返回一个状态更新字典。

def extract_node(state: State):
    """节点1：提取关键要点"""
    print("--- 节点1: 正在提取关键要点... ---")
    extraction_chain = prompt_extract | llm | StrOutputParser()
    # 从状态中获取输入
    key_points_text = extraction_chain.invoke({"feedback_text": state["feedback_text"]})
    # 将文本分割成列表
    key_points_list = [p.strip() for p in key_points_text.split('\n') if p.strip()]
    # 返回状态更新
    return {"key_points": key_points_list}

def classify_node(state: State):
    """节点2：对要点进行情感分类"""
    print("--- 节点2: 正在进行情感分类... ---")
    # 使用 JsonOutputParser 来确保输出是结构化的JSON
    classification_chain = prompt_classify | llm | JsonOutputParser()
    # 从状态中获取输入
    classified_points_list = classification_chain.invoke({"key_points": "\n".join(state["key_points"])})
    # 返回状态更新
    return {"classified_points": classified_points_list}

def summarize_node(state: State):
    """节点3：生成最终报告"""
    print("--- 节点3: 正在生成摘要报告... ---")
    summarization_chain = prompt_summarize | llm | StrOutputParser()
    # 从状态中获取输入
    final_report = summarization_chain.invoke({"classified_points": state["classified_points"]})
    # 返回状态更新
    return {"report": final_report}

# --- 5. 构建图 ---
# 创建一个状态图，传入我们定义的 State 类
workflow = StateGraph(State)

# 添加节点
workflow.add_node("extract", extract_node)
workflow.add_node("classify", classify_node)
workflow.add_node("summarize", summarize_node)

# 定义边，即节点的执行顺序
workflow.add_edge(START, "extract")
workflow.add_edge("extract", "classify")
workflow.add_edge("classify", "summarize")
workflow.add_edge("summarize", END)

# 编译图，使其成为一个可运行的应用
app = workflow.compile()

# --- 6. 运行图 ---
# 可选：可视化图的结构
app.get_graph().print_ascii()

print('--------------------')

# 示例客户反馈文本
customer_feedback = """
这款新出的手机真的很棒！屏幕色彩鲜艳，拿在手里的质感也超乎预期，非常高级。
不过，电池续航能力有点差，我一天下来需要充两次电，希望能改进。
另外，相机在夜间的表现不太理想，噪点比较多。
如果系统能增加一个应用双开功能就完美了，这对我的工作很有帮助。
总的来说，除了电池和相机，其他方面我都非常满意。
"""

# 创建初始状态
initial_state = {"feedback_text": customer_feedback}

# 调用图，传入初始状态
final_state = app.invoke(initial_state)

# --- 7. 打印最终结果 ---
print("\n\n--- 最终状态 ---")
print(final_state) # 你可以打印整个最终状态来查看所有信息

print("\n--- 客户反馈分析报告 ---")
print(final_state["report"])