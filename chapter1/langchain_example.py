from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. 初始化llm ---
from init_client import init_llm

llm = init_llm(0.1)

# --- 2. 定义提示词模板 ---

# 提示词 1：提取要点
prompt_extract = ChatPromptTemplate.from_template(
    "你是一位专业的客户反馈分析师。请仔细阅读以下客户反馈，并将其分解为独立的、简洁的要点。每个要点用一句话概括，"
    "不要添加任何额外的解释或评价。\n\n客户反馈：\n{feedback_text}\n\n关键要点："
)

# 提示词 2：情感分类
prompt_classify = ChatPromptTemplate.from_template(
    "你是一位情感分析专家。请对下面列出的每一个关键要点进行分类，判断其属于'正面评价'、'负面评价'还是'功能建议'。"
    "请严格按照以下JSON数组格式输出，每个对象包含'point'（原始要点）和'sentiment'（情感类别）。\n\n例如：[\n  {{\"point\": \"界面很漂亮\", \"sentiment\": \"正面评价\"}},\n  {{\"point\": \"加载速度太慢\", \"sentiment\": \"负面评价\"}}\n]\n\n待分类的要点：\n{key_points}"
)

# 提示词 3：生成报告
prompt_summarize = ChatPromptTemplate.from_template(
    "你是一位商业报告撰写专家。请根据下面已分类的客户反馈要点，生成一份结构清晰、专业的摘要报告。报告应包含三个部分："
    "'优点与赞扬'、'问题与批评'和'改进建议'。请使用适当的标题和项目符号，使报告易于阅读。\n\n已分类的反馈要点（JSON格式）：\n{classified_points}\n\n客户反馈摘要报告："
)

# --- 3. 构建提示词链 ---

# 链条 1：提取要点
extraction_chain = prompt_extract | llm | StrOutputParser()

# 链条 2：分类要点
classification_chain = {"key_points": extraction_chain} | prompt_classify | llm | StrOutputParser()

# 完整链条：生成最终报告
full_chain = {"classified_points": classification_chain} | prompt_summarize | llm | StrOutputParser()


# --- 4. 运行链条 ---

# 示例客户反馈文本
customer_feedback = """
这款新出的手机真的很棒！屏幕色彩鲜艳，拿在手里的质感也超乎预期，非常高级。
不过，电池续航能力有点差，我一天下来需要充两次电，希望能改进。
另外，相机在夜间的表现不太理想，噪点比较多。
如果系统能增加一个应用双开功能就完美了，这对我的工作很有帮助。
总的来说，除了电池和相机，其他方面我都非常满意。
"""

# 调用完整的链条
final_report = full_chain.invoke({"feedback_text": customer_feedback})

# --- 5. 打印最终结果 ---
print("\n--- 客户反馈分析报告 ---")
print(final_report)
