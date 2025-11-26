from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



# 初始化模型
from init_client import init_llm

llm = init_llm(0.1)

def generate_data_analysis_report():
    """
    使用反思模式生成高质量数据分析报告
    """
    # 定义核心任务
    task_prompt = """
    你是一名数据分析师，需要基于以下销售数据生成一份简洁而全面的分析报告：

    数据：
    - 产品A：Q1销售额120万，Q2销售额150万，Q3销售额180万，Q4销售额210万
    - 产品B：Q1销售额80万，Q2销售额75万，Q3销售额90万，Q4销售额85万
    - 产品C：Q1销售额200万，Q2销售额220万，Q3销售额195万，Q4销售额240万

    报告应包括：
    1. 整体销售趋势分析
    2. 各产品表现对比
    3. 季度增长/下降率
    4. 关键洞察和建议
    """

    # 生产者提示模板
    producer_template = """
    {task}

    {feedback}

    请根据以上要求生成数据分析报告：
    """

    # 评审者提示模板
    reviewer_template = """
    你是一名资深数据分析专家，负责评估数据分析报告的质量。

    原始任务要求：
    {task}

    待评估报告：
    {report}

    请评估报告是否满足以下标准：
    1. 数据准确性
    2. 分析深度
    3. 洞察价值
    4. 结构清晰度
    5. 语言专业性

    如果报告完美无缺，请回复"REPORT_IS_PERFECT"。
    否则，请提供具体改进建议，以项目符号形式列出：
    """

    # 创建生产者和评审者链
    producer_prompt = PromptTemplate(template=producer_template, input_variables=["task", "feedback"])
    producer_chain = producer_prompt | llm | StrOutputParser()

    reviewer_prompt = PromptTemplate(template=reviewer_template, input_variables=["task", "report"])
    reviewer_chain = reviewer_prompt | llm | StrOutputParser()

    # 初始化变量
    current_report = ""
    feedback = "请生成初始报告。"
    max_iterations = 3

    # 反思循环
    for i in range(max_iterations):
        print(f"\n{'=' * 30} 反思循环：迭代 {i + 1} {'=' * 30}")

        # 生产者生成/优化报告
        print("\n>>> 阶段1：生成/优化报告...")
        current_report = producer_chain.invoke({"task": task_prompt, "feedback": feedback})

        print(f"\n--- 生成的报告 (v{i + 1}) ---\n{current_report}")

        # 评审者评估报告
        print("\n>>> 阶段2：评估报告质量...")
        review = reviewer_chain.invoke({"task": task_prompt, "report": current_report})

        # 检查停止条件
        if "REPORT_IS_PERFECT" in review:
            print("\n--- 评估结果 ---\n报告质量满意，无需进一步改进。")
            break

        print(f"\n--- 评估结果 ---\n{review}")

        # 更新反馈用于下一轮迭代
        feedback = f"请根据以下评估结果优化报告：\n{review}"

    print(f"\n{'=' * 35} 最终结果 {'=' * 35}")
    print("\n经过反思过程优化的最终数据分析报告：\n")
    print(current_report)


if __name__ == "__main__":
    generate_data_analysis_report()