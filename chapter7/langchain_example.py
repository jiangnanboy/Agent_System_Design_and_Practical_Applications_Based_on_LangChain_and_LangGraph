from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 初始化llm ---
from init_client import init_llm

llm = init_llm(0.5)


def main():
    """
    使用 DeepSeek 模型和 LangChain，模拟一个为小学生生成个性化学习计划的教学团队。
    """
    # 1. 学情分析师：定义其提示模板
    analyst_prompt = ChatPromptTemplate.from_template("""
    你是一位经验丰富的小学教育专家。请仔细分析以下学生档案，并找出该学生在数学学科上最需要关注的2-3个知识薄弱点。
    学生档案：
    {student_profile}

    请以清晰、简洁的要点形式列出这些薄弱点，并简要说明理由。
    """)

    # 2. 教学资源研究员：定义其提示模板
    # 它的输入将是上一个环节的输出
    researcher_prompt = ChatPromptTemplate.from_template("""
    你是一位精通儿童心理和教学资源的研究员。针对以下小学生数学的薄弱点，请为他推荐3-4种不同类型的、有趣且免费的学习资源。

    知识薄弱点：
    {analysis}

    资源类型可以包括：趣味教学视频、在线互动数学游戏、可打印的练习题等。请为每个资源提供简短的描述和为什么它适合这个孩子。
    """)

    # 3. 学习计划设计师：定义其提示模板
    designer_prompt = ChatPromptTemplate.from_template("""
    你是一位富有创造力的课程设计师。请根据以下教学资源，为该学生设计一个为期一周（周一至周五）的数学提升计划。

    可用教学资源：
    {resources}

    计划要求：
    1. 每天学习时间不超过20分钟。
    2. 每天的活动应富于变化，避免枯燥。
    3. 计划应以鼓励和引导为主，而非强制任务。
    4. 请以清晰的每日任务列表形式呈现。
    """)

    # 4. 家长沟通专员：定义其提示模板
    communicator_prompt = ChatPromptTemplate.from_template("""
    你是一位善于与家长沟通的学校顾问。请将以下专业的一周学习计划，转化为一封温暖、清晰且易于理解的信，发送给学生的家长。

    一周学习计划：
    {plan}

    在信中，请：
    1. 首先肯定孩子的努力，并说明制定此计划的初衷。
    2. 用通俗的语言解释计划内容。
    3. 给予家长一些如何陪伴和鼓励孩子的建议。
    4. 全文保持积极、鼓励的语气。
    """)

    # 输出解析器，将模型的输出转换为纯字符串，方便下一个环节使用
    output_parser = StrOutputParser()

    # 这条链清晰地展示了数据如何从一个智能体流向下一个智能体
    # prompt | model | parser 是一个经典的 LCEL 模式
    overall_chain = (
            analyst_prompt | llm | output_parser |
            {"analysis": lambda x: x} | researcher_prompt | llm | output_parser |
            {"resources": lambda x: x} | designer_prompt | llm | output_parser |
            {"plan": lambda x: x} | communicator_prompt | llm | output_parser
    )

    # 初始输入：模拟的学生档案
    student_profile = """
    学生姓名：小明
    年级：小学二年级
    近期数学表现：期中考试中，关于“1到9的乘除法应用题”的题目失分较多。课堂练习时，对复杂的应用题理解较慢，但基础计算能力尚可。性格活泼，喜欢玩游戏和看动画片。
    """

    # 执行整个教学团队协作流程
    print("## 启动 LCEL 驱动的个性化教学支持团队... ##")
    try:
        # 使用 .invoke() 方法来启动链，并传入初始输入
        final_letter = overall_chain.invoke({"student_profile": student_profile})
        print("\n------------------\n")
        print("## 给家长的最终信件 ##")
        print(final_letter)
    except Exception as e:
        print(f"\n发生意外错误：{e}")


if __name__ == "__main__":
    main()