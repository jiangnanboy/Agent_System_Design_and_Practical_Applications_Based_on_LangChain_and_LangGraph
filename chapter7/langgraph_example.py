from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START


# --- 初始化llm ---
from init_client import init_llm

llm = init_llm(0.5)

# 1. 定义整个工作流共享的状态
class EducationState(TypedDict):
    """定义在整个图的工作流中传递的状态。"""
    student_profile: str
    analysis_report: str
    resource_list: str
    learning_plan: str
    final_letter: str

# --- 定义每个智能体（节点）的逻辑 ---

# 学情分析师节点
analyst_prompt = ChatPromptTemplate.from_template("""
你是一位经验丰富的小学教育专家。请仔细分析以下学生档案，并找出该学生在数学学科上最需要关注的2-3个知识薄弱点。
学生档案：
{student_profile}

请以清晰、简洁的要点形式列出这些薄弱点，并简要说明理由。
""")
analyst_chain = analyst_prompt | llm | StrOutputParser()

def analyst_node(state: EducationState):
    """分析师节点：分析学生情况并更新状态。"""
    print("--- 节点：学情分析师正在工作 ---")
    analysis = analyst_chain.invoke({"student_profile": state['student_profile']})
    return {"analysis_report": analysis}

# 教学资源研究员节点
researcher_prompt = ChatPromptTemplate.from_template("""
你是一位精通儿童心理和教学资源的研究员。针对以下小学生数学的薄弱点，请为他推荐3-4种不同类型的、有趣且免费的学习资源。

知识薄弱点：
{analysis_report}

资源类型可以包括：趣味教学视频、在线互动数学游戏、可打印的练习题等。请为每个资源提供简短的描述和为什么它适合这个孩子。
""")
researcher_chain = researcher_prompt | llm | StrOutputParser()

def researcher_node(state: EducationState):
    """研究员节点：查找资源并更新状态。"""
    print("--- 节点：教学资源研究员正在工作 ---")
    resources = researcher_chain.invoke({"analysis_report": state['analysis_report']})
    return {"resource_list": resources}

# 学习计划设计师节点
designer_prompt = ChatPromptTemplate.from_template("""
你是一位富有创造力的课程设计师。请根据以下教学资源，为该学生设计一个为期一周（周一至周五）的数学提升计划。

可用教学资源：
{resource_list}

计划要求：
1. 每天学习时间不超过20分钟。
2. 每天的活动应富于变化，避免枯燥。
3. 计划应以鼓励和引导为主，而非强制任务。
4. 请以清晰的每日任务列表形式呈现。
""")
designer_chain = designer_prompt | llm | StrOutputParser()

def designer_node(state: EducationState):
    """设计师节点：制定计划并更新状态。"""
    print("--- 节点：学习计划设计师正在工作 ---")
    plan = designer_chain.invoke({"resource_list": state['resource_list']})
    return {"learning_plan": plan}

# 家长沟通专员节点
communicator_prompt = ChatPromptTemplate.from_template("""
你是一位善于与家长沟通的学校顾问。请将以下专业的一周学习计划，转化为一封温暖、清晰且易于理解的信，发送给学生的家长。

一周学习计划：
{learning_plan}

在信中，请：
1. 首先肯定孩子的努力，并说明制定此计划的初衷。
2. 用通俗的语言解释计划内容。
3. 给予家长一些如何陪伴和鼓励孩子的建议。
4. 全文保持积极、鼓励的语气。
""")
communicator_chain = communicator_prompt | llm | StrOutputParser()

def communicator_node(state: EducationState):
    """沟通专员节点：撰写信件并更新状态。"""
    print("--- 节点：家长沟通专员正在工作 ---")
    letter = communicator_chain.invoke({"learning_plan": state['learning_plan']})
    return {"final_letter": letter}

if __name__ == "__main__":
    # --- 使用 LangGraph 构建工作流 ---

    # 创建一个基于我们定义状态的图
    workflow = StateGraph(EducationState)

    # 将节点（智能体）添加到图中
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("designer", designer_node)
    workflow.add_node("communicator", communicator_node)

    # 定义边，即工作流的执行顺序
    workflow.add_edge(START, "analyst")
    workflow.add_edge("analyst", "researcher")
    workflow.add_edge("researcher", "designer")
    workflow.add_edge("designer", "communicator")
    workflow.add_edge("communicator", END)

    # 编译图
    app = workflow.compile()
    # 可选：可视化图的结构
    app.get_graph().print_ascii()

    # 初始输入：模拟的学生档案
    initial_state = {
        "student_profile": """
        学生姓名：小明
        年级：小学二年级
        近期数学表现：期末考试中，关于“1到9的乘除法应用题”的题目失分较多。课堂练习时，对复杂的应用题理解较慢，但基础计算能力尚可。性格活泼，喜欢玩游戏和看动画片。
        """
    }

    # 执行整个图
    print("## 启动 LangGraph 驱动的个性化教学支持团队... ##")
    try:
        final_state = app.invoke(initial_state)
        print("\n------------------\n")
        print("## 给家长的最终信件 ##")
        print(final_state['final_letter'])
    except Exception as e:
        print(f"\n发生意外错误：{e}")
