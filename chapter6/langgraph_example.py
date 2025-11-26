from typing import Dict, List, Optional, TypedDict, Annotated, Literal
import json
import operator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 初始化LLM
from init_client import init_llm

llm = init_llm(0.7)


# 定义输出解析器，将模型输出转换为结构化数据
class TravelPlanParser(BaseOutputParser):
    def parse(self, text: str) -> Dict:
        try:
            # 尝试提取JSON部分
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # 如果没有找到JSON，返回原始文本
                return {"plan": text}
        except Exception as e:
            print(f"解析错误: {e}")
            return {"plan": text}


# 定义用户意图解析结果
class UserIntent(TypedDict):
    intent: Literal["create", "adjust", "unknown"]
    destination: Optional[str]
    duration: Optional[str]
    budget: Optional[str]
    interests: Optional[str]
    travel_date: Optional[str]
    adjustment_request: Optional[str]


# 定义用户意图解析器
class UserIntentParser(JsonOutputParser):
    def parse(self, text: str) -> UserIntent:
        try:
            result = super().parse(text)
            # 确保所有必需的字段都存在
            if "intent" not in result:
                result["intent"] = "unknown"
            return result
        except Exception as e:
            print(f"用户意图解析错误: {e}")
            return {"intent": "unknown"}


# 定义Agent状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_plan: Optional[Dict]
    destination: Optional[str]
    duration: Optional[str]
    budget: Optional[str]
    interests: Optional[str]
    travel_date: Optional[str]
    adjustment_request: Optional[str]
    user_intent: Optional[str]


# 创建用户意图解析模板
intent_extraction_template = """
你是一个旅行规划助手，需要解析用户的输入，确定他们的意图并提取相关信息。

用户输入：{user_input}

请分析用户输入，确定是创建新的旅行计划还是调整现有计划，并提取相关信息。

请以JSON格式返回结果，包含以下字段：
- intent: 用户意图，只能是 "create"（创建新计划）或 "adjust"（调整现有计划）或 "unknown"（无法确定）
- destination: 目的地（如果是创建计划）
- duration: 旅行时长（如果是创建计划）
- budget: 预算（如果是创建计划）
- interests: 兴趣偏好（如果是创建计划）
- travel_date: 出行时间（如果是创建计划）
- adjustment_request: 调整请求的具体内容（如果是调整计划）

示例：
输入："我想去巴黎玩5天，预算1万元，喜欢艺术和美食，下个月出发"
输出：{{"intent": "create", "destination": "巴黎", "duration": "5天", "budget": "1万元", "interests": "艺术和美食", "travel_date": "下个月", "adjustment_request": null}}

输入："把预算减少到8000元，增加一天行程"
输出：{{"intent": "adjust", "destination": null, "duration": null, "budget": null, "interests": null, "travel_date": null, "adjustment_request": "把预算减少到8000元，增加一天行程"}}
"""

# 创建规划模板
planning_template = """
你是一位专业的旅行规划师，擅长根据客户需求创建详细的旅行计划。

客户需求：
- 目的地：{destination}
- 旅行时长：{duration}
- 预算：{budget}
- 兴趣偏好：{interests}
- 出行时间：{travel_date}

请创建一个详细的旅行计划，包括：
1. 每日行程安排
2. 推荐景点和活动
3. 餐饮建议
4. 交通方案
5. 预算分配

请以JSON格式返回计划，结构如下：
{{
  "daily_itinerary": [
    {{
      "day": 1,
      "activities": ["活动1", "活动2"],
      "meals": ["早餐建议", "午餐建议", "晚餐建议"],
      "transportation": "当日交通方案"
    }}
  ],
  "budget_breakdown": {{
    "accommodation": "预算金额",
    "food": "预算金额",
    "activities": "预算金额",
    "transportation": "预算金额"
  }},
  "general_tips": ["旅行提示1", "旅行提示2"]
}}
"""

# 创建调整模板
adjustment_template = """
根据新的情况调整旅行计划：

原始计划：
{original_plan}

新情况/调整需求：
{adjustment_request}

请提供调整后的旅行计划，保持相同的JSON格式。
"""

# 创建提示模板
intent_extraction_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=intent_extraction_template
)

planning_prompt = PromptTemplate(
    input_variables=["destination", "duration", "budget", "interests", "travel_date"],
    template=planning_template
)

adjustment_prompt = PromptTemplate(
    input_variables=["original_plan", "adjustment_request"],
    template=adjustment_template
)

# 创建意图解析链
intent_extraction_chain = intent_extraction_prompt | llm | UserIntentParser()


# 定义LangGraph节点函数
def extract_travel_info(state: AgentState):
    """从消息中提取旅行信息"""
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content

        # 使用LLM解析用户意图
        try:
            intent_result = intent_extraction_chain.invoke({"user_input": content})
            intent = intent_result.get("intent", "unknown")

            if intent == "create":
                # 创建新计划的请求
                return {
                    "user_intent": "create",
                    "destination": intent_result.get("destination"),
                    "duration": intent_result.get("duration"),
                    "budget": intent_result.get("budget"),
                    "interests": intent_result.get("interests"),
                    "travel_date": intent_result.get("travel_date"),
                    "adjustment_request": None
                }
            elif intent == "adjust":
                # 调整计划的请求
                return {
                    "user_intent": "adjust",
                    "adjustment_request": intent_result.get("adjustment_request"),
                    "destination": None,
                    "duration": None,
                    "budget": None,
                    "interests": None,
                    "travel_date": None
                }
            else:
                # 无法确定意图
                return {
                    "user_intent": "unknown",
                    "messages": [AIMessage(content="抱歉，我不太理解您的需求。请明确说明是创建新的旅行计划还是调整现有计划。")]
                }
        except Exception as e:
            print(f"解析用户输入时出错: {e}")
            return {
                "user_intent": "unknown",
                "messages": [AIMessage(content="抱歉，解析您的请求时遇到了问题。请再试一次。")]
            }

    return state


def create_travel_plan(state: AgentState):
    """创建旅行计划"""
    response = planning_prompt | llm | TravelPlanParser()
    result = response.invoke({
        "destination": state.get("destination", ""),
        "duration": state.get("duration", ""),
        "budget": state.get("budget", ""),
        "interests": state.get("interests", ""),
        "travel_date": state.get("travel_date", "")
    })

    return {
        "current_plan": result,
        "messages": [AIMessage(content=f"已创建旅行计划：{json.dumps(result, indent=2, ensure_ascii=False)}")]
    }


def adjust_travel_plan(state: AgentState):
    """调整旅行计划"""
    if not state.get("current_plan"):
        return {
            "messages": [AIMessage(content="没有可调整的计划，请先创建计划。")]
        }

    response = adjustment_prompt | llm | TravelPlanParser()
    result = response.invoke({
        "original_plan": json.dumps(state["current_plan"], indent=2, ensure_ascii=False),
        "adjustment_request": state.get("adjustment_request", "")
    })

    return {
        "current_plan": result,
        "messages": [AIMessage(content=f"已调整旅行计划：{json.dumps(result, indent=2, ensure_ascii=False)}")]
    }


def handle_unknown_intent(state: AgentState):
    """处理未知意图"""
    return {
        "messages": [AIMessage(content="抱歉，我不太理解您的需求。请明确说明是创建新的旅行计划还是调整现有计划。")]
    }


def route_request(state: AgentState):
    """路由请求到适当的节点"""
    intent = state.get("user_intent", "unknown")
    if intent == "create":
        return "create_plan"
    elif intent == "adjust":
        return "adjust_plan"
    else:
        return "handle_unknown"


# 构建LangGraph
def build_travel_planner_graph():
    """构建旅行规划图"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("extract_info", extract_travel_info)
    workflow.add_node("create_plan", create_travel_plan)
    workflow.add_node("adjust_plan", adjust_travel_plan)
    workflow.add_node("handle_unknown", handle_unknown_intent)

    # 设置入口点
    workflow.set_entry_point("extract_info")

    # 添加条件边
    workflow.add_conditional_edges(
        "extract_info",
        route_request,
        {
            "create_plan": "create_plan",
            "adjust_plan": "adjust_plan",
            "handle_unknown": "handle_unknown"
        }
    )

    # 添加结束边
    workflow.add_edge("create_plan", END)
    workflow.add_edge("adjust_plan", END)
    workflow.add_edge("handle_unknown", END)

    # 添加内存
    memory = MemorySaver()

    return workflow.compile(checkpointer=memory)


# 定义旅行规划Agent类
class TravelPlannerAgent:
    def __init__(self):
        self.graph = build_travel_planner_graph()
        # 打印图的结构（可选，非常直观！）
        try:
            print("--- 图结构 ---")
            self.graph.get_graph().print_ascii()
            print("\n" + "=" * 20 + "\n")
        except Exception as e:
            print(f"无法打印图结构: {e}")

        self.config = {"configurable": {"thread_id": "travel_planner"}}
        self.current_plan = None

    def create_plan(self, destination, duration, budget, interests, travel_date):
        """创建初始旅行计划"""
        # 构建用户消息
        user_message = f"创建旅行计划到{destination}，时长{duration}天，预算{budget}，兴趣偏好：{interests}，出行时间：{travel_date}"

        # 运行图
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "destination": destination,
                "duration": duration,
                "budget": budget,
                "interests": interests,
                "travel_date": travel_date
            },
            config=self.config
        )

        self.current_plan = result.get("current_plan")
        return self.current_plan

    def adjust_plan(self, adjustment_request):
        """根据新情况调整计划"""
        # 构建用户消息
        user_message = f"调整计划：{adjustment_request}"

        # 运行图
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "adjustment_request": adjustment_request,
                "current_plan": self.current_plan
            },
            config=self.config
        )

        self.current_plan = result.get("current_plan")
        return self.current_plan

    def get_current_plan(self):
        """获取当前计划"""
        return self.current_plan

    def chat(self, user_input):
        """与Agent对话"""
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config=self.config
        )

        self.current_plan = result.get("current_plan")
        return result["messages"][-1].content


# 使用示例
if __name__ == "__main__":
    # 初始化旅行规划Agent
    agent = TravelPlannerAgent()

    # 创建初始计划
    print("## 创建初始旅行计划 ##")
    initial_plan = agent.create_plan(
        destination="中国北京",
        duration="5天",
        budget="10000元",
        interests="传统文化、美食、古迹",
        travel_date="2026年1月"
    )

    print("初始计划:")
    print(json.dumps(initial_plan, indent=2, ensure_ascii=False))

    # 调整计划
    print("\n## 调整旅行计划 ##")
    adjusted_plan = agent.adjust_plan("预算减少到8000元，并增加一天行程")

    print("调整后的计划:")
    print(json.dumps(adjusted_plan, indent=2, ensure_ascii=False))

    # 使用聊天接口 - 测试自然语言输入
    print("\n## 使用聊天接口 - 自然语言输入测试 ##")

    # 测试创建计划的自然语言输入
    response1 = agent.chat("我想去美国洛杉矶玩7天，预算15000元，喜欢好莱坞和美食，明年春天出发")
    print(response1)

    # 测试调整计划的自然语言输入
    response2 = agent.chat("把预算减少到12000元，增加一些购物时间")
    print(response2)