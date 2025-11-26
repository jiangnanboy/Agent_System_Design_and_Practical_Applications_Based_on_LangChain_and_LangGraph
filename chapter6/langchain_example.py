from typing import Dict
import json
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

# 初始化llm
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
planning_prompt = PromptTemplate(
    input_variables=["destination", "duration", "budget", "interests", "travel_date"],
    template=planning_template
)

adjustment_prompt = PromptTemplate(
    input_variables=["original_plan", "adjustment_request"],
    template=adjustment_template
)

# 使用LCEL语法创建链
planning_chain = planning_prompt | llm | TravelPlanParser()
adjustment_chain = adjustment_prompt | llm | TravelPlanParser()

# 创建记忆组件
memory = ConversationBufferMemory()


# 定义旅行规划Agent类
class TravelPlannerAgent:
    def __init__(self):
        self.current_plan = None
        self.planning_chain = planning_chain
        self.adjustment_chain = adjustment_chain
        self.memory = memory

    def create_plan(self, destination, duration, budget, interests, travel_date):
        """创建初始旅行计划"""
        response = self.planning_chain.invoke({
            "destination": destination,
            "duration": duration,
            "budget": budget,
            "interests": interests,
            "travel_date": travel_date
        })

        self.current_plan = response
        self.memory.save_context(
            {"input": f"创建旅行计划到{destination}，时长{duration}天，预算{budget}"},
            {"output": str(response)}
        )

        return response

    def adjust_plan(self, adjustment_request):
        """根据新情况调整计划"""
        if not self.current_plan:
            return "没有可调整的计划，请先创建计划。"

        response = self.adjustment_chain.invoke({
            "original_plan": str(self.current_plan),
            "adjustment_request": adjustment_request
        })

        self.current_plan = response
        self.memory.save_context(
            {"input": f"调整计划：{adjustment_request}"},
            {"output": str(response)}
        )

        return response

    def get_current_plan(self):
        """获取当前计划"""
        return self.current_plan


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