import json
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from init_client import init_llm

# --- 数据模型和状态定义保持不变 ---
class UserProfile(BaseModel):
    interests: List[str] = Field(description="用户主要兴趣领域列表")
    price_sensitivity: str = Field(description="价格敏感度(高/中/低)")
    preferred_categories: List[str] = Field(description="偏好商品类别列表")
    behavior_patterns: str = Field(description="行为模式描述")
    recent_focus: List[str] = Field(description="最近关注点")


class RecommendedItem(BaseModel):
    product_id: str = Field(description="商品的唯一标识符")
    name: str = Field(description="商品的名称")
    reason: str = Field(description="推荐该商品的理由")


class RecommendationResult(BaseModel):
    recommendations: List[RecommendedItem] = Field(description="推荐的商品列表")


class RecommendationState(TypedDict):
    user_id: str
    user_history: List[Dict[str, Any]]
    user_demographics: Dict[str, Any]
    user_profile: Optional[Dict[str, Any]]
    products: List[Dict[str, Any]]
    context: str
    recommendations: List[Dict[str, Any]]
    feedback: Optional[Dict[str, Any]]
    messages: Annotated[List, "messages"]


class AdaptiveRecommendationAgent:
    def __init__(self):
        self.llm = init_llm(temperature=0.7)
        self.profile_parser = JsonOutputParser(pydantic_object=UserProfile)
        self.recommendation_parser = JsonOutputParser(pydantic_object=RecommendationResult)
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        # 可选：可视化图的结构
        self.app.get_graph().print_ascii()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(RecommendationState)
        workflow.add_node("build_profile", self._build_user_profile)
        workflow.add_node("generate_recommendations", self._generate_recommendations)
        workflow.add_node("process_feedback", self._update_profile_based_on_feedback)
        workflow.set_entry_point("build_profile")
        workflow.add_edge("build_profile", "generate_recommendations")
        workflow.add_conditional_edges("generate_recommendations", self._should_process_feedback,
                                       {"process_feedback": "process_feedback", "end": END})
        workflow.add_edge("process_feedback", "generate_recommendations")
        return workflow

    def _should_process_feedback(self, state: RecommendationState) -> str:
        return "process_feedback" if state.get("feedback") is not None else "end"

    def _build_user_profile(self, state: RecommendationState) -> RecommendationState:
        if state.get("user_profile") is not None:
            state["messages"].append(AIMessage(content="用户画像已存在，跳过构建。"))
            return state

        history_text = "\n".join(
            [f"浏览商品: {item['product_name']}, 类别: {item['category']}, 时长: {item['duration']}秒, 是否购买: {item['purchased']}"
             for item in state["user_history"]])
        demographics_text = f"年龄: {state['user_demographics'].get('age', '未知')}, 性别: {state['user_demographics'].get('gender', '未知')}, 地理位置: {state['user_demographics'].get('location', '未知')}, 会员等级: {state['user_demographics'].get('membership_level', '未知')}"
        profile_prompt = ChatPromptTemplate.from_messages([("system", "你是一个专业的用户画像分析师。"), (
        "human", "基于以下信息构建用户画像:\n用户历史行为:\n{user_history}\n人口统计信息:\n{user_demographics}\n{format_instructions}")])
        profile_chain = profile_prompt | self.llm | self.profile_parser
        try:
            user_profile = profile_chain.invoke({"user_history": history_text, "user_demographics": demographics_text,
                                                 "format_instructions": self.profile_parser.get_format_instructions()})
            state["user_profile"] = user_profile
            state["messages"].append(AIMessage(content=f"已成功构建用户画像: {json.dumps(user_profile, ensure_ascii=False)}"))
        except Exception as e:
            print(f"解析用户画像时出错: {e}")
            state["user_profile"] = {"interests": [], "price_sensitivity": "中", "preferred_categories": [],
                                     "behavior_patterns": "无法解析用户画像", "recent_focus": []}
            state["messages"].append(AIMessage(content=f"构建用户画像时出错: {str(e)}，使用默认画像"))
        return state

    def _generate_recommendations(self, state: RecommendationState) -> RecommendationState:
        if not state.get("user_profile"):
            raise ValueError("FATAL: 用户画像不存在，无法生成推荐。这表明工作流存在严重问题。")

        all_products_text = "\n\n".join([
                                            f"商品ID: {p['id']}, 名称: {p['name']}, 类别: {p['category']}, 价格: {p['price']}, 品牌: {p['brand']}, 描述: {p['description']}, 特性: {', '.join(p['features'])}, 评分: {p['rating']}"
                                            for p in state["products"]])
        recommendation_prompt = ChatPromptTemplate.from_messages([("system", "你是一个顶级的商品推荐专家。"), ("human",
                                                                                                 "根据用户画像和场景推荐商品:\n商品列表:\n{all_products_text}\n用户画像:\n{user_profile}\n场景:\n{context}\n{format_instructions}")])
        recommendation_chain = recommendation_prompt | self.llm | self.recommendation_parser
        try:
            result = recommendation_chain.invoke({"all_products_text": all_products_text,
                                                  "user_profile": json.dumps(state["user_profile"], ensure_ascii=False),
                                                  "context": state.get("context", ""), "num_recommendations": 5,
                                                  "format_instructions": self.recommendation_parser.get_format_instructions()})
            recommendations = result['recommendations']
            state["recommendations"] = [rec for rec in recommendations]
            state["messages"].append(AIMessage(content=f"已生成推荐: {json.dumps(recommendations, ensure_ascii=False)}"))
        except Exception as e:
            print(f"生成推荐时出错: {e}")
            state["recommendations"] = []
            state["messages"].append(AIMessage(content=f"生成推荐时出错: {str(e)}"))
        return state

    def _update_profile_based_on_feedback(self, state: RecommendationState) -> RecommendationState:
        if not state.get("feedback"): return state

        last_recommendations = state.get("recommendations", [])
        feedback_text = "用户对推荐的反馈:\n"
        for item_id, feedback_item in state["feedback"].items():
            item_name = next((rec["name"] for rec in last_recommendations if rec["product_id"] == item_id), "未知商品")
            feedback_text += f"\n商品 {item_name} (ID: {item_id}): {feedback_item['rating']}分, 评论: {feedback_item['comment']}"

        feedback_prompt = ChatPromptTemplate.from_messages([("system", "你是一个专业的用户画像分析师，擅长根据用户反馈更新用户画像。"), ("human",
                                                                                                           "基于反馈更新用户画像:\n当前画像:\n{user_profile}\n推荐商品:\n{recommended_items}\n用户反馈:\n{user_feedback}\n{format_instructions}")])
        feedback_chain = feedback_prompt | self.llm | self.profile_parser

        try:
            updated_profile = feedback_chain.invoke(
                {"user_profile": json.dumps(state["user_profile"], ensure_ascii=False),
                 "recommended_items": json.dumps(last_recommendations, ensure_ascii=False),
                 "user_feedback": feedback_text, "format_instructions": self.profile_parser.get_format_instructions()})
            state["user_profile"] = updated_profile
            state["messages"].append(
                AIMessage(content=f"已根据反馈更新用户画像: {json.dumps(updated_profile, ensure_ascii=False)}"))
        except Exception as e:
            print(f"处理反馈时出错: {e}")
            state["messages"].append(AIMessage(content=f"处理反馈时出错: {str(e)}"))

        state["feedback"] = None
        return state

    # --- 修正 run 方法的函数签名 ---
    def run(self,
            user_id: str,
            context: str,
            feedback: Optional[Dict[str, Any]] = None,
            thread_id: str = "default",
            # 将初始化参数设为可选 ---
            user_history: Optional[List[Dict[str, Any]]] = None,
            user_demographics: Optional[Dict[str, Any]] = None,
            products: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        运行推荐系统。
        第一次调用时，必须提供 user_history, user_demographics, products。
        后续调用时，只需提供 context 和可选的 feedback。
        """

        input_state = {
            "context": context,
            "feedback": feedback
        }

        config = {"configurable": {"thread_id": thread_id}}
        current_checkpoint = self.memory.get(config)

        if not current_checkpoint:
            # --- 增加健壮性检查 ---
            if not all([user_history, user_demographics, products]):
                raise ValueError("为新线程进行初始化时，必须提供 user_history, user_demographics, 和 products。")

            # 如果是新线程，提供所有初始化数据
            input_state.update({
                "user_id": user_id,
                "user_history": user_history,
                "user_demographics": user_demographics,
                "products": products,
                "messages": []
            })

        result = self.app.invoke(input_state, config)
        return result


# --- 示例使用 ---
if __name__ == "__main__":
    agent = AdaptiveRecommendationAgent()

    products = [
        {"id": "p001", "name": "智能手表 Pro", "category": "电子产品", "price": 2999, "brand": "TechBrand",
         "description": "高端智能手表，支持健康监测、运动追踪和智能通知", "features": ["心率监测", "GPS定位", "防水设计", "7天续航"], "rating": 4.7},
        {"id": "p002", "name": "有机绿茶礼盒", "category": "食品饮料", "price": 168, "brand": "NaturePure",
         "description": "精选高山有机绿茶，清香甘醇，富含抗氧化物质", "features": ["有机认证", "礼盒包装", "产地直供", "传统工艺"], "rating": 4.8},
        {"id": "p003", "name": "多功能背包", "category": "箱包", "price": 459, "brand": "TravelPro",
         "description": "大容量多功能背包，适合通勤和短途旅行", "features": ["防水面料", "USB充电口", "防盗设计", "人体工学"], "rating": 4.5},
        {"id": "p004", "name": "编程入门教程", "category": "图书", "price": 89, "brand": "TechBooks",
         "description": "零基础学编程，包含Python和JavaScript实例", "features": ["零基础友好", "实例丰富", "配套视频", "在线答疑"], "rating": 4.6},
        {"id": "p005", "name": "无线降噪耳机", "category": "电子产品", "price": 1299, "brand": "AudioTech",
         "description": "主动降噪无线耳机，高保真音质，长续航", "features": ["主动降噪", "40小时续航", "快充技术", "多设备连接"], "rating": 4.9},
        {"id": "p006", "name": "瑜伽垫套装", "category": "运动健身", "price": 299, "brand": "FitLife",
         "description": "高密度环保瑜伽垫，防滑耐用，含瑜伽砖和拉力带", "features": ["环保材质", "防滑设计", "便携收纳", "套装组合"], "rating": 4.4},
        {"id": "p007", "name": "智能家居套装", "category": "智能家居", "price": 1599, "brand": "SmartHome",
         "description": "包含智能音箱、智能灯泡和智能插座，一键控制家居", "features": ["语音控制", "远程操作", "场景联动", "定时任务"], "rating": 4.7},
        {"id": "p008", "name": "精酿啤酒组合", "category": "食品饮料", "price": 258, "brand": "CraftBrew",
         "description": "精选6款不同风格的精酿啤酒，口感丰富多样", "features": ["多种口味", "限量酿造", "礼盒包装", "品鉴指南"], "rating": 4.6}
    ]

    user_id = "user123"
    user_history = [
        {"product_name": "智能手表 Pro", "category": "电子产品", "duration": 120, "purchased": True},
        {"product_name": "无线降噪耳机", "category": "电子产品", "duration": 95, "purchased": True},
        {"product_name": "智能家居套装", "category": "智能家居", "duration": 150, "purchased": False},
        {"product_name": "编程入门教程", "category": "图书", "duration": 60, "purchased": True},
        {"product_name": "多功能背包", "category": "箱包", "duration": 45, "purchased": False}
    ]
    user_demographics = {"age": 28, "gender": "男", "location": "北京", "membership_level": "黄金会员"}

    # --- 第一次运行：初始化 ---
    print("=== 第一次运行：初始化 ===")
    result1 = agent.run(
        user_id=user_id,
        user_history=user_history,  # 提供初始化数据
        user_demographics=user_demographics,  # 提供初始化数据
        products=products,  # 提供初始化数据
        context="新用户首次交互",
        thread_id=user_id
    )
    print("用户画像:", json.dumps(result1["user_profile"], ensure_ascii=False, indent=2))
    print("\n推荐商品:")
    for i, rec in enumerate(result1["recommendations"], 1):
        print(f"{i}. 商品ID: {rec.get('product_id')}, 名称: {rec.get('name')}")

    # --- 第二次运行：新上下文 ---
    print("\n=== 第二次运行：新上下文 ===")
    result2 = agent.run(
        user_id=user_id,  # 不再需要其他信息
        context="用户即将出差，需要购买一些旅行用品",
        thread_id=user_id
    )
    print("\n新推荐商品:")
    for i, rec in enumerate(result2["recommendations"], 1):
        print(f"{i}. 商品ID: {rec.get('product_id')}, 名称: {rec.get('name')}")

    # --- 第三次运行：处理反馈 ---
    print("\n=== 第三次运行：处理反馈 ===")
    feedback = {"p001": {"rating": 5, "comment": "非常符合我的需求，已经购买"}}
    result3 = agent.run(
        user_id=user_id,
        context="基于用户反馈的二次推荐",
        feedback=feedback,
        thread_id=user_id
    )
    print("\n更新后的用户画像:", json.dumps(result3["user_profile"], ensure_ascii=False, indent=2))
    print("\n新推荐商品:")
    for i, rec in enumerate(result3["recommendations"], 1):
        print(f"{i}. 商品ID: {rec.get('product_id')}, 名称: {rec.get('name')}")