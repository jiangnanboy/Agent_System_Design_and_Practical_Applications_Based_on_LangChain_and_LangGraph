import json
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from init_client import init_llm

# 定义用户画像的数据结构
class UserProfile(BaseModel):
    interests: List[str] = Field(description="用户主要兴趣领域列表")
    price_sensitivity: str = Field(description="价格敏感度(高/中/低)")
    preferred_categories: List[str] = Field(description="偏好商品类别列表")
    behavior_patterns: str = Field(description="行为模式描述")
    recent_focus: List[str] = Field(description="最近关注点")


# --- 定义更具体的推荐项结构 ---
class RecommendedItem(BaseModel):
    """单个推荐商品的结构"""
    product_id: str = Field(description="商品的唯一标识符")
    name: str = Field(description="商品的名称")
    reason: str = Field(description="推荐该商品的理由")


# --- 使用新的结构定义推荐结果 ---
class RecommendationResult(BaseModel):
    """推荐结果的顶层结构"""
    recommendations: List[RecommendedItem] = Field(description="推荐的商品列表")


class AdaptiveRecommendationAgent:
    def __init__(self):
        # 初始化llm
        self.llm = init_llm(temperature=0.7)

        # 初始化用户记忆和简单的商品列表
        self.user_memory = {}
        self.products = []  # 简单存储商品列表

        # 初始化输出解析器
        self.profile_parser = JsonOutputParser(pydantic_object=UserProfile)
        # --- 使用新的、更具体的解析器 ---
        self.recommendation_parser = JsonOutputParser(pydantic_object=RecommendationResult)

    def load_products(self, products: List[Dict[str, Any]]):
        """加载商品数据到内存"""
        self.products = products
        print(f"已加载商品数据库，共{len(products)}件商品")

    def build_user_profile(self, user_id: str, user_history: List[Dict[str, Any]],
                           user_demographics: Dict[str, Any]) -> Dict[str, Any]:
        """使用LCEL链式调用构建用户画像"""
        history_text = "\n".join([
            f"浏览商品: {item['product_name']}, 类别: {item['category']}, "
            f"时长: {item['duration']}秒, 是否购买: {item['purchased']}"
            for item in user_history
        ])

        demographics_text = f"""
        年龄: {user_demographics.get('age', '未知')}
        性别: {user_demographics.get('gender', '未知')}
        地理位置: {user_demographics.get('location', '未知')}
        会员等级: {user_demographics.get('membership_level', '未知')}
        """

        profile_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的用户画像分析师。请根据用户的历史行为和人口统计信息，构建一个详细的用户画像。"),
            ("human", """
            基于以下用户历史行为和人口统计信息，构建一个详细的用户画像。
            请识别用户的兴趣偏好、消费习惯、价格敏感度等关键特征。

            用户历史行为:
            {user_history}

            用户人口统计信息:
            {user_demographics}

            {format_instructions}
            """)
        ])

        profile_chain = profile_prompt | self.llm | self.profile_parser

        try:
            user_profile = profile_chain.invoke({
                "user_history": history_text,
                "user_demographics": demographics_text,
                "format_instructions": self.profile_parser.get_format_instructions()
            })
        except Exception as e:
            print(f"解析用户画像时出错: {e}")
            user_profile = {
                "interests": [],
                "price_sensitivity": "中",
                "preferred_categories": [],
                "behavior_patterns": "无法解析用户画像",
                "recent_focus": []
            }

        self.user_memory[user_id] = {
            "profile": user_profile,
            "history": user_history,
            "demographics": user_demographics
        }

        return user_profile

    def generate_recommendations(self, user_id: str, context: str = "", num_recommendations: int = 3) -> List[
        Dict[str, Any]]:
        """让LLM直接从全量商品中生成推荐"""
        if user_id not in self.user_memory:
            raise ValueError(f"用户 {user_id} 的画像不存在，请先构建用户画像")

        user_profile = self.user_memory[user_id]["profile"]

        # 将所有商品信息转换为文本，提供给LLM
        all_products_text = "\n\n".join([
            f"商品ID: {p['id']}, 名称: {p['name']}, 类别: {p['category']}, 价格: {p['price']}, 品牌: {p['brand']}, 描述: {p['description']}, 特性: {', '.join(p['features'])}, 评分: {p['rating']}"
            for p in self.products
        ])

        recommendation_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个顶级的商品推荐专家。你的任务是从给定的商品列表中，为用户挑选最合适的几款商品，并给出令人信服的理由。"),
            ("human", """
            以下是当前可供推荐的所有商品列表：
            ---
            {all_products_text}
            ---

            现在，请根据以下用户画像和当前场景，为用户推荐 {num_recommendations} 件最合适的商品。

            用户画像:
            {user_profile}

            当前场景/上下文:
            {context}

            请仔细分析用户画像中的兴趣、偏好类别和价格敏感度，结合商品信息，做出最佳推荐。
            确保推荐理由充分、个性化，并能打动用户。

            {format_instructions}
            """)
        ])

        recommendation_chain = recommendation_prompt | self.llm | self.recommendation_parser

        try:
            result = recommendation_chain.invoke({
                "all_products_text": all_products_text,
                "user_profile": json.dumps(user_profile, ensure_ascii=False),
                "context": context,
                "num_recommendations": num_recommendations,
                "format_instructions": self.recommendation_parser.get_format_instructions()
            })
            # --- 从解析后的对象中正确提取列表 ---
            recommendations = result['recommendations']
        except Exception as e:
            print(f"生成推荐时出错: {e}")
            recommendations = []

        # 存储推荐历史
        if "recommendations" not in self.user_memory[user_id]:
            self.user_memory[user_id]["recommendations"] = []

        # --- 存储时将Pydantic对象转换为字典，以保持一致性 ---
        self.user_memory[user_id]["recommendations"].append({
            "context": context,
            "recommendations": [rec for rec in recommendations],  # 转换为字典列表
            "timestamp": "当前时间"
        })

        return [rec for rec in recommendations]  # 返回字典列表

    def process_feedback(self, user_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户反馈，更新用户画像"""
        if user_id not in self.user_memory:
            raise ValueError(f"用户 {user_id} 的画像不存在")

        user_profile = self.user_memory[user_id]["profile"]

        # --- 增加健壮性检查，确保有推荐历史 ---
        if not self.user_memory[user_id].get("recommendations"):
            print("没有找到推荐历史，无法处理反馈。")
            return user_profile

        last_recommendations = self.user_memory[user_id]["recommendations"][-1]["recommendations"]

        feedback_text = "用户对推荐的反馈:\n"
        for item_id, feedback_item in feedback.items():
            item_name = next((rec["name"] for rec in last_recommendations if rec["product_id"] == item_id), "未知商品")
            feedback_text += f"\n商品 {item_name} (ID: {item_id}): {feedback_item['rating']}分, 评论: {feedback_item['comment']}"

        feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的用户画像分析师，擅长根据用户反馈更新用户画像。"),
            ("human", """
            基于用户对推荐商品的反馈，更新用户画像。

            当前用户画像:
            {user_profile}

            推荐的商品:
            {recommended_items}

            用户反馈:
            {user_feedback}

            请分析用户反馈，更新用户画像，特别关注:
            1. 用户喜欢的商品特征
            2. 用户不喜欢的商品特征
            3. 新发现的兴趣点
            4. 偏好变化

            {format_instructions}
            """)
        ])

        feedback_chain = feedback_prompt | self.llm | self.profile_parser

        try:
            updated_profile = feedback_chain.invoke({
                "user_profile": json.dumps(user_profile, ensure_ascii=False),
                "recommended_items": json.dumps(last_recommendations, ensure_ascii=False),
                "user_feedback": feedback_text,
                "format_instructions": self.profile_parser.get_format_instructions()
            })
            self.user_memory[user_id]["profile"] = updated_profile
            return updated_profile
        except Exception as e:
            print(f"更新用户画像时出错: {e}")
            return user_profile


# 示例使用 (与之前相同)
if __name__ == "__main__":
    rec_system = AdaptiveRecommendationAgent()
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
    rec_system.load_products(products)

    user_id = "user123"
    user_history = [
        {"product_name": "智能手表 Pro", "category": "电子产品", "duration": 120, "purchased": True},
        {"product_name": "无线降噪耳机", "category": "电子产品", "duration": 95, "purchased": True},
        {"product_name": "智能家居套装", "category": "智能家居", "duration": 150, "purchased": False},
        {"product_name": "编程入门教程", "category": "图书", "duration": 60, "purchased": True},
        {"product_name": "多功能背包", "category": "箱包", "duration": 45, "purchased": False}
    ]
    user_demographics = {"age": 28, "gender": "男", "location": "北京", "membership_level": "黄金会员"}

    print("=== 构建用户画像 ===")
    user_profile = rec_system.build_user_profile(user_id, user_history, user_demographics)
    print(f"用户画像: {json.dumps(user_profile, ensure_ascii=False, indent=2)}")

    print("\n=== 生成推荐 ===")
    context = "用户即将出差，需要购买一些旅行用品"
    recommendations = rec_system.generate_recommendations(user_id, context)

    print("为用户推荐的商品:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. 商品ID: {rec.get('product_id', '未知')}, 名称: {rec.get('name', '未知')}")
        print(f"   推荐理由: {rec.get('reason', '无')}")

    print("\n=== 处理用户反馈 ===")
    feedback = {
        "p001": {"rating": 5, "comment": "非常符合我的需求，已经购买"},
        "p003": {"rating": 4, "comment": "背包质量不错，但希望有更多颜色选择"},
        "p005": {"rating": 2, "comment": "已经有了类似的耳机，不需要再买"}
    }

    updated_profile = rec_system.process_feedback(user_id, feedback)
    print(f"更新后的用户画像: {json.dumps(updated_profile, ensure_ascii=False, indent=2)}")

    print("\n=== 基于反馈生成新推荐 ===")
    new_recommendations = rec_system.generate_recommendations(user_id, "基于用户反馈的二次推荐")

    print("为用户推荐的新商品:")
    for i, rec in enumerate(new_recommendations, 1):
        print(f"{i}. 商品ID: {rec.get('product_id', '未知')}, 名称: {rec.get('name', '未知')}")
        print(f"   推荐理由: {rec.get('reason', '无')}")
