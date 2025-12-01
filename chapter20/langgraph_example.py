import asyncio
from typing import List, Dict, Optional, TypedDict, Annotated
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from init_client import init_llm

llm = init_llm(
    temperature=0.1
)


# 邮件数据模型
class Email(BaseModel):
    """表示系统中的单个邮件"""
    id: str
    sender: str
    subject: str
    content: str
    timestamp: datetime
    priority_score: Optional[float] = None
    priority_category: Optional[str] = None  # High, Medium, Low
    is_read: bool = False
    is_replied: bool = False
    tags: List[str] = []


# 定义Agent状态
class EmailAgentState(TypedDict):
    """邮件处理Agent的状态"""
    emails: List[Email]
    current_email_id: Optional[str]
    high_priority_emails: List[Email]
    medium_priority_emails: List[Email]
    low_priority_emails: List[Email]
    processed_emails: List[str]
    messages: Annotated[List, "消息历史"]
    next_action: str


class EmailManager:
    """邮件管理器，负责存储和排序邮件"""

    def __init__(self):
        self.emails: Dict[str, Email] = {}
        self.next_email_id = 1

    def add_email(self, sender: str, subject: str, content: str, tags: List[str] = None) -> Email:
        """添加新邮件"""
        email_id = f"EMAIL-{self.next_email_id:03d}"
        new_email = Email(
            id=email_id,
            sender=sender,
            subject=subject,
            content=content,
            timestamp=datetime.now(),
            tags=tags or []
        )
        self.emails[email_id] = new_email
        self.next_email_id += 1
        return new_email

    def get_email(self, email_id: str) -> Optional[Email]:
        """获取特定邮件"""
        return self.emails.get(email_id)

    def update_email_priority(self, email_id: str, priority_score: float, priority_category: str) -> bool:
        """更新邮件优先级"""
        email = self.emails.get(email_id)
        if email:
            email.priority_score = priority_score
            email.priority_category = priority_category
            return True
        return False

    def mark_as_read(self, email_id: str) -> bool:
        """标记邮件为已读"""
        email = self.emails.get(email_id)
        if email:
            email.is_read = True
            return True
        return False

    def mark_as_replied(self, email_id: str) -> bool:
        """标记邮件为已回复"""
        email = self.emails.get(email_id)
        if email:
            email.is_replied = True
            return True
        return False

    def get_all_emails(self) -> List[Email]:
        """获取所有邮件"""
        return list(self.emails.values())

    def get_unprocessed_emails(self) -> List[Email]:
        """获取未处理的邮件"""
        return [email for email in self.emails.values() if not email.is_replied]


# 创建邮件管理器实例
email_manager = EmailManager()


# 分析邮件优先级的函数
def analyze_email_priority(email: Email) -> Dict[str, any]:
    """分析邮件优先级"""
    analysis_prompt = f"""你是一个邮件优先级分析专家。请根据以下标准评估邮件优先级：
    1. 紧急性（0-10分）：邮件内容是否需要立即处理？
    2. 重要性（0-10分）：邮件内容对用户目标的贡献度如何？
    3. 发件人重要性（0-10分）：发件人对用户的重要性如何？
    4. 时效性（0-10分）：邮件内容是否有时间敏感性？

    请计算总分（40分满分）并确定优先级类别：
    - High (30-40分): 需要立即处理
    - Medium (20-29分): 当天处理
    - Low (0-19分): 本周处理

    请以JSON格式返回结果：{{"score": 分数, "category": "High/Medium/Low", "reason": "评分原因"}}

    邮件信息：
    发件人: {email.sender}
    主题: {email.subject}
    内容: {email.content}
    标签: {', '.join(email.tags)}
    """

    # 获取LLM响应
    response = llm.invoke(analysis_prompt)

    try:
        # 解析JSON响应
        result = json.loads(response.content)
        priority_score = result.get("score", 20)
        priority_category = result.get("category", "Medium")
        reason = result.get("reason", "未提供原因")

        # 更新邮件优先级
        email_manager.update_email_priority(email.id, priority_score, priority_category)

        return {
            "email_id": email.id,
            "priority_score": priority_score,
            "priority_category": priority_category,
            "reason": reason
        }
    except:
        # 如果解析失败，使用默认值
        email_manager.update_email_priority(email.id, 20, "Medium")
        return {
            "email_id": email.id,
            "priority_score": 20,
            "priority_category": "Medium",
            "reason": "解析失败，使用默认优先级"
        }


# 定义LangGraph节点函数
def get_emails(state: EmailAgentState) -> EmailAgentState:
    """获取所有邮件的节点"""
    emails = email_manager.get_all_emails()

    # 添加消息到状态
    message = HumanMessage(content=f"获取到 {len(emails)} 封邮件")
    state["messages"].append(message)

    # 更新状态
    state["emails"] = emails
    state["next_action"] = "analyze_priorities"

    return state


def analyze_priorities(state: EmailAgentState) -> EmailAgentState:
    """分析邮件优先级的节点"""
    emails = state["emails"]
    high_priority = []
    medium_priority = []
    low_priority = []

    for email in emails:
        if not email.is_replied:  # 只分析未回复的邮件
            analysis = analyze_email_priority(email)

            # 根据分析结果分类邮件
            if analysis["priority_category"] == "High":
                high_priority.append(email)
            elif analysis["priority_category"] == "Medium":
                medium_priority.append(email)
            else:
                low_priority.append(email)

    # 按优先级分数排序
    high_priority.sort(key=lambda x: x.priority_score or 0, reverse=True)
    medium_priority.sort(key=lambda x: x.priority_score or 0, reverse=True)
    low_priority.sort(key=lambda x: x.priority_score or 0, reverse=True)

    # 添加消息到状态
    message = AIMessage(
        content=f"分析完成：高优先级 {len(high_priority)} 封，中优先级 {len(medium_priority)} 封，低优先级 {len(low_priority)} 封")
    state["messages"].append(message)

    # 更新状态
    state["high_priority_emails"] = high_priority
    state["medium_priority_emails"] = medium_priority
    state["low_priority_emails"] = low_priority
    state["next_action"] = "process_high_priority" if high_priority else "process_medium_priority"

    return state


def process_high_priority(state: EmailAgentState) -> EmailAgentState:
    """处理高优先级邮件的节点"""
    high_priority_emails = state["high_priority_emails"]

    if not high_priority_emails:
        state["next_action"] = "process_medium_priority"
        return state

    # 处理第一封高优先级邮件
    email = high_priority_emails[0]
    email_id = email.id

    # 标记为已读
    email_manager.mark_as_read(email_id)

    # 生成回复
    reply_prompt = f"""请为以下邮件生成一个专业的回复：

    发件人: {email.sender}
    主题: {email.subject}
    内容: {email.content}

    回复应该简洁、专业，并表明你将立即处理这个高优先级事项。
    """

    response = llm.invoke(reply_prompt)
    reply_content = response.content

    # 标记为已回复
    email_manager.mark_as_replied(email_id)

    # 添加消息到状态
    message = AIMessage(content=f"已处理高优先级邮件 {email_id}，回复内容：{reply_content}")
    state["messages"].append(message)

    # 更新状态
    state["current_email_id"] = email_id
    state["processed_emails"].append(email_id)

    # 如果还有高优先级邮件，继续处理；否则转到中优先级
    if len(high_priority_emails) > 1:
        state["high_priority_emails"] = high_priority_emails[1:]  # 移除已处理的邮件
        state["next_action"] = "process_high_priority"
    else:
        state["high_priority_emails"] = []
        state["next_action"] = "process_medium_priority"

    return state


def process_medium_priority(state: EmailAgentState) -> EmailAgentState:
    """处理中优先级邮件的节点"""
    medium_priority_emails = state["medium_priority_emails"]

    if not medium_priority_emails:
        state["next_action"] = "process_low_priority"
        return state

    # 处理第一封中优先级邮件
    email = medium_priority_emails[0]
    email_id = email.id

    # 标记为已读
    email_manager.mark_as_read(email_id)

    # 生成回复
    reply_prompt = f"""请为以下邮件生成一个专业的回复：

    发件人: {email.sender}
    主题: {email.subject}
    内容: {email.content}

    回复应该专业，并表明你将在当天处理这个中优先级事项。
    """

    response = llm.invoke(reply_prompt)
    reply_content = response.content

    # 标记为已回复
    email_manager.mark_as_replied(email_id)

    # 添加消息到状态
    message = AIMessage(content=f"已处理中优先级邮件 {email_id}，回复内容：{reply_content}")
    state["messages"].append(message)

    # 更新状态
    state["current_email_id"] = email_id
    state["processed_emails"].append(email_id)

    # 如果还有中优先级邮件，继续处理；否则转到低优先级
    if len(medium_priority_emails) > 1:
        state["medium_priority_emails"] = medium_priority_emails[1:]  # 移除已处理的邮件
        state["next_action"] = "process_medium_priority"
    else:
        state["medium_priority_emails"] = []
        state["next_action"] = "process_low_priority"

    return state


def process_low_priority(state: EmailAgentState) -> EmailAgentState:
    """处理低优先级邮件的节点"""
    low_priority_emails = state["low_priority_emails"]

    if not low_priority_emails:
        state["next_action"] = "end"
        return state

    # 处理第一封低优先级邮件
    email = low_priority_emails[0]
    email_id = email.id

    # 标记为已读
    email_manager.mark_as_read(email_id)

    # 生成回复
    reply_prompt = f"""请为以下邮件生成一个简短的回复：

    发件人: {email.sender}
    主题: {email.subject}
    内容: {email.content}

    回复应该简洁，并表明你将在本周有空时处理这个低优先级事项。
    """

    response = llm.invoke(reply_prompt)
    reply_content = response.content

    # 标记为已回复
    email_manager.mark_as_replied(email_id)

    # 添加消息到状态
    message = AIMessage(content=f"已处理低优先级邮件 {email_id}，回复内容：{reply_content}")
    state["messages"].append(message)

    # 更新状态
    state["current_email_id"] = email_id
    state["processed_emails"].append(email_id)

    # 如果还有低优先级邮件，继续处理；否则结束
    if len(low_priority_emails) > 1:
        state["low_priority_emails"] = low_priority_emails[1:]  # 移除已处理的邮件
        state["next_action"] = "process_low_priority"
    else:
        state["low_priority_emails"] = []
        state["next_action"] = "end"

    return state


def decide_next_step(state: EmailAgentState) -> str:
    """决定下一步执行哪个节点"""
    return state["next_action"]


# 构建LangGraph工作流
def build_email_workflow():
    """构建邮件处理工作流"""
    # 创建状态图
    workflow = StateGraph(EmailAgentState)

    # 添加节点
    workflow.add_node("get_emails", get_emails)
    workflow.add_node("analyze_priorities", analyze_priorities)
    workflow.add_node("process_high_priority", process_high_priority)
    workflow.add_node("process_medium_priority", process_medium_priority)
    workflow.add_node("process_low_priority", process_low_priority)

    # 设置入口点
    workflow.set_entry_point("get_emails")

    # 添加边
    workflow.add_edge("get_emails", "analyze_priorities")

    # 添加条件边，根据decide_next_step函数的返回值决定下一步
    workflow.add_conditional_edges(
        "analyze_priorities",
        decide_next_step,
        {
            "process_high_priority": "process_high_priority",
            "process_medium_priority": "process_medium_priority",
            "process_low_priority": "process_low_priority"
        }
    )

    workflow.add_conditional_edges(
        "process_high_priority",
        decide_next_step,
        {
            "process_high_priority": "process_high_priority",
            "process_medium_priority": "process_medium_priority",
            "process_low_priority": "process_low_priority"
        }
    )

    workflow.add_conditional_edges(
        "process_medium_priority",
        decide_next_step,
        {
            "process_medium_priority": "process_medium_priority",
            "process_low_priority": "process_low_priority"
        }
    )

    workflow.add_conditional_edges(
        "process_low_priority",
        decide_next_step,
        {
            "process_low_priority": "process_low_priority",
            "end": END
        }
    )

    # 编译工作流
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# 模拟函数
async def run_email_simulation():
    print("--- 邮件优先级排序系统模拟 ---")

    # 添加示例邮件
    print("\n[系统] 接收新邮件...")
    email1 = email_manager.add_email(
        sender="boss@company.com",
        subject="紧急：季度报告需要修改",
        content="请立即修改季度报告中的财务数据，董事会将在明天上午9点审查。",
        tags=["工作", "紧急", "财务"]
    )

    email2 = email_manager.add_email(
        sender="newsletter@techblog.com",
        subject="本周AI技术动态",
        content="本周AI领域最新研究进展和行业动态...",
        tags=["订阅", "技术"]
    )

    email3 = email_manager.add_email(
        sender="client@importantclient.com",
        subject="项目进度咨询",
        content="想了解一下我们委托的项目目前进展如何，是否按计划进行？",
        tags=["客户", "项目"]
    )

    # 构建工作流
    app = build_email_workflow()

    # 初始状态
    initial_state = {
        "emails": [],
        "current_email_id": None,
        "high_priority_emails": [],
        "medium_priority_emails": [],
        "low_priority_emails": [],
        "processed_emails": [],
        "messages": [HumanMessage(content="我有新邮件需要处理，请帮我分析优先级并处理最重要的邮件。")],
        "next_action": "get_emails"
    }

    # 运行工作流
    print("\n[用户] 处理我的邮件收件箱")
    config = {"configurable": {"thread_id": "email-thread-1"}}
    result = app.invoke(initial_state, config)

    # 打印处理结果
    print("\n--- 邮件处理结果 ---")
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")

    print(f"\n已处理邮件: {result['processed_emails']}")

    print("\n" + "-" * 60 + "\n")

    # 模拟新邮件到达
    print("[系统] 接收新邮件...")
    email4 = email_manager.add_email(
        sender="hr@company.com",
        subject="会议通知：明天下午3点团队会议",
        content="请准时参加明天下午3点的团队会议，地点：3楼会议室。",
        tags=["工作", "会议"]
    )

    # 再次运行工作流
    print("\n[用户] 有新邮件，重新评估优先级")
    new_state = {
        "emails": [],
        "current_email_id": None,
        "high_priority_emails": [],
        "medium_priority_emails": [],
        "low_priority_emails": [],
        "processed_emails": [],
        "messages": [HumanMessage(content="有新邮件到达，请重新评估所有邮件的优先级。")],
        "next_action": "get_emails"
    }

    result = app.invoke(new_state, config)

    # 打印处理结果
    print("\n--- 邮件处理结果 ---")
    for message in result["messages"]:
        print(f"{message.type}: {message.content}")

    print(f"\n已处理邮件: {result['processed_emails']}")

    print("\n--- 模拟完成 ---")


# 运行模拟
if __name__ == "__main__":
    asyncio.run(run_email_simulation())