import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
import json

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

    def get_sorted_emails(self) -> List[Email]:
        """获取按优先级排序的邮件列表"""
        # 过滤掉已回复的邮件
        active_emails = [email for email in self.emails.values() if not email.is_replied]

        # 按优先级分数排序（降序）
        sorted_emails = sorted(
            active_emails,
            key=lambda x: (x.priority_score or 0, x.timestamp),
            reverse=True
        )
        return sorted_emails

    def get_email_summary(self) -> str:
        """获取邮件摘要"""
        sorted_emails = self.get_sorted_emails()
        if not sorted_emails:
            return "没有待处理的邮件。"

        summary = "待处理邮件（按优先级排序）:\n"
        for email in sorted_emails:
            read_status = "已读" if email.is_read else "未读"
            summary += (
                f"ID: {email.id}, 发件人: {email.sender}, "
                f"主题: '{email.subject}', 优先级: {email.priority_category or '未评分'}, "
                f"状态: {read_status}\n"
            )
        return summary


# 创建邮件管理器实例
email_manager = EmailManager()


# 工具参数模型
class AnalyzeEmailArgs(BaseModel):
    email_id: str = Field(description="要分析的邮件ID，例如'EMAIL-001'")


class ReplyEmailArgs(BaseModel):
    email_id: str = Field(description="要回复的邮件ID，例如'EMAIL-001'")
    response: str = Field(description="回复内容")


# 工具函数 - 所有函数都需要接受一个参数
def analyze_email_priority(input_str: str) -> str:
    """分析邮件优先级"""
    try:
        # 解析输入参数
        input_data = json.loads(input_str)
        email_id = input_data.get("email_id")
    except:
        # 如果解析失败，尝试直接作为email_id
        email_id = input_str

    email = email_manager.get_email(email_id)
    if not email:
        return f"未找到ID为 {email_id} 的邮件。"

    # 创建分析提示 - 使用简单的字符串格式化
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
        priority_score = result.get("score", 0)
        priority_category = result.get("category", "Medium")
        reason = result.get("reason", "未提供原因")

        # 更新邮件优先级
        email_manager.update_email_priority(email_id, priority_score, priority_category)

        return f"邮件 {email_id} 优先级已更新: {priority_category} (分数: {priority_score}/40)。原因: {reason}"
    except:
        # 如果解析失败，使用默认值
        email_manager.update_email_priority(email_id, 20, "Medium")
        return f"邮件 {email_id} 已分配默认优先级: Medium (分数: 20/40)。"


def reply_to_email(input_str: str) -> str:
    """回复邮件"""
    # 尝试解析输入参数
    try:
        # 首先尝试直接解析JSON
        input_data = json.loads(input_str)
        if isinstance(input_data, dict) and "email_id" in input_data and "response" in input_data:
            email_id = input_data.get("email_id")
            response = input_data.get("response", "")
        else:
            # 如果格式不正确，尝试其他方法
            raise ValueError("Unexpected format")
    except:
        # 如果直接解析失败，尝试从字符串中提取
        import re
        # 查找email_id
        email_id_match = re.search(r'"email_id"\s*:\s*"([^"]+)"', input_str)
        email_id = email_id_match.group(1) if email_id_match else None

        # 查找response
        response_match = re.search(r'"response"\s*:\s*"([^"]+)"', input_str)
        response = response_match.group(1) if response_match else None

        if not email_id or not response:
            return "输入参数格式错误，无法解析email_id和response。"

    email = email_manager.get_email(email_id)
    if not email:
        return f"未找到ID为 {email_id} 的邮件。"

    # 标记为已回复
    if email_manager.mark_as_replied(email_id):
        return f"已回复邮件 {email_id}。回复内容: '{response}'"
    return f"回复邮件 {email_id} 失败。"


def get_email_list(input_str: str = "") -> str:
    """获取邮件列表"""
    return email_manager.get_email_summary()


def mark_email_as_read(input_str: str) -> str:
    """标记邮件为已读"""
    try:
        # 解析输入参数
        input_data = json.loads(input_str)
        email_id = input_data.get("email_id")
    except:
        # 如果解析失败，尝试直接作为email_id
        email_id = input_str

    if email_manager.mark_as_read(email_id):
        return f"邮件 {email_id} 已标记为已读。"
    return f"标记邮件 {email_id} 为已读失败。"


# 工具集合 - 使用更简单的参数处理
email_tools = [
    Tool(
        name="analyze_email_priority",
        func=analyze_email_priority,
        description="分析邮件优先级，根据紧急性、重要性等标准对邮件进行评分和分类。输入应为邮件ID，如EMAIL-001。",
    ),
    Tool(
        name="reply_to_email",
        func=reply_to_email,
        description="回复指定邮件，并标记为已回复。输入格式：email_id:EMAIL-001, response:回复内容",
    ),
    Tool(
        name="get_email_list",
        func=get_email_list,
        description="获取按优先级排序的邮件列表。不需要输入参数。"
    ),
    Tool(
        name="mark_email_as_read",
        func=mark_email_as_read,
        description="标记邮件为已读状态。输入应为邮件ID，如EMAIL-001。",
    ),
]

# 邮件处理Agent提示模板
email_prompt_template = ChatPromptTemplate.from_template("""你是一个智能邮件处理助手。你的目标是高效管理用户的邮件收件箱。

你有以下工具可用：
{tools}

工具名称：
{tool_names}

当收到新邮件时，遵循以下步骤：
1. 使用 get_email_list 工具查看当前所有邮件。
2. 使用 analyze_email_priority 工具分析每封邮件的优先级。
3. 根据优先级排序，优先处理高优先级邮件。
4. 对于高优先级邮件，标记为已读并考虑是否需要立即回复。
5. 对于中优先级邮件，标记为已读并安排在当天处理。
6. 对于低优先级邮件，可以稍后处理。

优先级类别说明：
- High (30-40分): 需要立即处理
- Medium (20-29分): 当天处理
- Low (0-19分): 本周处理

工具使用注意事项：
- analyze_email_priority 和 mark_email_as_read 工具输入邮件ID，如EMAIL-001
- reply_to_email 工具输入格式：email_id:EMAIL-001, response:回复内容
- get_email_list 工具不需要输入参数

使用以下格式：
Question: 用户的问题或请求
Thought: 你需要思考该做什么
Action: 选择一个工具
Action Input: 工具的输入参数
Observation: 工具执行的结果
... (可以重复Thought/Action/Action Input/Observation多次)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

# 创建Agent执行器
email_agent = create_react_agent(llm, email_tools, email_prompt_template)
email_agent_executor = AgentExecutor(
    agent=email_agent,
    tools=email_tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)


# 模拟函数
async def run_email_simulation():
    print("--- 智能邮件处理系统模拟 ---")

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

    # 处理邮件
    print("\n[用户] 处理我的邮件收件箱")
    await email_agent_executor.ainvoke({"input": "我有新邮件需要处理，请帮我分析优先级并处理最重要的邮件。"})

    print("\n" + "-" * 60 + "\n")

    # 模拟新邮件到达
    print("[系统] 接收新邮件...")
    email4 = email_manager.add_email(
        sender="hr@company.com",
        subject="会议通知：明天下午3点团队会议",
        content="请准时参加明天下午3点的团队会议，地点：3楼会议室。",
        tags=["工作", "会议"]
    )

    # 再次处理邮件
    print("\n[用户] 有新邮件，重新评估优先级")
    await email_agent_executor.ainvoke({"input": "有新邮件到达，请重新评估所有邮件的优先级。"})

    print("\n--- 模拟完成 ---")


# 运行模拟
if __name__ == "__main__":
    asyncio.run(run_email_simulation())