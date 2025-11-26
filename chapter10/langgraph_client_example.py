import asyncio
import warnings
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.warnings import LangGraphDeprecatedSinceV10
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import create_model
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from init_client import init_llm

# 忽略 LangGraph 的废弃警告
warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)


# 定义状态
class AgentState(TypedDict):
    messages: Annotated[List, "消息列表"]
    current_order: dict
    inventory_status: dict
    purchase_request: dict
    shipping_order: dict


class MCPToolManager:
    """一个用于管理 MCP 连接和工具生命周期的类。"""

    def __init__(self, server_script: str):
        self.server_script = server_script
        self.session: ClientSession | None = None
        self.tools = []
        self._stdio_cm = None
        self._session_cm = None

    async def connect(self):
        server_params = StdioServerParameters(
            command="python", args=[self.server_script]
        )
        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()

        await self.session.initialize()
        print("✅ MCP 服务器连接成功。")

    async def load_tools(self):
        if not self.session:
            raise RuntimeError("未连接到 MCP 服务器。请先调用 connect()。")

        response = await self.session.list_tools()

        self.tools = []
        for mcp_tool in response.tools:
            tool_name = mcp_tool.name
            tool_desc = mcp_tool.description
            input_schema = mcp_tool.inputSchema

            # 动态创建 Pydantic 模型作为 args_schema
            fields = {}
            required_fields = input_schema.get("required", [])
            for prop_name, prop_details in input_schema.get("properties", {}).items():
                prop_type = prop_details.get("type")
                python_type = str
                if prop_type == "number":
                    python_type = float
                elif prop_type == "integer":
                    python_type = int
                elif prop_type == "array":
                    python_type = list

                if prop_name in required_fields:
                    fields[prop_name] = (python_type, ...)
                else:
                    fields[prop_name] = (python_type, None)

            dynamic_args_model = create_model(f'{tool_name}Args', **fields)

            async def make_call_tool(s, tn):
                async def call_tool(**kwargs):
                    result = await s.call_tool(tn, arguments=kwargs)
                    return result.content[0].text

                return call_tool

            actual_call_tool = await make_call_tool(self.session, tool_name)
            actual_call_tool.__name__ = tool_name
            actual_call_tool.__doc__ = tool_desc

            langchain_tool = tool(actual_call_tool)
            langchain_tool.args_schema = dynamic_args_model

            self.tools.append(langchain_tool)

        print(f"✅ 成功从 MCP 服务器加载了 {len(self.tools)} 个工具。")

    async def close(self):
        if self.session and self._session_cm:
            await self._session_cm.__aexit__(None, None, None)
            self.session = None
            self._session_cm = None
        if self._stdio_cm:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None
        print("🔌 MCP 服务器连接已关闭。")


# 主执行逻辑
async def main():
    manager = MCPToolManager("mcp_utility_server.py")

    try:
        await manager.connect()
        await manager.load_tools()

        if not manager.tools:
            print("未能从MCP服务器加载任何工具。")
            return

        # 初始化 DeepSeek 模型
        llm = init_llm(
            temperature=0
        )

        system_prompt = """
        你是一个企业级销售订单处理专家，负责协调和自动化处理销售订单流程。

        你的工作流程包括：
        1. 接收新的销售订单
        2. 检查库存是否充足
        3. 如果库存不足，创建采购申请并等待批准
        4. 批准采购申请后，更新库存
        5. 创建发货单并更新库存
        6. 发送通知给相关人员

        请始终遵循以下原则：
        - 确保每个步骤都正确完成后再进行下一步
        - 在库存不足时，必须创建采购申请
        - 在创建发货单前，确保库存充足
        - 在关键步骤完成后，发送通知给相关人员
        - 始终保持专业和高效
        请使用中文进行思考和回答。
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])

        tools_dict = {tool.name: tool for tool in manager.tools}

        # 节点函数保持不变...
        async def call_model(state: AgentState):
            messages = state["messages"]
            response = await llm.ainvoke(prompt.format_messages(messages=messages))
            return {"messages": [response]}

        async def call_tool(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]

            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call = last_message.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tools_dict:
                    tool_result = await tools_dict[tool_name].ainvoke(tool_args)
                    # 使用 ToolMessage 来表示工具返回的结果
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                    return {"messages": [tool_message]}

            return {"messages": []}

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]

            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"

            return END

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tool)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
        )

        workflow.add_edge("tools", "agent")

        memory = MemorySaver()

        app = workflow.compile(checkpointer=memory)

        # --- 添加 config 参数 ---
        config = {"configurable": {"thread_id": "order-processing-thread-1"}}

        # 交互循环
        while True:
            user_input = input("你: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "current_order": {},
                "inventory_status": {},
                "purchase_request": {},
                "shipping_order": {}
            }

            print("Agent: ", end="", flush=True)
            # --- 在 astream 中传入 config ---
            async for event in app.astream(initial_state, config):
                for node, output in event.items():
                    if node == "agent" and "messages" in output:
                        for message in output["messages"]:
                            if hasattr(message, 'content') and message.content:
                                print(message.content, end="", flush=True)
                    elif node == "tools" and "messages" in output:
                        for message in output["messages"]:
                            if hasattr(message, 'content') and message.content:
                                print(f"\n[工具执行结果]: {message.content}", end="", flush=True)
            print("\n", end="", flush=True)

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())

    '''
    你：我需要处理一个新的销售订单，订单ID是SO-2023-001，客户ID是CUST-001，包含以下项目：2台商务笔记本(LAPTOP-001)，10个无线鼠标(MOUSE-001)，5个机械键盘(KEYBOARD-001)

    你：我需要处理一个新的销售订单，订单ID是SO-2023-002，客户ID是CUST-002，包含以下项目：3台27寸显示器(MONITOR-001)，5个机械键盘(KEYBOARD-001)
    你：我是采购经理，我需要批准采购申请 PR-20231115103520
    '''