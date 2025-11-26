import asyncio
import warnings  # 导入 warnings 模块来忽略警告
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.warnings import LangGraphDeprecatedSinceV10
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import create_model

# --- 使用最稳定的 LangGraph Agent 构建器 ---
from langgraph.prebuilt import create_react_agent

# --- 忽略 LangGraph 的废弃警告 ---
warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)

from init_client import init_llm

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


# --- 主执行逻辑 ---
async def main():
    manager = MCPToolManager("mcp_utility_server.py")

    try:
        await manager.connect()
        await manager.load_tools()

        if not manager.tools:
            print("未能从MCP服务器加载任何工具。")
            return

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

        # 使用 LangGraph 的 prompt 构建方式
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),  # LangGraph 会自动填充对话历史
        ])

        # 使用最稳定的 create_react_agent
        agent_executor = create_react_agent(llm, manager.tools, prompt=prompt_template)

        while True:
            user_input = input("你: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            print("Agent: ", end="", flush=True)
            # LangGraph 的 astream 输入格式
            async for chunk in agent_executor.astream({"messages": [HumanMessage(content=user_input)]}):
                if 'agent' in chunk:
                    print(chunk['agent']['messages'][0].content, end="", flush=True)
                elif 'tools' in chunk:
                    # 为了清晰，我们手动打印工具调用信息
                    tool_msg = chunk['tools']['messages'][0]
                    print(f"\n[调用工具: {tool_msg.name}]")
                    print(f"[工具输入: {tool_msg.content}]")
            print("\n", end="", flush=True)

    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())