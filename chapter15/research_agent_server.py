from fastapi import FastAPI, HTTPException, Header
from langchain_classic.agents import AgentExecutor, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from pydantic import BaseModel
from typing import List, Optional, Literal


# --- 1. A2A 消息结构定义 (使用 Pydantic 进行验证) ---
from init_client import init_llm


class A2AMessagePart(BaseModel):
    type: str
    text: Optional[str] = None


class A2AMessage(BaseModel):
    role: Literal["user", "assistant"]
    parts: List[A2AMessagePart]


class A2ARequestParams(BaseModel):
    id: str
    message: A2AMessage


class A2ARequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    method: Literal["sendTask"]
    params: A2ARequestParams


class A2AResponseResult(BaseModel):
    taskId: str
    status: dict
    message: A2AMessage


class A2AResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: A2AResponseResult


# --- 2. 初始化 LangChain 研究员 Agent ---
try:
    llm = init_llm(temperature=0)

    # 创建一个简单的"研究"工具，实际上只是让LLM基于其训练数据回答
    research_prompt = PromptTemplate.from_template(
        "作为一个专业研究员，请提供关于以下主题的最新信息和进展：{topic}\n\n"
        "请提供详细、准确的信息，包括关键概念、最新发展和重要事件。"
    )


    def research_function(topic: str) -> str:
        """研究一个主题并提供相关信息"""
        return llm.invoke(research_prompt.format(topic=topic))


    tools = [
        Tool(
            name="Research",
            func=research_function,
            description="研究一个主题并提供相关信息"
        )
    ]

    # 使用 ReAct 框架创建一个能够思考和行动的 Agent
    agent_executor: AgentExecutor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_execution_time=60,
    )
    print("LangChain agent initialized successfully.")
except Exception as e:
    print(f"Error initializing LangChain agent: {e}")
    agent_executor = None

# --- 3. 创建 FastAPI 应用 ---
app = FastAPI(
    title="Research Agent A2A Server",
    description="An A2A server that provides research capabilities."
)

# 模拟的 API Key 存储
VALID_A2A_API_KEY = "fjroejg50j34hg50"


# A2A 端点，用于处理任务
@app.post("/a2a", response_model=A2AResponse)
async def handle_a2a_request(
        request: A2ARequest,
        authorization: str = Header(None, description="Expected format: 'Bearer <API_KEY>'")
):
    # 1. 身份验证
    if not authorization or f"Bearer {VALID_A2A_API_KEY}" != authorization:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    if not agent_executor:
        raise HTTPException(status_code=503, detail="Agent executor is not available.")

    # 2. 解析请求
    if request.method != "sendTask":
        raise HTTPException(status_code=400, detail="Unsupported method")

    task_id = request.params.id
    user_message = ""
    # 从 A2A 消息结构中提取文本
    for part in request.params.message.parts:
        if part.type == "text" and part.text:
            user_message = part.text
            break

    print(f"[Server] Received task '{task_id}': {user_message}")

    # 3. 调用 LangChain Agent 执行任务
    try:
        # 使用 ainvoke 进行异步调用
        result = await agent_executor.ainvoke({"input": user_message})
        response_text = result['output']
    except Exception as e:
        print(f"[Server] Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

    # 4. 封装 A2A 响应
    a2a_response = A2AResponse(
        id=request.id,
        result=A2AResponseResult(
            taskId=task_id,
            status={"state": "completed"},
            message=A2AMessage(
                role="assistant",
                parts=[A2AMessagePart(type="text", text=response_text)]
            )
        )
    )
    print(f"[Server] Sending response for task '{task_id}'.")
    return a2a_response


if __name__ == "__main__":
    import uvicorn

    print("Starting Research Agent Server on http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001)