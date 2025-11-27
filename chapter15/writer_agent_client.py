import asyncio
import httpx

# --- 1. 初始化写作者 Agent 的 LLM ---
from init_client import init_llm

try:
    writer_llm = init_llm(temperature=0.7)
except Exception as e:
    print(f"Error initializing writer LLM: {e}")
    writer_llm = None


# --- 2. 定义 A2A 客户端逻辑 ---
async def run_writer_agent():
    if not writer_llm:
        print("Writer LLM is not available. Exiting.")
        return

    # A2A 服务器配置
    server_url = "http://127.0.0.1:8001/a2a"
    api_key = "fjroejg50j34hg50"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 用户请求
    user_topic = "最新的量子计算进展"
    print(f"用户请求: 撰写一份关于 '{user_topic}' 的报告。")
    print("-" * 30)

    # 步骤 1: 写作者 Agent 向研究员 Agent 发送 A2A 请求
    research_request_payload = {
        "jsonrpc": "2.0",
        "id": "req-001",
        "method": "sendTask",
        "params": {
            "id": "task-research-001",
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": f"请搜索关于 '{user_topic}' 的最新信息，总结关键突破和重要发现。"
                    }
                ]
            }
        }
    }

    research_data = "未能获取到研究数据。"

    # --- 1: 增加超时时间 ---
    # 将超时时间设置为 3 分钟 (180秒)，给 Agent 充足的时间
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            print("[Writer Agent] 正在向研究员 Agent 请求数据...")
            response = await client.post(server_url, json=research_request_payload, headers=headers)
            response.raise_for_status()  # 如果状态码不是 2xx，则引发异常

            response_json = response.json()
            # 检查 JSON-RPC 是否有错误
            if "error" in response_json:
                print(f"[Writer Agent] 研究员 Agent 返回错误: {response_json['error']}")
            else:
                # 从 A2A 响应中提取研究数据
                research_data = response_json.get("result", {}).get("message", {}).get("parts", [{}])[0].get("text", "")
                print("[Writer Agent] 成功获取研究数据。")
                print("\n--- 来自研究员 Agent 的原始数据 ---\n")
                print(research_data)
                print("\n" + "-" * 30 + "\n")

        # --- 2: 捕获并打印更详细的错误 ---
        except httpx.HTTPStatusError as e:
            print(f"\n[Writer Agent] HTTP 错误: {e.response.status_code} {e.response.reason_phrase}")
            print(f"[Writer Agent] 服务器响应内容: {e.response.text}")
        except httpx.RequestError as e:
            # 这将捕获连接错误、超时等
            print(f"\n[Writer Agent] 请求失败: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"\n[Writer Agent] 处理响应时发生未知错误: {type(e).__name__}: {e}")

    # 步骤 2: 写作者 Agent 基于获取的数据撰写报告
    if research_data != "未能获取到研究数据。":
        print("[Writer Agent] 正在基于数据撰写报告...")
        writer_prompt = f"""
        你是一位专业的科技报告写作者。你的任务是基于下面提供的研究数据，撰写一份关于 '{user_topic}' 的简洁、清晰、结构化的报告。
        报告应包含引言、关键进展总结和结论。

        --- 研究数据 ---
        {research_data}
        --- 数据结束 ---

        请开始撰写报告：
        """

        final_report = await writer_llm.ainvoke(writer_prompt)

        print("\n--- 最终报告 ---\n")
        print(final_report)
    else:
        print("[Writer Agent] 由于无法获取研究数据，报告撰写失败。")


if __name__ == "__main__":
    asyncio.run(run_writer_agent())