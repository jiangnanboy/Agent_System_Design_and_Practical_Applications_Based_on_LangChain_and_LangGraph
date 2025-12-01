from typing import Optional

from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool

from init_client import init_llm

llm = init_llm(temperature=0.2)

# 使用本地模拟搜索结果
def local_search(query: str) -> str:
    """使用本地模拟搜索结果"""
    # 这里可以根据关键词返回一些预定义的搜索结果
    # 实际应用中，可以构建一个本地知识库或使用离线搜索引擎

    # 模拟一些常见漏洞的搜索结果
    if "CVE" in query or "vulnerability" in query.lower():
        return """
        常见漏洞信息：
        1. CVE-2023-1234: Apache Struts2 远程代码执行漏洞，影响版本2.0.0-2.5.30
        2. CVE-2023-5678: OpenSSL 拒绝服务漏洞，影响版本1.1.1-1.1.1t
        3. CVE-2023-9012: Spring Framework 路径遍历漏洞，影响版本5.3.0-5.3.31
        """
    elif "apache" in query.lower():
        return """
        Apache常见安全问题：
        1. 默认配置可能泄露敏感信息
        2. 旧版本可能存在已知漏洞
        3. 不安全的HTTP方法可能被启用
        """
    elif "ssh" in query.lower():
        return """
        SSH常见安全问题：
        1. 弱密码或默认密码
        2. 使用不安全的加密算法
        3. 允许root用户直接登录
        4. 未限制登录尝试次数
        """
    else:
        return "未找到相关信息。请尝试更具体的搜索词，如'Apache CVE'或'SSH安全配置'。"


# 使用本地漏洞数据库
def local_vulnerability_db(component: str, version: str = "") -> str:
    """查询本地漏洞数据库"""
    # 这里可以维护一个本地漏洞数据库，如JSON文件或SQLite数据库
    # 示例中使用简单的字典

    vulnerability_db = {
        "apache": {
            "2.4.48": ["CVE-2021-34798: mod_proxy 漏洞", "CVE-2021-36160: 路径遍历漏洞"],
            "2.4.49": ["CVE-2021-40438: mod_proxy 漏洞", "CVE-2021-44790: 缓冲区溢出"],
            "2.4.50": ["CVE-2021-41773: 路径遍历漏洞", "CVE-2021-42013: 路径遍历漏洞"]
        },
        "openssh": {
            "8.2p1": ["CVE-2021-28041: scp 客户端漏洞"],
            "8.5p1": ["CVE-2021-41617: 特权提升漏洞"],
            "9.0p1": ["CVE-2023-25136: 前置条件竞争漏洞"]
        },
        "nginx": {
            "1.18.0": ["CVE-2021-23017: 解析器漏洞"],
            "1.20.1": ["CVE-2021-3618: 内存泄漏漏洞"],
            "1.21.6": ["CVE-2022-41741: 内存损坏漏洞"]
        }
    }

    component = component.lower()
    if component in vulnerability_db:
        if version and version in vulnerability_db[component]:
            return f"{component} {version} 的已知漏洞: {', '.join(vulnerability_db[component][version])}"
        else:
            all_vulns = []
            for ver, vulns in vulnerability_db[component].items():
                all_vulns.extend(vulns)
            return f"{component} 的已知漏洞: {', '.join(all_vulns)}"
    else:
        return f"未找到 {component} 的漏洞信息"


# 定义工具
tools = [
    Tool(
        name="NetworkScan",
        func=lambda target: "模拟网络扫描结果：发现端口22(SSH), 80(HTTP), 443(HTTPS)",
        description="对目标IP或域名执行基本的网络扫描，识别开放端口和服务"
    ),
    Tool(
        name="WebCrawl",
        func=lambda url: f"模拟网页内容爬取：从 {url} 获取到服务器信息：Apache/2.4.49",
        description="爬取网站的基本信息，包括HTML内容和响应头"
    ),
    Tool(
        name="LocalSearch",
        func=local_search,
        description="使用本地搜索功能查找安全相关信息，不需要外网访问"
    ),
    Tool(
        name="LocalVulnerabilityDB",
        func=local_vulnerability_db,
        description="查询本地漏洞数据库，查找已知组件的漏洞信息"
    ),
    Tool(
        name="DeepSeekAnalyze",
        func=lambda x: llm.invoke(
            f"作为网络安全专家，请分析以下信息中的潜在安全漏洞和威胁:\n\n{x}\n\n请提供:\n1. 发现的潜在漏洞列表\n2. 每个漏洞的严重程度评估(高/中/低)\n3. 可能的攻击向量\n4. 建议的修复措施"),
        description="使用DeepSeek进行深度安全分析，识别代码或配置中的潜在漏洞"
    )
]

# 创建Agent
# 1. 网络扫描Agent
network_scanner_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名网络安全专家，负责对目标系统进行初步扫描和信息收集。你的任务是使用NetworkScan和WebCrawl工具收集目标的基本信息，为后续分析做准备。"),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

network_scanner_agent = create_openai_tools_agent(llm, tools, network_scanner_prompt)
network_scanner_executor = AgentExecutor(agent=network_scanner_agent, tools=tools, verbose=True)

# 2. 漏洞分析Agent
vulnerability_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名漏洞分析专家，负责分析系统配置和代码中的潜在安全漏洞。使用DeepSeekAnalyze工具进行深度分析，并使用LocalVulnerabilityDB和LocalSearch查找已知漏洞。"),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

vulnerability_analyzer_agent = create_openai_tools_agent(llm, tools, vulnerability_analyzer_prompt)
vulnerability_analyzer_executor = AgentExecutor(agent=vulnerability_analyzer_agent, tools=tools, verbose=True)

# 3. 威胁评估Agent
threat_assessor_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名威胁评估专家，负责评估已发现漏洞的威胁程度和潜在影响。你需要考虑漏洞的严重性、利用难度和潜在的业务影响。"),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

threat_assessor_agent = create_openai_tools_agent(llm, tools, threat_assessor_prompt)
threat_assessor_executor = AgentExecutor(agent=threat_assessor_agent, tools=tools, verbose=True)

# 4. 报告生成Agent
report_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名安全报告专家，负责生成详细的安全评估报告，包括发现的漏洞、威胁评估和修复建议。报告应该清晰、专业且可操作。"),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}")
])

report_generator_agent = create_openai_tools_agent(llm, tools, report_generator_prompt)
report_generator_executor = AgentExecutor(agent=report_generator_agent, tools=tools, verbose=True)


# 主函数
def security_analysis(target: str, config_or_code: Optional[str] = None) -> str:
    """执行完整的安全分析流程"""

    # 阶段1：网络扫描和信息收集
    print("=== 阶段1：网络扫描和信息收集 ===")
    scan_result = network_scanner_executor.invoke({
        "input": f"请对目标 {target} 进行网络扫描和信息收集，包括开放端口、运行的服务和网站基本信息。"
    })

    # 阶段2：漏洞分析
    print("\n=== 阶段2：漏洞分析 ===")
    vulnerability_input = f"基于以下扫描结果分析潜在漏洞：\n{scan_result['output']}"
    if config_or_code:
        vulnerability_input += f"\n\n同时分析以下配置或代码中的安全漏洞：\n{config_or_code}"

    vulnerability_result = vulnerability_analyzer_executor.invoke({
        "input": vulnerability_input
    })

    # 阶段3：威胁评估
    print("\n=== 阶段3：威胁评估 ===")
    threat_result = threat_assessor_executor.invoke({
        "input": f"基于以下漏洞分析结果，评估每个漏洞的威胁程度和潜在影响：\n{vulnerability_result['output']}"
    })

    # 阶段4：生成报告
    print("\n=== 阶段4：生成安全报告 ===")
    report_input = f"""
    基于以下信息生成详细的安全评估报告：

    1. 扫描结果：
    {scan_result['output']}

    2. 漏洞分析：
    {vulnerability_result['output']}

    3. 威胁评估：
    {threat_result['output']}

    报告应包括：
    - 执行摘要
    - 发现的漏洞列表及严重程度
    - 潜在业务影响
    - 详细的修复建议
    - 长期安全改进措施
    """

    report_result = report_generator_executor.invoke({
        "input": report_input
    })

    return report_result['output']


# 使用示例
if __name__ == "__main__":
    # 示例1：模拟分析网站
    target_website = "example.com"
    security_report = security_analysis(target_website)
    print("=== 安全分析报告 ===")
    print(security_report)

    # 示例2：分析网络配置
    network_config = """
    # 路由器配置
    interface GigabitEthernet0/0
     ip address 192.168.1.1 255.255.255.0
     no shutdown
    !
    line vty 0 4
     password cisco123
     login
    !
    enable secret 5 $1$mERr$hh5V0fV9B9kXpWEp6oY6X.
    """

    security_report_config = security_analysis("192.168.1.1", network_config)
    print("\n=== 网络配置安全分析报告 ===")
    print(security_report_config)
