import os
import json
import re
from typing import List, Dict, Any, ClassVar
from pathlib import Path

# PDF处理库
import pdfplumber

# LangChain组件
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from init_client import init_llm

class PDFParser:
    """PDF解析器 - 使用RunnableLambda包装"""

    def __init__(self):
        self.parser = RunnableLambda(self._parse_pdf)

    def _parse_pdf(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pdf_path = inputs["pdf_path"]

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        print(f"📄 开始解析PDF文件: {pdf_path}")

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content = []

                # 提取前10页内容（可根据需要调整）
                max_pages = min(len(pdf.pages), 10)
                for i in range(max_pages):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text:
                        text_content.append(text)

                paper_text = "\n\n".join(text_content)

                # 清理文本
                paper_text = re.sub(r'\s+', ' ', paper_text)
                paper_text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()[]{}"\'-]', '', paper_text)

                print(f"✅ PDF解析完成，共提取 {len(paper_text)} 字符")
                return {"paper_text": paper_text}

        except Exception as e:
            print(f"❌ PDF解析失败: {e}")
            return {"paper_text": ""}


class PaperAnalysisOutputParser(BaseOutputParser[Dict[str, Any]]):
    """论文分析输出解析器"""

    # 将 schemas 定义为类变量
    analysis_schemas: ClassVar[List[ResponseSchema]] = [
        ResponseSchema(name="title", description="论文标题"),
        ResponseSchema(name="authors", description="论文作者列表"),
        ResponseSchema(name="abstract", description="论文摘要"),
        ResponseSchema(name="key_findings", description="主要发现，以列表形式呈现"),
        ResponseSchema(name="methodology", description="研究方法简述"),
        ResponseSchema(name="limitations", description="研究局限性"),
        ResponseSchema(name="future_work", description="未来工作建议")
    ]

    # 使用 @property 按需创建解析器实例
    @property
    def _parser(self) -> StructuredOutputParser:
        return StructuredOutputParser.from_response_schemas(self.analysis_schemas)

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            return self._parser.parse(text)
        except Exception as e:
            print(f"⚠️ 分析结果解析失败，使用原始响应: {e}")
            return {"raw_analysis": text}

    def get_format_instructions(self) -> str:
        return self._parser.get_format_instructions()


class SummaryEvaluationOutputParser(BaseOutputParser[Dict[str, Any]]):
    """摘要评估输出解析器"""

    # 将 schemas 定义为类变量
    evaluation_schemas: ClassVar[List[ResponseSchema]] = [
        ResponseSchema(name="meets_goals", description="摘要是否满足所有设定目标，回答'是'或'否'"),
        ResponseSchema(name="accuracy_score", description="摘要准确度评分，1-10"),
        ResponseSchema(name="clarity_score", description="摘要清晰度评分，1-10"),
        ResponseSchema(name="completeness_score", description="摘要完整性评分，1-10"),
        ResponseSchema(name="feedback", description="改进建议，如果不满足目标")
    ]

    # 使用 @property 按需创建解析器实例
    @property
    def _parser(self) -> StructuredOutputParser:
        return StructuredOutputParser.from_response_schemas(self.evaluation_schemas)

    def parse(self, text: str) -> Dict[str, Any]:
        try:
            return self._parser.parse(text)
        except Exception as e:
            print(f"⚠️ 评估结果解析失败: {e}")
            return {"error": str(e)}

    def get_format_instructions(self) -> str:
        return self._parser.get_format_instructions()


class IntelligentPaperAnalyzer:
    """智能论文分析器 - 使用管道操作符构建Agent"""

    def __init__(self, max_iterations: int = 3):
        """初始化智能论文分析器"""
        self.llm = init_llm(
            temperature=0.1
        )
        self.max_iterations = max_iterations
        self.goals = []

        # 初始化组件 - 现在可以正常实例化
        self.pdf_parser = PDFParser()
        self.analysis_parser = PaperAnalysisOutputParser()
        self.evaluation_parser = SummaryEvaluationOutputParser()

        # 构建分析管道
        self._build_analysis_pipeline()
        # 构建评估管道
        self._build_evaluation_pipeline()
        # 构建改进管道
        self._build_improvement_pipeline()

    def _build_analysis_pipeline(self):
        """构建论文分析管道"""
        # 分析提示模板
        self.analysis_prompt = PromptTemplate(
            input_variables=["paper_text", "format_instructions"],
            template="""
            你是一个专业的研究论文分析助手。请仔细阅读以下研究论文，并提取关键信息。
            
            {format_instructions}
            
            论文内容:
            {paper_text}
            
            请确保提取的信息准确完整。
                        """
        )

        # 构建分析管道
        self.analysis_pipeline = (
                RunnablePassthrough.assign(
                    format_instructions=lambda _: self.analysis_parser.get_format_instructions()
                )
                | self.analysis_prompt
                | self.llm
                | RunnableLambda(lambda x: x.content)
                | self.analysis_parser
        )

    def _build_evaluation_pipeline(self):
        """构建摘要评估管道"""
        # 评估提示模板
        self.evaluation_prompt = PromptTemplate(
            input_variables=["goals", "analysis", "summary", "format_instructions"],
            template="""
            评估以下研究论文摘要是否满足设定的目标:
            
            目标: {goals}
            
            论文分析:
            {analysis}
            
            摘要:
            {summary}
            
            {format_instructions}
            
            请客观评估摘要质量，并提供具体的改进建议。
                        """
        )

        # 构建评估管道
        self.evaluation_pipeline = (
                RunnablePassthrough.assign(
                    format_instructions=lambda _: self.evaluation_parser.get_format_instructions()
                )
                | self.evaluation_prompt
                | self.llm
                | RunnableLambda(lambda x: x.content)
                | self.evaluation_parser
        )

    def _build_improvement_pipeline(self):
        """构建摘要改进管道"""
        # 改进提示模板
        self.improvement_prompt = PromptTemplate(
            input_variables=["summary", "feedback", "goals"],
            template="""
            根据以下反馈改进研究论文摘要:
            
            当前摘要:
            {summary}
            
            改进反馈:
            {feedback}
            
            目标要求:
            {goals}
            
            请提供改进后的摘要，要求:
            1. 保持简洁明了，不超过200字
            2. 充分考虑反馈意见
            3. 确保满足所有目标要求
            
            直接返回改进后的摘要，不要包含其他解释。
                        """
        )

        # 构建改进管道
        self.improvement_pipeline = (
                self.improvement_prompt
                | self.llm
                | RunnableLambda(lambda x: x.content.strip())
        )

    def set_goals(self, goals: List[str]) -> None:
        """设定分析目标"""
        self.goals = [g.strip() for g in goals]
        print(f"🎯 分析目标已设定: {', '.join(self.goals)}")

    def analyze_paper(self, pdf_path: str) -> Dict[str, Any]:
        """分析PDF论文的主流程"""
        if not self.goals:
            raise ValueError("请先使用set_goals()方法设定分析目标")

        print(f"\n🚀 开始分析PDF论文: {pdf_path}")
        print("=" * 60)

        # 第一步：解析PDF
        parse_result = self.pdf_parser.parser.invoke({"pdf_path": pdf_path})
        paper_text = parse_result["paper_text"]

        if not paper_text.strip():
            return {"success": False, "error": "无法解析PDF内容"}

        # 第二步：使用管道分析论文
        print("🔍 开始分析论文内容...")
        analysis = self.analysis_pipeline.invoke({
            "paper_text": paper_text[:8000]  # 限制文本长度
        })

        # 第三步：生成初始摘要
        print("📝 生成研究论文摘要...")
        summary_pipeline = (
                PromptTemplate(
                    input_variables=["goals", "analysis"],
                    template="""
                    基于以下研究论文分析，生成一个简洁明了的摘要，满足以下目标:
                    {goals}
                    
                    论文分析:
                    {analysis}
                    
                    摘要应该:
                    1. 简明扼要，不超过200字
                    2. 突出研究的主要贡献
                    3. 使用清晰易懂的语言
                    4. 避免技术术语过多
                    
                    请直接返回摘要内容，不要包含其他解释。
                                    """
                )
                | self.llm
                | RunnableLambda(lambda x: x.content.strip())
        )

        current_summary = summary_pipeline.invoke({
            "goals": ", ".join(self.goals),
            "analysis": json.dumps(analysis, ensure_ascii=False, indent=2)
        })

        # 第四步：迭代评估和改进
        iteration = 0
        final_evaluation = None

        while iteration < self.max_iterations:
            print(f"\n--- 🔁 评估迭代 {iteration + 1}/{self.max_iterations} ---")

            # 使用评估管道
            evaluation = self.evaluation_pipeline.invoke({
                "goals": ", ".join(self.goals),
                "analysis": json.dumps(analysis, ensure_ascii=False, indent=2),
                "summary": current_summary
            })

            meets_goals = evaluation.get("meets_goals", "").lower() == "是"
            final_evaluation = evaluation

            print(f"✅ 评估完成 - 满足目标: {evaluation.get('meets_goals', '未知')}")
            print(f"   准确度: {evaluation.get('accuracy_score', 'N/A')}/10")
            print(f"   清晰度: {evaluation.get('clarity_score', 'N/A')}/10")
            print(f"   完整性: {evaluation.get('completeness_score', 'N/A')}/10")

            if meets_goals:
                print(f"\n✅ 摘要满足所有目标，分析完成 (迭代次数: {iteration + 1})")
                break

            # 如果不满足目标且未达到最大迭代次数，则改进摘要
            if iteration < self.max_iterations - 1:
                print("\n🛠️ 摘要不满足目标，进行改进...")
                current_summary = self.improvement_pipeline.invoke({
                    "summary": current_summary,
                    "feedback": evaluation.get("feedback", "需要改进"),
                    "goals": ", ".join(self.goals)
                })
                iteration += 1
            else:
                print(f"\n⚠️ 已达到最大迭代次数 ({self.max_iterations})，返回当前结果")
                break

        # 保存结果
        self._save_results(pdf_path, analysis, current_summary, final_evaluation)

        return {
            "success": True,
            "analysis": analysis,
            "summary": current_summary,
            "evaluation": final_evaluation,
            "iterations": iteration + 1
        }

    def _save_results(self, pdf_path: str, analysis: Dict, summary: str, evaluation: Dict) -> None:
        """保存分析结果到文件"""
        pdf_name = Path(pdf_path).stem
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)

        # 保存分析报告
        report_path = output_dir / f"{pdf_name}_analysis_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== 论文分析报告 ===\n\n")
            f.write(f"论文标题: {analysis.get('title', '未知')}\n")
            f.write(f"作者: {analysis.get('authors', '未知')}\n\n")

            f.write("=== 分析结果 ===\n")
            f.write(json.dumps(analysis, ensure_ascii=False, indent=2))
            f.write("\n\n")

            f.write("=== 摘要 ===\n")
            f.write(summary)
            f.write("\n\n")

            f.write("=== 评估结果 ===\n")
            f.write(json.dumps(evaluation, ensure_ascii=False, indent=2))

        print(f"\n💾 分析报告已保存至: {report_path}")


# 使用示例
if __name__ == "__main__":
    # 创建智能论文分析器
    analyzer = IntelligentPaperAnalyzer(max_iterations=3)

    # 设定分析目标
    analyzer.set_goals([
        "简洁明了",
        "突出研究贡献",
        "适合非专业读者理解",
        "包含关键发现",
        "不超过200字"
    ])

    # 分析PDF论文
    # 请将路径替换为实际的PDF文件路径
    pdf_file_path = "基于关系驱动多模态嵌入塑形的图像描述生成.pdf"  # 替换为你的PDF文件路径

    if os.path.exists(pdf_file_path):
        result = analyzer.analyze_paper(pdf_file_path)

        if result["success"]:
            print("\n" + "=" * 60)
            print("📊 最终分析结果:")
            print("=" * 60)
            print(f"\n📝 最终摘要:\n{result['summary']}")

            if "evaluation" in result and result["evaluation"]:
                eval_data = result["evaluation"]
                print(f"\n📈 评估结果:")
                print(f"   满足目标: {eval_data.get('meets_goals', '未知')}")
                print(f"   准确度: {eval_data.get('accuracy_score', 'N/A')}/10")
                print(f"   清晰度: {eval_data.get('clarity_score', 'N/A')}/10")
                print(f"   完整性: {eval_data.get('completeness_score', 'N/A')}/10")

            print(f"\n🔄 迭代次数: {result['iterations']}")
        else:
            print(f"❌ 分析失败: {result['error']}")
    else:
        print(f"⚠️ PDF文件不存在: {pdf_file_path}")
        print("请将PDF文件放在当前目录下，或修改pdf_file_path变量指向正确的文件路径")