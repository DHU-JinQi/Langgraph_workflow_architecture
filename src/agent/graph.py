"""
金融投资分析 LangGraph 框架
标准工作流处理机制
"""

import xml.etree.ElementTree as ET
import os
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 环境配置
load_dotenv()
model = ChatDeepSeek(model="deepseek-chat", max_tokens=8000)
logger.info("🚀 初始化 DeepSeek 模型完成")

# ============= 数据模型定义 =============

class FinancialAnalysisStep(BaseModel):
    step: str = Field(description="分析步骤名称")
    method: str = Field(description="使用的分析方法")
    data_needed: str = Field(description="此步骤需要的数据")

class FinancialAnalysisPlan(BaseModel):
    analysis_steps: List[FinancialAnalysisStep]

class FinancialReport(BaseModel):
    executive_summary: str = Field(description="执行摘要")
    detailed_report: str = Field(description="详细分析报告")
    investment_rating: str = Field(description="投资评级")
    target_price: str = Field(description="目标价位")
    risk_factors: List[str] = Field(description="风险因素")

# ============= 状态定义 =============

class FinancialAnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    analysis_plan: Optional[FinancialAnalysisPlan]
    collected_data: Optional[str]
    report: Optional[FinancialReport]
    workflow_stage: str

# ============= 工具定义 =============

@tool
def get_stock_data(symbol: str, period: str = "1y") -> str:
    """获取股票基础数据"""
    logger.info(f"📊 调用股票数据工具 - 股票代码: {symbol}, 周期: {period}")
    return f"""
    股票代码: {symbol}
    时间周期: {period}
    
    📈 基础数据:
    • 当前价格: $125.50
    • 市值: 500亿美元
    • P/E比率: 18.5
    • P/B比率: 2.3
    • ROE: 15.2%
    • 52周高点: $145.20
    • 52周低点: $98.30
    
    📊 近期表现:
    • 日涨跌幅: +2.1%
    • 周涨跌幅: +5.3%
    • 月涨跌幅: +12.8%
    """

@tool
def get_financial_news(keyword: str, days: int = 7) -> str:
    """获取金融新闻信息"""
    logger.info(f"📰 调用财经新闻工具 - 关键词: {keyword}, 天数: {days}")
    return f"""
    关键词: {keyword}
    时间范围: 最近{days}天
    
    🔥 主要新闻:
    1. 📋 公司发布Q3财报，营收同比增长15%
    2. 🏆 获得重要政府订单，总价值约10亿元
    3. 💰 董事会批准股份回购计划
    4. 📈 分析师上调目标价至$150
    5. 🎯 行业政策利好，相关板块普涨
    """

@tool
def technical_analysis(symbol: str, indicator: str = "MA") -> str:
    """技术分析工具"""
    logger.info(f"📉 调用技术分析工具 - 股票: {symbol}, 指标: {indicator}")
    return f"""
    技术指标分析 - {symbol}
    指标类型: {indicator}
    
    📊 移动平均线:
    • MA5: $123.45 (支撑位)
    • MA20: $118.20 (强支撑)
    • MA60: $115.80 (长期趋势线)
    
    🎯 技术信号:
    • MACD: 金叉信号，多头排列 ✅
    • RSI: 65 (略偏强势区域) ⚠️
    • 成交量: 较前期放大30% 📈
    
    🎪 关键价位:
    • 支撑位: $120.00
    • 阻力位: $130.00
    """

@tool
def portfolio_optimization(assets: str, risk_level: str = "medium") -> str:
    """投资组合优化分析"""
    logger.info(f"📊 调用投资组合优化工具 - 资产: {assets}, 风险级别: {risk_level}")
    return f"""
    💼 投资组合优化结果:
    资产类别: {assets}
    风险水平: {risk_level}
    
    🎯 建议配置:
    • 股票: 60% (蓝筹股40% + 成长股20%)
    • 债券: 30% (政府债券20% + 企业债10%)  
    • 现金: 10%
    
    📈 预期表现:
    • 预期收益: 8-12%
    • 最大回撤: 15%
    • 夏普比率: 1.2
    """

@tool
def risk_assessment(position_size: str, market_cap: str) -> str:
    """风险评估工具"""
    logger.info(f"⚠️ 调用风险评估工具 - 持仓: {position_size}, 市值: {market_cap}")
    return f"""
    🛡️ 风险评估报告:
    持仓规模: {position_size}
    市值规模: {market_cap}
    
    📊 风险指标:
    • VaR (95%): 单日最大损失2.5%
    • Beta系数: 1.2 (高于市场平均)
    • 流动性风险: 低 ✅
    • 信用风险: 中等 ⚠️
    • 行业集中度: 偏高 🔥
    
    💡 风险建议: 适当分散投资，控制单一持仓比例
    """

# ============= 链和代理定义 =============

# 搜索工具
search_tool = TavilySearch(max_results=5, topic="general")

# 基础工具集
basic_tools = [get_stock_data, get_financial_news, technical_analysis, search_tool]

# 高级工具集
advanced_tools = basic_tools + [portfolio_optimization, risk_assessment]

# 财务数据分析规划器
FINANCIAL_PLANNER_INSTRUCTIONS = """
你是一个专业的金融投资分析助手，根据用户的投资分析需求，制定详细的分析计划。
针对不同类型的投资分析需求，给出相应的分析步骤：

分析类型包括：
- 股票分析：基本面分析、技术面分析、行业分析
- 市场趋势：宏观经济分析、市场情绪分析、板块轮动
- 投资组合：风险评估、收益分析、资产配置建议
- 财报分析：盈利能力、偿债能力、成长性分析

只返回XML格式，按照如下格式：
<analysis_plan>
<step>
<name>分析步骤</name>
<method>分析方法</method>
<data_needed>所需数据</data_needed>
</step>
</analysis_plan>
"""

planner_prompt = ChatPromptTemplate.from_messages([
    ("system", FINANCIAL_PLANNER_INSTRUCTIONS), 
    ("human", "{query}")
])

planner_chain = planner_prompt | model

# 数据收集代理
DATA_COLLECTION_INSTRUCTIONS = """
你是金融数据收集专家。根据分析计划中的每个步骤，使用相应的工具收集所需的数据。
收集完成后，对数据进行初步整理和摘要。
"""

data_agent = create_react_agent(
    model=model,
    prompt=DATA_COLLECTION_INSTRUCTIONS,
    tools=basic_tools,
)

# 金融分析报告生成器
ANALYSIS_REPORT_PROMPT = """
你是资深金融分析师，负责撰写专业的投资分析报告。
基于收集的数据和分析计划，生成全面的投资分析报告。

报告要求:
• 使用 Markdown 格式
• 包含投资建议和风险提示
• 至少1000字的详细分析
• 包含图表说明和数据引用
• 给出明确的投资评级和目标价位

仅返回XML格式:
<financial_report>
<executive_summary>执行摘要内容</executive_summary>
<detailed_report>详细报告内容（Markdown格式）</detailed_report>
<investment_rating>买入/持有/卖出</investment_rating>
<target_price>目标价格</target_price>
<risk_factors>
<risk>风险因素1</risk>
<risk>风险因素2</risk>
</risk_factors>
</financial_report>
"""

report_prompt = ChatPromptTemplate.from_messages([
    ("system", ANALYSIS_REPORT_PROMPT), 
    ("human", "{content}")
])

report_chain = report_prompt | model

# 智能规划代理
AGENT_PLANNER_PROMPT = """
你是金融投资的智能规划助手，能够根据用户的具体需求自动选择最合适的分析工具和方法。

你需要:
1. 深入分析用户的具体需求
2. 制定更深入的分析策略
3. 选择合适的高级分析工具
4. 提供个性化的投资建议

可用的高级工具:
- get_stock_data: 获取详细股票数据
- get_financial_news: 收集最新财经资讯
- technical_analysis: 深度技术分析
- portfolio_optimization: 投资组合优化
- risk_assessment: 专业风险评估
- search_tool: 网络搜索补充信息

请根据用户需求智能选择工具组合，提供专业的投资分析建议。
"""

intelligent_agent = create_react_agent(
    model=model,
    prompt=AGENT_PLANNER_PROMPT,
    tools=advanced_tools,
)

def parse_xml_plan(xml_text: str) -> FinancialAnalysisPlan:
    """解析XML格式的分析计划"""
    logger.info("📋 开始解析分析计划XML")
    try:
        xml_text = xml_text.strip()
        if not xml_text.startswith('<'):
            start_idx = xml_text.find('<analysis_plan>')
            if start_idx != -1:
                xml_text = xml_text[start_idx:]
                end_idx = xml_text.find('</analysis_plan>') + len('</analysis_plan>')
                xml_text = xml_text[:end_idx]
        
        root = ET.fromstring(xml_text)
        steps = []
        
        for step_elem in root.findall('step'):
            step = FinancialAnalysisStep(
                step=step_elem.find('name').text if step_elem.find('name') is not None else "默认步骤",
                method=step_elem.find('method').text if step_elem.find('method') is not None else "默认方法",
                data_needed=step_elem.find('data_needed').text if step_elem.find('data_needed') is not None else "基础数据"
            )
            steps.append(step)
        
        logger.info(f"✅ XML解析成功，共解析到 {len(steps)} 个分析步骤")
        return FinancialAnalysisPlan(analysis_steps=steps)
        
    except Exception as e:
        logger.warning(f"⚠️ XML解析失败: {e}，使用默认计划")
        return FinancialAnalysisPlan(
            analysis_steps=[
                FinancialAnalysisStep(
                    step="基本面分析",
                    method="财务指标分析",
                    data_needed="股价、财务数据、市场数据"
                ),
                FinancialAnalysisStep(
                    step="技术面分析", 
                    method="技术指标分析",
                    data_needed="价格走势、技术指标"
                ),
                FinancialAnalysisStep(
                    step="行业分析",
                    method="行业比较分析",
                    data_needed="行业新闻、市场趋势"
                )
            ]
        )

def parse_xml_report(xml_text: str) -> FinancialReport:
    """解析XML格式的分析报告"""
    logger.info("📊 开始解析分析报告XML")
    try:
        xml_text = xml_text.strip()
        if not xml_text.startswith('<'):
            start_idx = xml_text.find('<financial_report>')
            if start_idx != -1:
                xml_text = xml_text[start_idx:]
                end_idx = xml_text.find('</financial_report>') + len('</financial_report>')
                xml_text = xml_text[:end_idx]
        
        root = ET.fromstring(xml_text)
        
        risk_factors = []
        risk_factors_elem = root.find('risk_factors')
        if risk_factors_elem is not None:
            for risk_elem in risk_factors_elem.findall('risk'):
                if risk_elem.text:
                    risk_factors.append(risk_elem.text)
        
        report = FinancialReport(
            executive_summary=root.find('executive_summary').text if root.find('executive_summary') is not None else "默认摘要",
            detailed_report=root.find('detailed_report').text if root.find('detailed_report') is not None else "# 默认报告\n\n分析完成。",
            investment_rating=root.find('investment_rating').text if root.find('investment_rating') is not None else "持有",
            target_price=root.find('target_price').text if root.find('target_price') is not None else "市场价格",
            risk_factors=risk_factors if risk_factors else ["一般风险"]
        )
        
        logger.info("✅ 报告XML解析成功")
        return report
        
    except Exception as e:
        logger.warning(f"⚠️ 报告XML解析失败: {e}，使用默认报告")
        return FinancialReport(
            executive_summary="生成默认分析报告",
            detailed_report="# 默认投资分析报告\n\n基础分析已完成，建议进一步研究。",
            investment_rating="中性", 
            target_price="待定",
            risk_factors=["一般市场风险"]
        )

def financial_planner_node(state: FinancialAnalysisState) -> Dict[str, Any]:
    """制定分析计划"""
    logger.info("🎯 开始执行分析规划节点")
    try:
        user_query = state["messages"][-1].content
        logger.info(f"📝 用户查询: {user_query}")
        
        response = planner_chain.invoke({"query": user_query})
        plan_text = response.content
        logger.info("📋 分析计划生成完成")
        
        plan = parse_xml_plan(plan_text)
        
        # 美化输出
        plan_display = "🎯 **分析计划制定完成**\n\n"
        for i, step in enumerate(plan.analysis_steps, 1):
            plan_display += f"**步骤 {i}: {step.step}**\n"
            plan_display += f"• 分析方法: {step.method}\n"
            plan_display += f"• 所需数据: {step.data_needed}\n\n"
        
        return {
            "messages": [AIMessage(content=plan_display)],
            "analysis_plan": plan,
            "workflow_stage": "planning_complete"
        }
        
    except Exception as e:
        logger.error(f"❌ 规划节点执行失败: {e}")
        default_plan = FinancialAnalysisPlan(
            analysis_steps=[
                FinancialAnalysisStep(
                    step="基础分析",
                    method="综合分析",
                    data_needed="基础数据"
                )
            ]
        )
        return {
            "messages": [AIMessage(content="⚠️ 使用默认分析计划")],
            "analysis_plan": default_plan,
            "workflow_stage": "planning_complete"
        }

def data_collection_node(state: FinancialAnalysisState) -> Dict[str, Any]:
    """数据收集"""
    logger.info("📊 开始执行数据收集节点")
    plan = state["analysis_plan"]
    collection_tasks = []
    
    for i, step in enumerate(plan.analysis_steps, 1):
        task_description = f"步骤{i}: 执行{step.step}，使用{step.method}方法，收集{step.data_needed}"
        collection_tasks.append(task_description)
        logger.info(f"📋 收集任务{i}: {step.step}")
    
    combined_tasks = "\n".join(collection_tasks)
    logger.info("🔄 开始执行数据收集代理")
    run = data_agent.invoke({"messages": [HumanMessage(content=combined_tasks)]})
    
    collected_info = run["messages"][-1].content
    logger.info("✅ 数据收集完成")
    
    return {
        "messages": [AIMessage(content=f"📊 **数据收集完成**\n\n{collected_info}")],
        "collected_data": collected_info,
        "workflow_stage": "data_collected"
    }

def report_generation_node(state: FinancialAnalysisState) -> Dict[str, Any]:
    """生成分析报告"""
    logger.info("📝 开始执行报告生成节点")
    try:
        original_query = state["messages"][0].content
        collected_data = state["collected_data"]
        
        report_input = f"用户需求: {original_query}\n收集的数据: {collected_data}"
        logger.info("🔄 开始生成分析报告")
        
        response = report_chain.invoke({"content": report_input})
        report_text = response.content
        
        report = parse_xml_report(report_text)
        logger.info("✅ 分析报告生成完成")
        
        # 美化报告输出
        report_display = "📊 **投资分析报告**\n\n"
        report_display += f"**🎯 投资评级:** {report.investment_rating}\n"
        report_display += f"**💰 目标价格:** {report.target_price}\n\n"
        report_display += f"**📋 执行摘要:**\n{report.executive_summary}\n\n"
        report_display += f"**⚠️ 主要风险因素:**\n"
        for risk in report.risk_factors:
            report_display += f"• {risk}\n"
        report_display += f"\n**📈 详细报告:**\n{report.detailed_report}"
        
        return {
            "messages": [AIMessage(content=report_display)],
            "report": report,
            "workflow_stage": "report_generated"
        }
        
    except Exception as e:
        logger.error(f"❌ 报告生成失败: {e}")
        default_report = FinancialReport(
            executive_summary="生成默认分析报告",
            detailed_report="# 默认投资分析报告\n\n基础分析已完成，建议进一步研究。",
            investment_rating="中性", 
            target_price="待定",
            risk_factors=["一般市场风险"]
        )
        return {
            "messages": [AIMessage(content="⚠️ 生成默认分析报告")],
            "report": default_report,
            "workflow_stage": "report_generated"
        }

def intelligent_agent_node(state: FinancialAnalysisState) -> Dict[str, Any]:
    """智能代理深度分析"""
    logger.info("🤖 开始执行智能代理深度分析节点")
    
    original_query = state["messages"][0].content
    existing_report = state.get("report")
    
    agent_task = f"""
    原始投资分析需求: {original_query}
    
    基础分析已完成，现在需要提供更深入、更全面的投资分析建议。
    请使用高级分析工具进行深度分析，包括：
    1. 更详细的风险评估
    2. 投资组合优化建议
    3. 市场前景预测
    4. 个性化投资策略
    """
    
    logger.info("🔄 开始执行智能代理分析")
    run = intelligent_agent.invoke({"messages": [HumanMessage(content=agent_task)]})
    agent_response = run["messages"][-1].content
    logger.info("✅ 智能代理分析完成")
    
    # 美化深度分析输出
    final_output = f"🤖 **智能深度分析报告**\n\n{agent_response}\n\n"
    final_output += "=" * 50 + "\n"
    final_output += "🎉 **分析完成！** 感谢使用金融投资分析系统\n"
    final_output += "💡 如需进一步分析，请提出新的问题"
    
    return {
        "messages": [AIMessage(content=final_output)],
        "workflow_stage": "analysis_completed"
    }

# ============= 构建工作流图 =============

def build_financial_analysis_graph():
    """构建金融分析工作流图"""
    logger.info("🏗️ 开始构建金融分析工作流图")
    builder = StateGraph(FinancialAnalysisState)
    
    # 添加节点
    builder.add_node("planner", financial_planner_node)
    builder.add_node("data_collection", data_collection_node)  
    builder.add_node("report_generation", report_generation_node)
    builder.add_node("intelligent_agent", intelligent_agent_node)
    
    # 设置工作流边
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "data_collection")
    builder.add_edge("data_collection", "report_generation") 
    builder.add_edge("report_generation", "intelligent_agent")
    builder.add_edge("intelligent_agent", END)
    
    logger.info("✅ 金融分析工作流图构建完成")
    return builder.compile()

graph = build_financial_analysis_graph()
