from typing import List
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from tools.a_stock_data_tool import AStockDataTool
from tools.financial_tool import FinancialAnalysisTool
from tools.market_sentiment_tool import MarketSentimentTool
from tools.calculator_tool import CalculatorTool

import os
from dotenv import load_dotenv
load_dotenv()

# from langchain.llms import Ollama
# llm = Ollama(model="llama3.1")

# 从环境变量读取模型配置
model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
temperature = float(os.getenv("TEMPERATURE", "0.8"))
max_tokens = int(os.getenv("MAX_TOKENS", "14000"))


import os
from crewai import LLM

# 1. 从系统环境变量中读取 Key（变量名建议统一叫 GEMINI_API_KEY）
api_key = os.getenv("GEMINI_API_KEY")

# 2. 把读取到的 Key 传给 Gemini 模型
llm = LLM(
    model="gemini/gemini-1.5-pro",
    api_key=api_key,
    temperature=0.5
)



@CrewBase
class AStockAnalysisCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def a_stock_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['a_stock_analyst'],
            verbose=True,
            llm=llm,
            tools=[
                AStockDataTool(),
                FinancialAnalysisTool(),
                CalculatorTool(),
            ]
        )

    @task
    def market_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['market_analysis'],
            agent=self.a_stock_analyst(),
        )

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_analyst'],
            verbose=True,
            llm=llm,
            tools=[
                AStockDataTool(),
                FinancialAnalysisTool(),
                CalculatorTool(),
            ]
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['financial_analysis'],
            agent=self.financial_analyst(),
        )

    @agent
    def market_sentiment_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['market_sentiment_analyst'],
            verbose=True,
            llm=llm,
            tools=[
                AStockDataTool(),
                MarketSentimentTool(),
            ]
        )

    @task
    def sentiment_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['sentiment_analysis'],
            agent=self.market_sentiment_agent(),
        )

    @agent
    def investment_advisor(self) -> Agent:
        return Agent(
            config=self.agents_config['investment_advisor'],
            verbose=True,
            llm=llm,
            tools=[
                CalculatorTool(),
            ]
        )

    @task
    def investment_recommendation(self) -> Task:
        return Task(
            config=self.tasks_config['investment_recommendation'],
            agent=self.investment_advisor(),
        )

    @crew
    def crew(self) -> Crew:
        """创建A股分析团队"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )