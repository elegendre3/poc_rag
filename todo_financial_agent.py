from dotenv import load_dotenv

from phi.agent import Agent
from phi.llm.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

portfolio = {
    "NVDA": 9.26,
    "XOM": 5.38,
    "GOOGL": 2.49,
    "ABT": 2.51,
    "RDDT": 2.02,
    "QCOM": 1.,
    "ORCL": 0.61,
    "INTC": 3.77,
    "RL": 0.14
}

agent = Agent(
    llm=OpenAIChat(model="gpt-4o-mini"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    show_tool_calls=True,
    markdown=True,
)

# OpenAiChat (from Phidata) is different from ChatOpenAI from langchain
non_agent = ChatOpenAI(model="gpt-4o-mini") | StrOutputParser() 

# agent.print_response(
  # "Write a comparison between the top chip manufactureres, use all tools available."
# )

if __name__ == "__main__":
  # q =  f"Considering the following portfolio {portfolio}, what are the analysts view of it? And analysis regarding buying, holding or selling?"
  q =  f"Considering the following portfolio {portfolio}, what is the current value of the portfolio?"

  print(non_agent.invoke(q))
  print(" ------------------------------------ ")
  print(" ------------------------------------ ")
  agent.print_response(q)
