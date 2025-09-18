from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, add_messages, END
from langgraph.types import interrupt, Command
from langchain_ollama import ChatOllama
from langgraph.types import Command
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from utils.state import *
from utils.node import *
from langgraph.types import Command

graph = StateGraph(GraphState)

# 승현
graph.add_node("planner", planner)
graph.add_node("chatbot", chatbot)
graph.add_node("get_goal", get_goal)
graph.add_node("load_profile", load_profile)
graph.add_node("hitl_confirm_input", hitl_confirm_input)
# 주엽
graph.add_node("select_fin_prdt", select_fin_prdt)
graph.add_node("select_stock_products", select_stock_products)
graph.add_node("build_indicators", build_indicators)
graph.add_node("build_portfolios", build_portfolios)
# 지수
graph.add_node("crawl_news", crawl_news)
graph.add_node("summarize_news", summarize_news)
graph.add_node("analyze_sentiment", analyze_sentiment)
graph.add_node("evaluate_rebalance", evaluate_rebalance)

graph.set_entry_point("planner")
graph.add_conditional_edges(
    "planner",
    lambda s: s.get("route", "chatbot"),
    {
        "get_goal":"get_goal",
        "chatbot":"chatbot"
    }
)
graph.add_edge("get_goal", "load_profile")
graph.add_edge("load_profile", 'hitl_confirm_input')
graph.add_edge("hitl_confirm_input", "select_fin_prdt")
graph.add_edge("select_fin_prdt", "select_stock_products")
graph.add_edge("select_stock_products", "build_indicators")
graph.add_edge("build_indicators", "build_portfolios")
graph.add_edge("build_portfolios", END)
graph.add_edge("chatbot", END)

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)