from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from utils.state import *
from utils.node import *

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
        "chatbot":"chatbot", 
        "crawl_news":"crawl_news"
    }
)
graph.add_edge("get_goal", "load_profile")
graph.add_edge("load_profile", 'hitl_confirm_input')
graph.add_edge("hitl_confirm_input", "select_fin_prdt")
graph.add_edge("select_fin_prdt", "select_stock_products")
graph.add_conditional_edges(
    "select_stock_products",
    lambda s: "rebalance" if int(s.get("months_passed", 0)) > 0 else "initial",
    {
        "rebalance": "evaluate_rebalance",  # 리밸런싱
        "initial": "build_indicators",      
    }
)
graph.add_edge("build_indicators", "build_portfolios")
graph.add_edge("crawl_news", "summarize_news")
graph.add_conditional_edges(
    "crawl_news",
    lambda s: "summarize_news" if len(s.get("news_signals")) == 0 else "fail",
    {
        "summarize_news": "summarize_news",
        "fail": END
    }
)
graph.add_edge("summarize_news", "analyze_sentiment")
graph.add_conditional_edges(
    "analyze_sentiment",
    lambda s: "positive" if int(s.get("majority_sentiment", 0)) == 1 else "negative",
    {
        "positive": END,                       
        "negative": "select_stock_products",   
    }
)
graph.add_edge("evaluate_rebalance", END)
graph.add_edge("build_portfolios", END)
graph.add_edge("chatbot", END)

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)