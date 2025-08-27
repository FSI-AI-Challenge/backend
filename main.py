from langgraph.graph import StateGraph, add_messages, END
from langgraph.types import interrupt, Command
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from langchain_ollama import ChatOllama
import json
import time

llm = ChatOllama(model='gpt-oss:20b', streaming=True)

@dataclass
class Goal:
    target_amount: int
    target_months: int         

@dataclass
class IncomeExpense:
    fixed_income: int                  # 월급(세후 등 기준 통일)
    fixed_expense: int                   # 고정지출(월)

class GraphState(TypedDict): 
    user_id: int
    created_ts: str
    question:str
    answer:str

    # 입력/프로필
    goal: Goal
    investable_amount: int

    # 대화 메시지 (LangChain messages)
    messages: Annotated[List, add_messages]

def start(state:GraphState) -> GraphState:
    return GraphState()

def chatbot(state:GraphState) -> GraphState:
    print("챗봇 시작")
    question = state['question']
    response = llm.invoke(question)
    print(f"챗봇 종료: {response.content}")
    return GraphState(answer=response.content)

def get_goal(state:GraphState) -> GraphState:
    print("목표 금액, 기간 추출 시작")
    question = state['question']

    prompt = f"""
        Extract the user's savings goal.
        Return ONLY valid JSON, no markdown, no code block.
        Keys:
        - target_amount (int)
        - target_months (int)
        User input: {question}
    """

    response = llm.invoke(prompt)
    data = json.loads(response.content)
    
    print(f"목표 금액, 기간 추출 종료: {data}")
    return GraphState(
        goal=Goal(
            target_amount=int(data["target_amount"]), 
            target_months=int(data["target_months"])
        )
    )

def load_profile(state:GraphState) -> GraphState:
    print("사용자 수입 및 지출 계산 시작")
    # 추후 api로 대체
    user_id = state['user_id']
    with open(f"./data/user_example_data{user_id}.json", "r", encoding="utf-8") as f:
        user_profile = json.load(f)

    ts = datetime.fromisoformat(state["created_ts"])
    current_year = ts.year
    current_month = ts.month

    recent_months = []
    for i in range(3):
        month = (current_month - i - 1) % 12 + 1
        year = current_year if current_month - i > 0 else current_year - 1
        recent_months.append(f"{year}-{month:02d}")

    recent_months = set(recent_months)

    filtered_months = [
        m for m in user_profile["months"] if m["month"] in recent_months
    ]

    template = '''
        You are an assistant that analyzes personal finance transaction data.

        Tasks:
        1) Identify the average fixed income per month.
        2) Identify the average fixed expenses per month.
        3) Identify the average variable expenses per month.
        4) Compute the average investable amount per month = fixed_income - (fixed_expenses + variable_expenses).

        Return ONLY valid JSON, no markdown, no code block. 
        Keys:
        - fixed_income (int)
        - fixed_expenses (int)
        - variable_expenses (int)
        - investable_amount (int)

        User Input:
        {}
    '''
    prompt = template.format(filtered_months)

    response = llm.invoke(prompt)
    print(response)
    data = json.loads(response.content)

    print(f"사용자 수입 및 지출 계산 종료: {data}")
    return GraphState(
        investable_amount=int(data["investable_amount"])
    )

def hitl_confirm_input(state:GraphState) -> GraphState:
    print("사용자 입력 검증 시작")
    proposed = {
        "target_amount": state["goal"].target_amount,
        "target_months": state["goal"].target_months,
        "investable_amount": state["investable_amount"]
    }
    
    decision = interrupt({
        "step": "confirm_input",
        "message": "목표 금액/기간, 투자 가능 금액을 확인 및 수정해주세요.",
        "proposed": proposed,
        "fields": [
            {"name": "target_amount", "type": "number", "label": "목표 금액(원)"},
            {"name": "target_months", "type": "number", "label": "목표 기간(개월)"},
            {"name": "investable_amount", "type": "number", "label": "투자 가능 금액(원)"},
        ],
        "buttons": ["submit"]
    })
    
    target_amount = int(decision.get("target_amount", proposed["target_amount"]))
    target_months = int(decision.get("target_months", proposed["target_months"]))
    investable_amount = int(decision.get("investable_amount", proposed["investable_amount"]))

    if target_amount < 0 or target_months < 0 or investable_amount < 0:
        raise ValueError("입력 값이 유효하지 않습니다.")

    print(f"사용자 입력 검증 종료: {target_amount, target_months, investable_amount}")
    return GraphState(
        goal=Goal(
            target_amount=int(target_amount),
            target_months=int(target_months),
        ),
        investable_amount=int(investable_amount)
    )

def is_our_service(state:GraphState) -> str:
    print("서비스 판단 시작")
    question = state['question']

    prompt = f'''
        You are a classifier. Decide whether the user's message expresses intent to use a savings or investment guidance service.
        Answer only yes or no.

        User Input: {question}
    '''
    response = llm.invoke(prompt)
    print(f"서비스 판단 종료: {response.content}")
    return response.content

graph = StateGraph(GraphState)

graph.add_node("start", start)
graph.add_node("chatbot", chatbot)
graph.add_node("get_goal", get_goal)
graph.add_node("load_profile", load_profile)
graph.add_node("hitl_confirm_input", hitl_confirm_input)

graph.set_entry_point("start")
graph.add_conditional_edges(
    "start",
    is_our_service,
    {
        "yes":"get_goal",
        "no":"chatbot"
    }
)
graph.add_edge("get_goal", "load_profile")
graph.add_edge("load_profile", 'hitl_confirm_input')
graph.add_edge("hitl_confirm_input", END)
graph.add_edge("chatbot", END)

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid

KST = timezone(timedelta(hours=9))

user_id = 1
target_amount = 1000000
target_months = 6
created_ts = datetime.now(KST).isoformat()

config = RunnableConfig(recursion_limit=10, configurable={"thread_id":random_uuid()})

from typing import List, Literal, AsyncGenerator, Any
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import asyncio
import os

# === (A) 개발 시 CORS 허용 ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def chunk_to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # LangChain 계열
    c = getattr(x, "content", None)
    if isinstance(c, str):
        return c
    if hasattr(x, "text"):
        t = getattr(x, "text")
        if isinstance(t, str):
            return t
    if isinstance(x, dict):
        return x.get("content") or x.get("text") or x.get("output") or ""
    return ""

def last_user_message(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

@app.post("/api/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()
    question = last_user_message(body.get("messages", []))

    # ⚠️ TypedDict(GraphState)를 생성자처럼 호출하지 말고 dict 사용
    state_in = {
        "user_id": user_id,          # 너가 이미 모듈 전역에 둔 값들 그대로 사용
        "created_ts": created_ts,
        "question": question,
        # 필요 시 다른 상태 키도 추가 가능: "messages": [], "goal": None, ...
    }

    async def eventgen():
        yield 'data: {"delta":" "}\n\n'  # 핸드셰이크 (프론트 onDelta 트리거)
        final_text = ""
        try:
            async for ev in agent.astream_events(state_in, config=config, version="v1"): 
                # print(ev) # ev로 마지막 노드명 확인해서 마지막 값만 출력
                etype = ev.get("event")
                name = ev.get("name")
                if etype in ("on_chain_end", "on_graph_end") and name == "LangGraph":
                    data = ev.get("data")
                    output = data.get("output")
                    chatbot = output.get("chatbot")
                    final_text = chatbot.get("answer")
                    break
        except Exception as e:
            final_text = f"[server error] {type(e).__name__}: {e}"
        finally:
            if final_text:
                for ch in final_text:
                    yield f'data: {json.dumps({"delta": ch}, ensure_ascii=False)}\n\n'
                    await asyncio.sleep(0)
            else:
                yield 'data: {"delta":"[empty final output]"}\n\n'
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        eventgen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )