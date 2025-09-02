from langgraph.graph import StateGraph, add_messages, END
from langgraph.types import interrupt, Command
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import json
import traceback
from progress import PROGRESS_MAP

llm = ChatOllama(model='gpt-oss:20b', streaming=True)

@dataclass
class Goal:
    target_amount: int
    target_months: int         

class GraphState(TypedDict): 
    user_id: int
    created_ts: str
    question:str
    answer:str

    route:str

    # 입력/프로필
    goal: Goal
    investable_amount: int

    # 대화 메시지 (LangChain messages)
    messages: Annotated[List, add_messages]

def planner(state:GraphState) -> GraphState:
    print("서비스 판단 시작")
    question = state['question']

    prompt = f'''
        You are a routing agent that decides the next step in a workflow.  

        Decide which node to go to next based on the user input.  
        You MUST choose exactly one of the following nodes:
        - "get_goal": when the user is asking about our financial planning / savings / investment service.
        - "chatbot": when the user is making a general request or any query not related to our service.

        User Input:
        {question}

        Return ONLY node name:
        "get_goal" or "chatbot"
    '''
    response = llm.invoke(prompt)
    print(f"서비스 판단 종료: {response.content}")
    return GraphState(route=response.content)

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
    recent_keys = [(ts - relativedelta(months=i)).strftime("%Y-%m") for i in range(1, 4)]
    months = [m for m in user_profile["months"] if m["month"] in recent_keys]

    unique_pairs = sorted({(d["payee"], d["type"].upper()) for m in months for d in m["days"]})
    items_for_llm = [{"payee": p, "direction": t} for (p, t) in unique_pairs]

    template = """
        You are a strict finance transaction labeler.

        For each item, classify into EXACTLY one category based on its direction:

        - If direction == "INCOME":
            choose one of: ["FIXED_INCOME", "OTHER"]
            (salary/wages/regular payroll/interest/dividends → FIXED_INCOME)

        - If direction == "EXPENSE":
            choose one of: ["FIXED_EXPENSE", "VARIABLE_EXPENSE", "OTHER"]
            (rent/insurance/telecom/utilities/loan/subscription → FIXED_EXPENSE;
            food/shopping/entertainment/transport/leisure → VARIABLE_EXPENSE)

        Do NOT infer direction yourself; use the provided "direction".

        Return ONLY valid JSON array like:
        [
        {{"payee":"청계하우스","category":"FIXED_EXPENSE"}},
        {{"payee":"ABC주식회사","category":"FIXED_INCOME"}}
        ]

        Items to classify:
        {items}
    """
    prompt = template.format(items=json.dumps(items_for_llm, ensure_ascii=False))
    response = llm.invoke(prompt)
    data = json.loads(response.content)
    mapping = {x["payee"]: x["category"] for x in data}

    sums = defaultdict(int)
    for m in months:
        for d in m["days"]:
            cat = mapping.get(d["payee"], "OTHER")
            amt = d["amount"]
            if d["type"] == "income":
                sums[cat] += amt
            elif d["type"] == "expense":
                sums[cat] -= amt

    fixed_income = sums["FIXED_INCOME"] // len(months)
    fixed_expenses = abs(sums["FIXED_EXPENSE"]) // len(months)
    variable_expenses = abs(sums["VARIABLE_EXPENSE"]) // len(months)
    investable_amount = fixed_income - (fixed_expenses + variable_expenses)
    print(f"사용자 수입 및 지출 계산 종료: {investable_amount}")
    return GraphState(
        investable_amount=investable_amount
    )

def hitl_confirm_input(state:GraphState) -> GraphState:
    print("사용자 입력 검증 시작")
    decision = interrupt({
        "step": "confirm_input",
        "message": "목표 금액/기간, 투자 가능 금액을 확인 및 수정해주세요.",
        "proposed": {
            "target_amount": state["goal"].target_amount,
            "target_months": state["goal"].target_months,
            "investable_amount": state["investable_amount"],
        },
        "fields": [
            {"name": "target_amount", "type": "number", "label": "목표 금액(원)"},
            {"name": "target_months", "type": "number", "label": "목표 기간(개월)"},
            {"name": "investable_amount", "type": "number", "label": "투자 가능 금액(원)"},
        ],
        "buttons": ["submit"]
    })
    target_amount = int(decision.get("target_amount", state["goal"].target_amount))
    target_months = int(decision.get("target_months", state["goal"].target_months))
    investable_amount = int(decision.get("investable_amount", state["investable_amount"]))

    print(f"사용자 입력 검증 종료: {target_amount, target_months, investable_amount}")
    return GraphState(
        goal=Goal(
            target_amount=int(target_amount),
            target_months=int(target_months),
        ),
        investable_amount=int(investable_amount)
    )

graph = StateGraph(GraphState)

graph.add_node("planner", planner)
graph.add_node("chatbot", chatbot)
graph.add_node("get_goal", get_goal)
graph.add_node("load_profile", load_profile)
graph.add_node("hitl_confirm_input", hitl_confirm_input)

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
graph.add_edge("hitl_confirm_input", END)
graph.add_edge("chatbot", END)

memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

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
from langgraph.types import Command
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

@app.post("/api/chat/stream")
async def chat_stream(req: Request):
    body = await req.json()
    
    if 'resume' in body:
        events = agent.astream_events(Command(resume=body.get('resume')), config=config, version="v1")
    else:
        question = next((m.get("content", "") for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
        state_in = {
            "user_id": user_id,          
            "created_ts": created_ts,
            "question": question,
        }
        events = agent.astream_events(state_in, config=config, version="v1")

    async def eventgen():
        try:
            async for ev in events: 
                # print(ev) # ev로 마지막 노드명 확인하고 노드명도 변경해서 마지막 값만 출력
                etype = ev.get("event")
                node = ev.get("name")

                # 🔹 진행 상태 전송
                if etype == "on_chain_start":
                    step = PROGRESS_MAP.get(node)
                    if step:
                        sid, label = step
                        yield f'data: {json.dumps({"kind":"progress","id":sid,"status":"running","label":label}, ensure_ascii=False)}\n\n'

                if etype == "on_chain_end":
                    step = PROGRESS_MAP.get(node)
                    if step:
                        sid, _ = step
                        yield f'data: {json.dumps({"kind":"progress","id":sid,"status":"done"}, ensure_ascii=False)}\n\n'

                    data = ev.get("data")
                    output = data.get("output")

                    if isinstance(output, dict):
                        if '__interrupt__' in output:
                            raw = output['__interrupt__']
                            intr_obj = raw[0] if isinstance(raw, (list, tuple)) else raw
                            intr_payload = {
                                "value": getattr(intr_obj, "value", intr_obj),
                                "id": getattr(intr_obj, "id", None),
                            }
                            yield 'data: {"kind":"done"}\n\n'
                            yield f'data: {json.dumps({"kind":"interrupt","payload": intr_payload}, ensure_ascii=False)}\n\n'
                            return

                        # 일반 답변(예: chatbot 단계)을 delta로 흘려보내기
                        out_text = ""
                        if 'chatbot' in output:
                            out_text = output.get("chatbot", {}).get("answer", "")
                        # 필요시 다른 노드 출력도 합치기
                        if out_text:
                            yield 'data: {"kind":"done"}\n\n'
                            for ch in out_text:
                                yield f'data: {json.dumps({"delta": ch}, ensure_ascii=False)}\n\n'
                                await asyncio.sleep(0.02)
                            return
        except Exception as e:
            yield f'data: {json.dumps({"delta": f"[server error] {type(e).__name__}: {e}"}, ensure_ascii=False)}\n\n'
            yield 'data: {"kind":"done"}\n\n'
            traceback.print_exc()
            return

    return StreamingResponse(
        eventgen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )