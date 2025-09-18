from langgraph.types import Command
from datetime import datetime, timezone, timedelta
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from progress import PROGRESS_MAP
from utils.node import *
from utils.state import *
from graph import agent
import json
import traceback
import asyncio

KST = timezone(timedelta(hours=9))

user_id = 1
created_ts = datetime.now(KST).isoformat()
config = RunnableConfig(recursion_limit=10, configurable={"thread_id":random_uuid()})

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