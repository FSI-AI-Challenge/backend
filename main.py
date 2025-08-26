# backend/main.py
from typing import List, Literal, AsyncGenerator
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

# === (B) 메시지 스키마 ===
Role = Literal["user", "assistant", "system"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    content: str

# === (C) 간단 REST 응답 (/api/chat) ===
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 가장 마지막 user 메시지 추출
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    # TODO: 여기서 LLM 호출/업무 로직 수행
    reply = f"[REST] 백엔드가 받은 메시지: {last_user}"
    return ChatResponse(content=reply)

# === (D) 스트리밍 응답 (/api/chat/stream, SSE) ===
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

    async def eventgen() -> AsyncGenerator[str, None]:
        # TODO: 실제 LLM 스트리밍 토큰을 yield 하면 됨
        fake = f"[SSE] 받음: {last_user}  → 토큰 단위 스트리밍 예시."
        for ch in fake:
            yield f"data: {{\"delta\": \"{ch}\"}}\n\n"
            await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"

    return StreamingResponse(eventgen(), media_type="text/event-stream")

# === (E) 헬스체크 ===
@app.get("/healthz")
def health():
    return {"ok": True}
