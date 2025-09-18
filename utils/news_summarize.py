from __future__ import annotations

import json
import os
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from dataclasses import dataclass
from enum import Enum
from email.utils import parsedate_to_datetime

from FSI.utils.state import (
    GraphState, NewsSignal, Sentiment, append_news_signals, initial_state
)

# LangChain / LangGraph
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

# Naver community tool
from langchain_naver_community.tool import NaverNewsSearch

# ENV & LLM
load_dotenv()
NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# 모델은 필요에 따라 변경 가능합니다.
llm = ChatOllama(model="gpt-oss:20b") 
MAX_CONTENT_CHARS = 1200


def get_naver_tool() -> NaverNewsSearch:
    if not NAVER_ID or not NAVER_SECRET:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 없음")
    
    return NaverNewsSearch(naver_client_id=NAVER_ID, naver_client_secret=NAVER_SECRET)

naver_tool = get_naver_tool()

# LangGraph 파이프라인
class NewsGraphState(TypedDict, total=False):
    # 입력
    query: str
    n_items: int
    k_sentences: int

    # 진행 관리
    attempts: int
    max_attempts: int
    collected: List[Dict]
    cleaned: List[Dict]
    summaries: List[str]
    dropped: int

    # 결과(JSON 직전)
    items: List[Dict[str, Any]]


class SummaryAgentState(TypedDict, total=False):
    content: str  # 입력 기사 내용
    summary_strategy: str  # LLM이 결정할 요약 전략
    summary: str  # 최종 요약 결과

def node_decide_strategy(state: SummaryAgentState) -> SummaryAgentState:
    """기사 내용을 분석하여 요약 전략을 결정하는 노드"""
    prompt = ChatPromptTemplate.from_template(
        """다음 뉴스 기사를 분석하고, 어떤 방식으로 요약하는 것이 가장 효과적인지 한두 문장으로 설명해줘.
        예시:
        - "주요 재무 지표와 시장 반응을 중심으로 요약합니다."
        - "사건의 발생 경위와 영향을 중심으로 요약합니다."
        - "제품의 핵심 기능과 특징을 중심으로 요약합니다."
        
        기사 본문:
        {content}
        
        결정:"""
    )
    chain = prompt | llm
    
    strategy = chain.invoke({"content": state["content"]}).content
    state["summary_strategy"] = strategy
    print(f"[DEBUG] 결정된 요약 전략: {strategy}", flush=True)
    return state

def node_summarize_with_strategy(state: SummaryAgentState) -> SummaryAgentState:
    """결정된 전략에 따라 기사를 요약하는 노드"""
    prompt = ChatPromptTemplate.from_template(
        """다음 지침에 따라 기사를 요약해줘.
        지침: {strategy}
        
        기사 본문:
        {content}
        
        요약:"""
    )
    chain = prompt | llm
    
    summary = chain.invoke({
        "strategy": state["summary_strategy"], 
        "content": state["content"]
    }).content
    state["summary"] = summary
    return state
    
def build_summary_agent_graph():
    """Agentic 요약 서브 그래프를 구축합니다."""
    g = StateGraph(SummaryAgentState)
    g.add_node("decide_strategy", node_decide_strategy)
    g.add_node("summarize_with_strategy", node_summarize_with_strategy)
    g.add_edge(START, "decide_strategy")
    g.add_edge("decide_strategy", "summarize_with_strategy")
    g.add_edge("summarize_with_strategy", END)
    return g.compile()

# Agentic 요약 서브 그래프 인스턴스
summary_agent = build_summary_agent_graph()

# 유틸리티

_TAG_RE = re.compile(r"</?[^>]+>")

def _strip_tags(s: str) -> str:
    return _TAG_RE.sub("", s or "").strip()

def _to_iso8601(s: str) -> str:
    """RFC822 → ISO8601. 실패 시 YYYY-MM-DD만 추출."""
    if not s:
        return ""
    try:
        dt = parsedate_to_datetime(s)
        return dt.isoformat()
    except Exception:
        m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})", s)
        if m:
            y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
            return f"{y}-{mo}-{d}"
        return ""

def _normalize_naver_payload(raw: Any) -> dict:
    """NaverNewsSearch.run 반환을 항상 {'items':[...]}로 통일."""
    try:
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = {"items": []}
        elif isinstance(raw, list):
            data = {"items": raw}
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {"items": []}
    except Exception:
        data = {"items": []}

    if not isinstance(data, dict):
        data = {"items": []}
    if not isinstance(data.get("items"), list):
        data["items"] = []
    return data

def _fingerprint(title: str, link: str) -> str:
    base = (title or "").strip().lower() + "|" + (link or "").strip().lower()
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def _dedup_and_clean(items: List[Dict[str, Any]]) -> Tuple[List[Dict], int]:
    seen, out, dropped = set(), [], 0
    for it in items:
        title = _strip_tags(it.get("title") or it.get("headline") or "")
        link = (it.get("originallink") or it.get("link") or it.get("url") or "").strip()
        content = _strip_tags(it.get("content") or it.get("description") or it.get("summary") or "")
        pub_raw = it.get("published_at") or it.get("pubDate") or it.get("date") or ""
        pub = _to_iso8601(pub_raw)

        if not title or not link:
            dropped += 1
            continue

        fp = _fingerprint(title, link)
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)

        out.append({"title": title, "link": link, "content": content, "published_at": pub})
    return out, dropped

# 쿼리 변형
REFINE_LIST = [
    "{q}",
    "{q} 실적 OR 공시",
    "{q} 뉴스 -블로그 -카페",
    "site:news.naver.com {q}",
]
def _refined_query(q: str, attempt: int) -> str:
    pat = REFINE_LIST[min(attempt, len(REFINE_LIST)-1)]
    return pat.replace("{q}", q)


# 메인 LangGraph 노드
def node_search(state: NewsGraphState) -> NewsGraphState:
    """항상 display=N. 재시도 때 start 오프셋 증가, 쿼리 리파인."""
    base_q = state["query"]
    N = state["n_items"]
    attempts = state.get("attempts", 0)

    q = _refined_query(base_q, attempts)
    start = 1 + attempts * N  

    try:
        raw = naver_tool.run(q, display=N, start=start, sort="date")
    except TypeError:
        raw = naver_tool.run(q, display=N)

    payload = _normalize_naver_payload(raw)
    items = payload.get("items", [])

    print(f"[DEBUG] attempt={attempts+1}, query='{q}', fetched={len(items)}, start={start}", flush=True)
    state["collected"] = items

    state["attempts"] = attempts + 1
    return state

def node_clean_and_check(state: NewsGraphState) -> NewsGraphState:
    collected = state.get("collected", [])
    print(f"[DEBUG] collected_total={len(collected)} before clean", flush=True)
    cleaned, dropped = _dedup_and_clean(collected)
    print(f"[DEBUG] cleaned={len(cleaned)}, dropped={dropped}", flush=True)
    state["cleaned"] = cleaned
    state["dropped"] = dropped
    return state

def route_retry(state: NewsGraphState) -> str:
    N = state["n_items"]
    max_attempts = state["max_attempts"]
    cleaned = state.get("cleaned", [])
    attempts = state.get("attempts", 0)

    if len(cleaned) >= N:
        return "enough"
    if attempts >= max_attempts:
        return "exhausted"
    return "retry"

def node_summarize(state: NewsGraphState) -> NewsGraphState:
    """Agentic 요약 에이전트를 사용하여 기사들을 요약하는 노드"""
    N = state["n_items"]
    cleaned = state.get("cleaned", [])[:N]
    
    final_items = []
    
    # 각 기사별로 Agentic 요약 에이전트 실행
    for i, it in enumerate(cleaned):
        content = str(it.get("content", ""))[:MAX_CONTENT_CHARS]
        if not content:
            continue
            
        print(f"[DEBUG] 기사 #{i+1} 요약 시작...", flush=True)
        
        # Agentic 요약 에이전트 호출
        res: SummaryAgentState = summary_agent.invoke({"content": content})
        
        final_items.append({
            "title": it["title"],
            "link": it["link"],
            "published_at": it.get("published_at", ""),
            "summary": res.get("summary", "")
        })
        
    state["items"] = final_items
    state["summaries"] = [it.get("summary", "") for it in final_items]
    return state

def node_finalize(state_ng: NewsGraphState, app_state: Optional[GraphState] = None) -> NewsGraphState:
    """전역 GraphState에 뉴스 시그널 적재."""
    if app_state is not None:
        signals = [
            NewsSignal(
                ticker=state_ng["query"],
                summary=it.get("summary", ""),
                sentiment=Sentiment.UNKNOWN,
                decision=None
            ) for it in state_ng.get("items", [])
        ]
        append_news_signals(app_state, signals)
    return state_ng

def build_news_graph():
    g = StateGraph(NewsGraphState)
    g.add_node("search", node_search)
    g.add_node("clean", node_clean_and_check)
    g.add_node("summarize", node_summarize)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "search")
    g.add_edge("search", "clean")
    g.add_conditional_edges("clean", route_retry,
                            {"retry": "search", "enough": "summarize", "exhausted": "summarize"})
    g.add_edge("summarize", "finalize")
    g.add_edge("finalize", END)
    return g.compile()

def run_news_graph(
    app_state: GraphState,
    query: str,
    n_items: int = 5,
    k_sentences: int = 3,
    max_attempts: int = 3,
) -> Tuple[GraphState, Dict[str, Any]]:
    graph = build_news_graph()
    init: NewsGraphState = {
        "query": query,
        "n_items": n_items,
        "k_sentences": k_sentences,
        "attempts": 0,
        "max_attempts": max_attempts,
        "collected": [],
        "cleaned": [],
        "summaries": [],
        "dropped": 0,
        "items": []
    }
    out: NewsGraphState = graph.invoke(init)
    node_finalize(out, app_state)  # 전역 GraphState 반영

    result_json = {
        "items": out.get("items", []),
        "attempts": out.get("attempts", 0),
        "dropped": out.get("dropped", 0),
    }
    return app_state, result_json

# 데모
if __name__ == "__main__":
    st = initial_state()  # 초기화
    st, result = run_news_graph(
        app_state=st,
        query="구글 주가",
        n_items=2, 
        k_sentences=2,
        max_attempts=3
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)