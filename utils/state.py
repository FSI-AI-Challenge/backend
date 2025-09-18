# python >=3.10
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import add_messages



class UIStep(str, Enum):
    INPUT_GOAL = "INPUT_GOAL"                     # 목표 금액/기간 입력
    INPUT_PERCENT = "INPUT_PERCENT"               # % 입력 (투자 가능 금액 안내)
    ASK_HEART = "ASK_HEART"                       # 마음 물어보기
    SHOW_PORTFOLIOS = "SHOW_PORTFOLIOS"           # 30/50/70% 시나리오 제시
    CONFIRM_PORTFOLIO = "CONFIRM_PORTFOLIO"       # 사용자 선택 대기
    REBALANCING = "REBALANCING"                   # 리밸런싱 단계
    CHATBOT = "CHATBOT"                           # 챗봇 상호작용
    DONE = "DONE"                                 # 종료

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class NewsDecision(str, Enum):
    HOLD = "hold"          # 그대로 보유
    SELL = "sell"          # 매도
    REVIEW = "review"      # 재검토(추가 분석 필요)
    RECOMMEND_REFRESH = "recommend_refresh"  # 추천 알고리즘 다시 시작

@dataclass
class Goal:
    target_amount: int
    target_months: int         

@dataclass
class IncomeExpense:
    fixed_income: int                  # 월급(세후 등 기준 통일)
    fixed_expense: int                   # 고정지출(월)

@dataclass
class SelectedFinPrdt:
    kor_co_nm: str
    fin_prdt_nm: str
    max_limit: int
    intr_rate_type_nm: str # 단리, 복리
    fin_type: str           # 예금, 적금
    save_trm: int
    intr_rate: float
    etc_notes: Optional[str] = None

@dataclass
class SelectedStockPrdt:
    kor_co_nm: str
    rate: float
    risk: float
    risk_pct: float

@dataclass
class RebalanceAction:
    from_ticker: str
    to_ticker: str
    amount: float                          # 원화 금액(+/-)
    reason: str

@dataclass
class Portfolio:
    fin_prdt: SelectedFinPrdt | None
    stock_prdts: SelectedStockPrdt | None
    stock_allocation: float = 0.0           # 주식 비중(%)
    final_amount: float = 0.0             # 만기 예상 금액(원)

@dataclass
class RebalancePlan:
    before: Portfolio
    after: Portfolio
    actions: List[RebalanceAction]
    trigger: str                           # 트리거 설명 (밴드 이탈, 드로우다운 등)

@dataclass
class NewsSignal:
    ticker: str
    headline: str
    url: str
    published_at: str                      # ISO8601
    summary: str
    sentiment: Sentiment
    decision: NewsDecision | None = None   # 판단 결과

class GraphState(TypedDict): 
    user_id: int
    created_ts: str
    question:str
    answer:str
    
    route:str

    # 대화/프론트 단계
    ui_step: UIStep    

    # 입력/프로필
    goal: Goal
    investable_amount: int

    # 재무상품 선택
    selected_fin_prdt: SelectedFinPrdt
    selected_stock_prdt: SelectedStockPrdt

    # 포트폴리오
    indicators: Dict[int, Portfolio]        # 30/50/70% 시나리오
    user_selected_portfolio: Portfolio       # 사용자 선택 최종 포트폴리오

    # 리밸런싱
    rebalance_plan: RebalancePlan | None

    # 뉴스 파이프라인
    news_signals: List[NewsSignal]                # 수집/요약/감성 결과
    news_last_cursor: str | None                  # 크롤링 커서(옵션)

    # 감사·실행 추적
    events: List[Dict]                            # 작은 이벤트 로그(노드 진입/결정 등)

    # 지나간 개월수 (몇번째 리벨런싱인지)
    months_passed: int

    # 대화 메시지 (LangChain messages)
    messages: Annotated[List, add_messages]


# ===== 초기 상태 헬퍼 =====

def initial_state() -> GraphState:
    return GraphState(
        ui_step=UIStep.INPUT_GOAL,
        messages=[],
        events=[],
        selected_fin_prdt=None,
        news_signals=[],
    )

