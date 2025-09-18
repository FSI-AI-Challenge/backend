from datetime import datetime, timedelta, timezone
from collections import defaultdict
from utils.state import *
import json
from langgraph.types import interrupt
from langchain_ollama import ChatOllama

llm = ChatOllama(model='gpt-oss:20b', streaming=True)

from utils.state import *
from utils.tools import *

from langchain_core.messages import SystemMessage, HumanMessage

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
    proposed = {
        "target_amount": state["goal"].target_amount,
        "target_months": state["goal"].target_months,
        "investable_amount": state["investable_amount"]
    }
    
    decision = interrupt({
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

    print(f"사용자 입력 검증 종료: {target_amount, target_months, investable_amount}")
    return GraphState(
        goal=Goal(
            target_amount=target_amount,
            target_months=target_months,
        ),
        investable_amount=investable_amount
    )

def _to_int(v): 
    return int(v) if v is not None and str(v).strip() != "" else 0
def _to_float(v):
    return float(v) if v is not None and str(v).strip() != "" else 0.0
def _to_str(v):
    return None if v is None else str(v)

def select_fin_prdt(state:GraphState):
    print("안전 자산 조회 시작")
    financial_products = pd.read_csv('./data/financial_products.csv')

    top_10_products = financial_products[
        (financial_products["save_trm"] <= state["goal"].target_months) &
        ((financial_products["max_limit"] > state["investable_amount"]) & (financial_products["label"] == "적금"))
    ].sort_values("intr_rate", ascending=False).head(10)

    top_10_products = top_10_products[["kor_co_nm", "fin_prdt_nm", "max_limit", "intr_rate_type_nm", "save_trm", "intr_rate", "etc_note", "label"]].to_dict(orient='records')

    system = SystemMessage(content=(
        "너는 예금/적금 상품 전문가야. "
        "아래 후보 중 '금리가 높고', 그리고 '사용자의 목표 기간(state[\"goal\"].target_months)과 가장 가까운 상품'을 1개 고른다. "
        "같은 금리라면 복리 우선, 그리고 etc_notes에 이상하거나 부적절한 내용(예: 없음, 불명확, 혜택 없음 등)이 없는 상품을 선택한다. "
        "우대조건(우대금리, 자동이체, 급여이체, 비대면/모바일, 주거래, 청년, 마이데이터, 세금우대 등)이 있으면 가점. "
        "최종 출력은 오직 JSON 한 개 객체만. 다른 텍스트 금지."
        "비어있는 값은 null로 채워고, 비어있는 값을 절대 임의로 채우지마 etc_notes에는 원본 그대로 넣어줘"
    ))
    user = HumanMessage(content=(
        "후보 리스트는 다음과 같아:\n"
        f"{json.dumps(top_10_products, ensure_ascii=False, indent=2)}\n\n"
        "아래 JSON 스키마에 정확히 맞춰 1개만 반환해줘.\n"
        "스키마: {\n"
        '  "kor_co_nm": str,\n'
        '  "fin_prdt_nm": str,\n'
        '  "max_limit": int,\n'
        '  "intr_rate_type_nm": "단리" | "복리",\n'
        '  "save_trm": int,\n'
        '  "intr_rate": float,\n'
        '  "etc_notes": str | null\n'
        '  "label": str | null\n'
        "}\n"
        "반드시 키 이름/타입을 정확히 지켜줘."
    ))

    resp = llm.invoke([system, user])
    picked_raw = extract_json(resp.content)

    selected = SelectedFinPrdt(
        kor_co_nm=_to_str(picked_raw.get("kor_co_nm", "")) or "",
        fin_prdt_nm=_to_str(picked_raw.get("fin_prdt_nm", "")) or "",
        max_limit=_to_int(picked_raw.get("max_limit", 0)),
        intr_rate_type_nm=_to_str(picked_raw.get("intr_rate_type_nm", "")) or "",
        save_trm=_to_int(picked_raw.get("save_trm", 0)),
        intr_rate=_to_float(picked_raw.get("intr_rate", 0.0)),
        etc_notes=_to_str(picked_raw.get("etc_notes", None)),
        fin_type=_to_str(picked_raw.get("label", None)),
    )
    print("안전 자산 조회 종료")
    return {**state, "selected_fin_prdt": selected}

def select_stock_products(state:GraphState):
    print("위험 자산 조회 시작")
    top_20_products = pd.read_csv('./data/krx_top100_rate_risk.csv')
    top_20_products = top_20_products.sort_values("rate", ascending=False).head(20).to_dict(orient='records')

    system = SystemMessage(content=(
        "너는 주식 상품 전문가야. "
        "아래 후보 중 '수익률이 높고', 그리고 '리스크가 낮은 상품'을 1개 고른다. "
        "최종 출력은 오직 JSON 배열만. 다른 텍스트 금지."
        "비어있는 값은 null로 채워고, 비어있는 값을 절대 임의로 채우지마"
    ))
    user = HumanMessage(content=(
        "후보 리스트는 다음과 같아:\n"
        f"{json.dumps(top_20_products, ensure_ascii=False, indent=2)}\n\n"
        "아래 JSON 스키마에 정확히 맞춰 1개만 반환해줘.\n"
        "스키마: {\n"
        '  "Name": str,\n'
        '  "rate": float,\n'
        '  "risk": float,\n'
        '  "risk_pct": float | null\n'
        '}\n'
        "반드시 키 이름/타입을 정확히 지켜줘."
    ))

    resp = llm.invoke([system, user])
    picked_raw = extract_json(resp.content)

    selected = SelectedStockPrdt(
        kor_co_nm=_to_str(picked_raw.get("Name", "")) or "",
        rate=_to_float(picked_raw.get("rate", 0.0)),
        risk=_to_float(picked_raw.get("risk", 0.0)),
        risk_pct=_to_float(picked_raw.get("risk_pct", 0.0))
    )
    print("위험 자산 조회 종료")
    return {**state, "selected_stock_prdt": selected}

def build_indicators(state: GraphState):
    print("포트폴리오 후보 생성 시작")
    investable_amount = state["investable_amount"]
    months = state["goal"].target_months - state.get("months_passed", 0)

    fin = state.get("selected_fin_prdt")
    stock = state.get("selected_stock_prdt")

    # 시나리오별 비율
    ratios = [0.3, 0.5, 0.7]
    indicators = {}

    for ratio in ratios:
        saving_amt = int(investable_amount * (1 - ratio))
        stock_amt = investable_amount - saving_amt

        # 적금 만기 수령액
        if fin:
            saving_final = calculate_savings_final_amount(
                monthly_deposit=saving_amt,
                intr_rate=fin.intr_rate,
                intr_rate_type=fin.intr_rate_type_nm,
                save_trm=months
            )
        else:
            saving_final = 0

        # 주식 만기 수령액
        if stock:
            stock_final = calculate_stock_final_amount(
                invest_amount=stock_amt,
                rate=stock.rate,
                months=months
            )
        else:
            stock_final = 0

        indicators[int(ratio * 100)] = Portfolio(
            fin_prdt=fin,
            stock_prdts=stock,
            stock_allocation=ratio,
            final_amount=saving_final + stock_final
        )
    print("포트폴리오 후보 생성 종료")
    return {**state, "indicators": indicators}

def build_portfolios(state: GraphState):
    print("포트폴리오 생성 시작")
    # HUMAN-in-the-loop로 비율을 입력받음
    decision = interrupt({
        "message": "적금/주식 투자 비율(0~100%)을 입력해주세요. \n(예: 30은 적금 70%, 주식 30%)",
        "proposed": {"stock_allocation_pct": 30},
        "fields": [
            {"name": "stock_allocation_pct", "type": "number", "label": "주식 비율(%)"},
        ],
        "buttons": ["submit"]
    })
    stock_allocation_pct = int(decision.get("stock_allocation_pct", 30))
    ratio = stock_allocation_pct / 100.0
    investable_amount = state["investable_amount"]
    months = state["goal"].target_months - state.get("months_passed", 0)
    fin = state.get("selected_fin_prdt")
    stock = state.get("selected_stock_prdt")
    saving_amt = int(investable_amount * (1 - ratio))
    stock_amt = investable_amount - saving_amt
    # 적금 만기 수령액
    if fin:
        saving_final = calculate_savings_final_amount(
            monthly_deposit=saving_amt,
            intr_rate=fin.intr_rate,
            intr_rate_type=fin.intr_rate_type_nm,
            save_trm=months
        )
    else:
        saving_final = 0
    # 주식 만기 수령액
    if stock:
        stock_final = calculate_stock_final_amount(
            invest_amount=stock_amt,
            rate=stock.rate,
            months=months
        )
    else:
        stock_final = 0
    portfolio = Portfolio(
        fin_prdt=fin,
        stock_prdts=stock,
        stock_allocation=ratio,
        final_amount=saving_final + stock_final
    )
    print("포트폴리오 생성 종료")
    return {**state, "user_selected_portfolio": portfolio}

def crawl_news(state:GraphState):
    return GraphState()

def summarize_news(state:GraphState):
    return GraphState()

def analyze_sentiment(state:GraphState):
    return GraphState()

def evaluate_rebalance(state:GraphState):
    return GraphState()

def is_goal_reached(state:GraphState):
    return "yes"

def is_rebalance_needed(state:GraphState):
    return "yes"