# pip install yfinance pandas numpy python-dateutil
import yfinance as yf
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import os, re, json
from typing import Dict, Any, Tuple, List

from langchain_naver_community.tool import NaverNewsSearch
from email.utils import parsedate_to_datetime
import hashlib
from dotenv import load_dotenv

load_dotenv()
NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")

_TAG_RE = re.compile(r"</?[^>]+>")
_REFINE_LIST = [
    "{q}",
    "{q} 실적 OR 공시",
    "{q} 뉴스 -블로그 -카페",
    "site:news.naver.com {q}",
]

def extract_json(text: str) -> Dict[str, Any]:
    # ```json ... ``` 블록 우선 추출 → 실패 시 중괄호 첫/끝 매칭
    codeblock = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if codeblock:
        text = codeblock.group(1)
    else:
        # 가장 바깥 {} 추정
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
    return json.loads(text)

def get_month_window(today=None):
    """야후 휴장일을 감안해 최근 거래일 기반 윈도우를 확보합니다."""
    if today is None:
        today = datetime.today()
    start = today - relativedelta(months=1) - timedelta(days=3)  # 버퍼 3일
    end = today + timedelta(days=1)                              # 오늘 포함
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def _krx_code_to_yahoo_ticker(row: pd.Series) -> Tuple[str, str]:
    """
    KRX 종목코드 → 야후 파이낸스 티커 변환
    - 코스피: .KS
    - 코스닥: .KQ
    """
    code = str(row["Code"]).zfill(6)
    market = str(row.get("Market", "KOSPI")).upper()
    if "KOSDAQ" in market:
        suffix = ".KQ"
    else:
        suffix = ".KS"
    return code, f"{code}{suffix}"

def fetch_top100_krx_by_marketcap() -> Dict[str, str]:
    """
    KRX 상장사 전체에서 시가총액 상위 100개 종목을 가져와
    {종목명: 야후티커} dict로 반환합니다.
    - FinanceDataReader의 상장사 목록: Code, Name, Market, MarketCap 등
    """
    # 상장사 리스트
    listed = fdr.StockListing("KRX")

    # 시총 컬럼명 방어적 처리 (FinanceDataReader 버전에 따라 다를 수 있음)
    # 보통 'Marcap' 또는 'MarketCap' 형태가 제공됩니다.
    marcap_col_candidates = [c for c in listed.columns if c.lower() in ("marcap", "marketcap", "market_cap", "mktcap")]
    if not marcap_col_candidates:
        raise ValueError("시가총액 컬럼을 찾을 수 없습니다. FinanceDataReader 버전을 확인해주세요.")
    marcap_col = marcap_col_candidates[0]

    # 결측 제거 후 시총 순 정렬
    df_top = (
        listed.dropna(subset=[marcap_col, "Code", "Name"])
              .sort_values(by=marcap_col, ascending=False)
              .head(100)
              .reset_index(drop=True)
    )

    # 야후 티커 생성
    df_top[["KRXCode", "YahooTicker"]] = df_top.apply(_krx_code_to_yahoo_ticker, axis=1, result_type="expand")

    # 이름→티커 dict
    name_to_ticker = dict(zip(df_top["Name"], df_top["YahooTicker"]))
    return name_to_ticker

def compute_rate_and_risk_from_tickers(name_to_ticker: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    입력: { '삼성전자': '005930.KS', ... }
    출력: {
      "삼성전자": {"rate": float, "risk": float, "risk_pct": float},
      ...
    }
    """
    start, end = get_month_window()
    result: Dict[str, Dict[str, float]] = {}

    if not name_to_ticker:
        return result

    tickers = list(name_to_ticker.values())
    # yfinance에서 대량 종목 다운로드
    df = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        threads=True,
        group_by="ticker",
        progress=False
    )

    # 단일/다중 티커 모두 처리 가능한 helper
    def get_series(tk, col):
        if isinstance(df.columns, pd.MultiIndex):
            return df[(tk, col)].dropna()
        else:
            # 단일 티커인 경우
            return df[col].dropna()

    for name, tk in name_to_ticker.items():
        try:
            close = get_series(tk, "Close")
            high  = get_series(tk, "High")
            low   = get_series(tk, "Low")

            if close.empty or high.empty or low.empty:
                result[name] = {"error": "가격 데이터 부족"}
                continue

            first_close = close.iloc[0]
            last_close  = close.iloc[-1]

            rate = float((last_close / first_close) - 1)

            month_high = float(high.max())
            month_low  = float(low.min())
            risk_abs = month_high - month_low
            risk_pct = float(risk_abs / first_close)

            result[name] = {
                "rate": round(rate, 6),        # 예: 0.0345 → 3.45%
                "risk": round(risk_abs, 4),    # 절대가격(원)
                "risk_pct": round(risk_pct, 6) # 예: 0.0812 → 8.12%
            }
        except Exception as e:
            result[name] = {"error": f"계산 실패: {e}"}

    return result

def calculate_savings_final_amount(monthly_deposit: float, intr_rate: float, intr_rate_type: str, save_trm: int) -> float:
    """
    적금(월별 납입) 만기 시 수령액 계산
    - monthly_deposit: 월별 납입액(원)
    - intr_rate: 연이율(%)
    - intr_rate_type: "단리" or "복리"
    - save_trm: 적립 기간(개월)
    """
    n = save_trm
    r = intr_rate / 100 / 12  # 월이율

    if intr_rate_type == "단리":
        # 단리: 각 월별 납입금에 대해 남은 기간만큼만 이자 적용
        total = 0
        for i in range(n):
            months = n - i
            total += monthly_deposit * (1 + r * months)
        return round(total, 2)
    elif intr_rate_type == "복리":
        # 복리: 각 월별 납입금에 대해 남은 기간만큼 복리 적용
        total = 0
        for i in range(n):
            months = n - i
            total += monthly_deposit * ((1 + r) ** months)
        return round(total, 2)
    else:
        raise ValueError("intr_rate_type은 '단리' 또는 '복리'여야 합니다.")

def calculate_stock_final_amount(invest_amount: float, rate: float, months: int) -> float:
    """
    단순 주식 수익 계산: 투자금액 * 수익률 * 개월수
    - invest_amount: 투자 금액(원)
    - rate: 월별 수익률(예: 0.03 → 3%)
    - months: 투자 개월 수
    """
    return round(invest_amount * (1 + rate) * months, 2)

# ===== 사용 예시 =====
if __name__ == "__main__":
    # 1) KRX 시가총액 TOP100 자동 수집 (코스피/코스닥 포함)
    name_to_ticker_top100 = fetch_top100_krx_by_marketcap()

    # 2) 한 달 수익률 및 리스크 계산
    out = compute_rate_and_risk_from_tickers(name_to_ticker_top100)

    # 3) 예시 출력: 수익률 상위 10개
    df_out = (
        pd.DataFrame.from_dict(out, orient="index")
          .reset_index().rename(columns={"index": "Name"})
    )

    # 결과를 로컬 CSV로 저장
    df_out.to_csv("krx_top100_rate_risk.csv", index=False, encoding="utf-8-sig")

    if "rate" in df_out.columns:
        df_top10 = df_out.sort_values("rate", ascending=False).head(10)
        print(df_top10)
    else:
        print(df_out.head())
        

def get_naver_tool() -> NaverNewsSearch:
    if not NAVER_ID or not NAVER_SECRET:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수 없음")
    return NaverNewsSearch(naver_client_id=NAVER_ID, naver_client_secret=NAVER_SECRET)

# ---- 뉴스 전처리 & 검색 헬퍼 ----

def _strip_tags(s: str) -> str:
    return _TAG_RE.sub("", s or "").strip()

def _to_iso8601(s: str) -> str:
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

def dedup_and_clean(items: List[Dict[str, Any]]) -> Tuple[List[Dict], int]:
    seen, out, dropped = set(), [], 0
    for it in items:
        title = _strip_tags(it.get("title") or it.get("headline") or "")
        link  = (it.get("originallink") or it.get("link") or it.get("url") or "").strip()
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


def refined_query(q: str, attempt: int) -> str:
    pat = _REFINE_LIST[min(attempt, len(_REFINE_LIST)-1)]
    return pat.replace("{q}", q)

def naver_search_once(naver_tool, query: str, start: int, display: int) -> List[Dict[str, Any]]:
    try:
        raw = naver_tool.run(query, display=display, start=start, sort="date")
    except TypeError:
        raw = naver_tool.run(query, display=display)
    payload = _normalize_naver_payload(raw)
    return payload.get("items", [])