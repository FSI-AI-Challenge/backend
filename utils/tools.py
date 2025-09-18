# pip install yfinance pandas numpy python-dateutil
import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import re, json
from typing import Dict, Any, Tuple


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