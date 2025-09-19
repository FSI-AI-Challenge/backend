from dotenv import load_dotenv
import requests
import os
import pandas as pd

load_dotenv()
FSS_API_KEY = os.getenv("FSS_API_KEY")

JOIN_DENY_MAP = {"1":"제한없음", "2":"서민전용", "3":"일부제한"}

def make_url(product, auth, topFinGrpNo, pageNo):
    return f"https://finlife.fss.or.kr/finlifeapi/{product}ProductsSearch.json?auth={auth}&topFinGrpNo={topFinGrpNo}&pageNo={pageNo}"

def depositProductsSearch():
    
    topFinGrpNo = ["020000", "030200", "030300", "050000", "060000",]

    for no in topFinGrpNo:
        url = make_url("deposit", FSS_API_KEY, no, 1)
        response = requests.get(url)
        first = response.json()

        # 누적 리스트 준비
        base_rows = list(first['result'].get('baseList', []))
        opt_rows  = list(first['result'].get('optionList', []))

        max_page = int(first['result'].get('max_page_no', 1) or 1)

        if max_page > 1:
            for page in range(2, max_page + 1):
                url = make_url("deposit", FSS_API_KEY, no, page)
                response = requests.get(url)
                js = response.json()
                base_rows.extend(js['result'].get('baseList', []))
                opt_rows.extend(js['result'].get('optionList', []))
                print(f"Fetching page {page} for topFinGrpNo {no}")

        # DataFrame 변환
        base_df = pd.DataFrame(base_rows)
        opt_df  = pd.DataFrame(opt_rows)

        # 조인 키 지정
        key = ["dcls_month", "fin_co_no", "fin_prdt_cd"]
        for c in key:
            if c not in base_df: base_df[c] = None
            if c not in opt_df:  opt_df[c]  = None

        # baseList와 optionList 조인
        df = opt_df.merge(base_df, on=key, how="left")

        # 컬럼이 없으면 None으로 생성 → 항상 존재하도록 보장
        df = df.assign(join_deny=df.get('join_deny'))

        # 매핑(값이 있는 행만 안전하게 처리)
        mask = df['join_deny'].notna()
        df.loc[mask, 'join_deny'] = df.loc[mask, 'join_deny'].astype(str).map(JOIN_DENY_MAP).fillna(df.loc  [mask, 'join_deny'])

        # CSV 저장
        df.to_csv(f"../data/예금/{no}.csv", index=False, encoding="utf-8-sig")
        print(f"[{no}] 저장 완료: {len(df)} rows")

def savingProductsSearch():

    topFinGrpNo = ["020000", "030200", "030300", "050000", "060000",]

    for no in topFinGrpNo:
        url = make_url("saving", FSS_API_KEY, no, 1)
        response = requests.get(url)
        first = response.json()

        # 누적 리스트 준비
        base_rows = list(first['result'].get('baseList', []))
        opt_rows  = list(first['result'].get('optionList', []))

        max_page = int(first['result'].get('max_page_no', 1) or 1)

        if max_page > 1:
            for page in range(2, max_page + 1):
                url = make_url("saving", FSS_API_KEY, no, page)
                response = requests.get(url)
                js = response.json()
                base_rows.extend(js['result'].get('baseList', []))
                opt_rows.extend(js['result'].get('optionList', []))
                print(f"Fetching page {page} for topFinGrpNo {no}")

        # DataFrame 변환
        base_df = pd.DataFrame(base_rows)
        opt_df  = pd.DataFrame(opt_rows)

        # 조인 키 지정
        key = ["dcls_month", "fin_co_no", "fin_prdt_cd"]
        for c in key:
            if c not in base_df: base_df[c] = None
            if c not in opt_df:  opt_df[c]  = None

        # baseList와 optionList 조인
        df = opt_df.merge(base_df, on=key, how="left")

        # 컬럼이 없으면 None으로 생성 → 항상 존재하도록 보장
        df = df.assign(join_deny=df.get('join_deny'))

        # 매핑(값이 있는 행만 안전하게 처리)
        mask = df['join_deny'].notna()
        df.loc[mask, 'join_deny'] = df.loc[mask, 'join_deny'].astype(str).map(JOIN_DENY_MAP).fillna(df.loc[mask, 'join_deny'])

        # CSV 저장
        df.to_csv(f"../data/적금/{no}.csv", index=False, encoding="utf-8-sig")
        print(f"[{no}] 저장 완료: {len(df)} rows")

if __name__ == "__main__":
    depositProductsSearch()
    savingProductsSearch()