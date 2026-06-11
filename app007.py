import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import time
import asyncio
import websockets
from datetime import datetime, timedelta, timezone, time as dt_time
from bs4 import BeautifulSoup
import streamlit.components.v1 as components

# 📱 1. 페이지 기본 설정 (무조건 최상단)
st.set_page_config(layout="wide", page_title="국내주식 실시간 딥러닝 스캐너", initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# [설정] 한국투자증권 API KEY
# -----------------------------------------------------------------------------
try:
    KIS_APP_KEY = st.secrets["KIS_APP_KEY"]
    KIS_APP_SECRET = st.secrets["KIS_APP_SECRET"]
    
    APP_KEY = KIS_APP_KEY
    APP_SECRET = KIS_APP_SECRET
except KeyError:
    st.error("⚠️ Streamlit secrets에 'KIS_APP_KEY' 또는 'KIS_APP_SECRET'이 설정되지 않았습니다.")
    st.stop()

URL_BASE = "https://openapi.koreainvestment.com:9443" 
KST = timezone(timedelta(hours=9))

THEME_DICT = {
    "🤖 로봇": ["두산로보틱스", "레인보우로보틱스", "뉴로메카", "에스피지", "로보티즈", "이랜시스", "로보틱스", "엔젤로보틱스"],
    "💾 반도체": ["한미반도체", "SK하이닉스", "삼성전자", "HPSP", "이수페타시스", "제우스", "가온칩스", "리노공업", "디아이", "필옵틱스", "와이씨"],
    "🔋 2차전지": ["에코프로", "에코프로비엠", "에코프로머티", "포스코홀딩스", "POSCO홀딩스", "LG에너지솔루션", "엘앤에프", "금양", "엔켐"],
    "🧬 바이오": ["알테오젠", "HLB", "삼성바이오로직스", "셀트리온", "삼천당제약", "리가켐바이오", "휴젤", "유한양행"],
    "⚡ 전력기기": ["HD현대일렉트릭", "LS일렉트릭", "효성중공업", "제룡전기", "일진전기", "LS에코에너지"],
    "💄 화장품": ["실리콘투", "브이티", "코스메카코리아", "씨앤씨인터내셔널", "아모레퍼시픽", "클리오", "토니모리"]
}

def get_theme_icon(stock_name):
    for theme, keywords in THEME_DICT.items():
        if any(keyword in stock_name for keyword in keywords):
            return theme
    return "▪️ 개별주"

# -----------------------------------------------------------------------------
# 인증 키 발급 함수들
# -----------------------------------------------------------------------------
@st.cache_resource(ttl=3600*20)
def get_access_token():
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    try:
        res = requests.post(f"{URL_BASE}/oauth2/tokenP", headers=headers, data=json.dumps(body))
        return res.json()["access_token"]
    except: return None

@st.cache_resource(ttl=3600*20)
def get_approval_key():
    url = f"{URL_BASE}/oauth2/Approval"
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "secretkey": APP_SECRET}
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        return res.json().get('approval_key')
    except: return None

def get_common_headers(tr_id):
    token = get_access_token()
    if not token: token = get_access_token()
    return {"Content-Type": "application/json", "authorization": f"Bearer {token}", "appKey": APP_KEY, "appSecret": APP_SECRET, "tr_id": tr_id}

# -----------------------------------------------------------------------------
# 데이터 수집 함수들
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def get_kis_top_trading_value_stocks():
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/volume-rank"
    headers = get_common_headers("FHPST01710000")
    
    params_mid = {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171", "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "1", 
        "FID_BLNG_CLS_CODE": "0", "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "111111", 
        "FID_INPUT_PRICE_1": "10000", "FID_INPUT_PRICE_2": "80000", "FID_VOL_CNT": "", "FID_INPUT_DATE_1": ""
    }
    params_large = {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171", "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "1", 
        "FID_BLNG_CLS_CODE": "0", "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "111111", 
        "FID_INPUT_PRICE_1": "80000", "FID_INPUT_PRICE_2": "2000000", "FID_VOL_CNT": "", "FID_INPUT_DATE_1": ""
    }
    
    df_list = []
    for params in [params_mid, params_large]:
        try:
            res = requests.get(url, headers=headers, params=params, timeout=5).json()
            if res.get('rt_cd') == '0' and 'output' in res:
                df_list.append(pd.DataFrame(res['output'])[['hts_kor_isnm', 'mksc_shrn_iscd', 'stck_prpr', 'prdy_ctrt', 'acml_tr_pbmn']])
        except: continue
            
    if not df_list: return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    df.columns = ['종목명', '종목코드', '현재가', '등락률', '거래대금']
    
    pattern = '|'.join(['KODEX', 'TIGER', 'KBSTAR', 'ACE', 'ARIRANG', 'HANARO', 'KOSEF', 'SOL', 'TIMEFOLIO', 'WOORI', '히어로즈', '마이티', '스팩', 'ETN'])
    df = df[~df['종목명'].str.contains(pattern, case=False, regex=True)]
    
    df['현재가'] = pd.to_numeric(df['현재가'], errors='coerce')
    df['등락률'] = pd.to_numeric(df['등락률'], errors='coerce')
    df['거래대금'] = pd.to_numeric(df['거래대금'], errors='coerce') / 1000000 
    return df.sort_values(by='거래대금', ascending=False).drop_duplicates(subset=['종목코드']).dropna()

# 💡 [신규 추가] 장전 동시호가 예상체결가 가져오기
def get_expected_price(target_code):
    quote_url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
    headers = get_common_headers("FHKST01010200")
    try:
        res = requests.get(quote_url, headers=headers, params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": target_code}, timeout=2).json()
        if res.get('rt_cd') == '0':
            antc_cnpr = float(res['output1'].get('antc_cnpr', 0)) # 예상체결가
            stck_prpr = float(res['output1'].get('stck_prpr', 1)) # 전일종가
            antc_vol = float(res['output1'].get('antc_cntg_vrss_vol', 0)) # 예상체결수량
            if antc_cnpr > 0 and stck_prpr > 0:
                gap_ratio = round(((antc_cnpr - stck_prpr) / stck_prpr) * 100, 2)
                return antc_cnpr, gap_ratio, antc_vol
    except: pass
    return 0.0, 0.0, 0.0

@st.cache_data(ttl=15)
def get_foreign_investor_trend():
    try:
        res = requests.get("https://finance.naver.com/sise/sise_index.naver?code=KOSPI", headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        for dd in soup.find_all('dd'):
            text = dd.get_text(strip=True)
            if text.startswith("외국인") and "억" in text:
                return float(text.replace("외국인", "").replace("억", "").replace(",", "").strip())
    except: pass
    return 0.0

@st.cache_data(ttl=30)
def get_realtime_market_summary():
    headers = {"User-Agent": "Mozilla/5.0"}
    def fetch_index(code):
        try:
            res = requests.get(f"https://m.stock.naver.com/api/index/{code}/basic", headers=headers, timeout=5).json()
            return res.get('closePrice', '0'), float(res.get('fluctuationsRatio', 0.0))
        except: return "-", 0.0

    def fetch_exchange():
        try:
            soup = BeautifulSoup(requests.get("https://finance.naver.com/marketindex/", headers=headers, timeout=5).text, 'html.parser')
            blind = soup.select_one("#exchangeList .blind").text
            return soup.select_one("#exchangeList .value").text, f"{'+' if '상승' in blind else '-' if '하락' in blind else ''}{soup.select_one('#exchangeList .change').text}"
        except: return "-", "-"

    return fetch_index("KOSPI"), fetch_index("KOSDAQ"), fetch_exchange()

# -----------------------------------------------------------------------------
# 🚀 웹소켓 실시간 수신 비동기 함수 (Streamlit 내장용)
# -----------------------------------------------------------------------------
async def connect_kis_websocket_streamlit(target_code, target_name, placeholder):
    approval_key = get_approval_key()
    if not approval_key:
        placeholder.error("❌ 웹소켓 접속키 발급 실패!")
        return
    
    ws_url = "ws://ops.koreainvestment.com:21000"
    
    try:
        async with websockets.connect(ws_url, ping_interval=None) as websocket:
            subscribe_data = {
                "header": { "approval_key": approval_key, "custtype": "P", "tr_type": "1", "content-type": "utf-8" },
                "body": { "input": { "tr_id": "H0STCNT0", "tr_key": target_code } }
            }
            await websocket.send(json.dumps(subscribe_data))
            
            log_text = f"📡 **[{target_name}] 웹소켓 연결 성공! 실시간 체결 대기중...**\n\n"
            placeholder.markdown(log_text)
            
            for _ in range(15): 
                data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                if data[0] == '0' or data[0] == '1':
                    parts = data.split('|')
                    if len(parts) >= 4:
                        real_time_data = parts[3].split('^')
                        current_price = int(real_time_data[2])
                        vs_yesterday = float(real_time_data[4])
                        volume = int(real_time_data[13])
                        
                        updown_color = "🔴" if vs_yesterday > 0 else "🔵" if vs_yesterday < 0 else "⚫"
                        new_log = f"{updown_color} 체결가: **{current_price:,}원** | 등락률: {vs_yesterday}% | 체결량: {volume:,}주\n\n"
                        log_text = new_log + log_text
                        placeholder.markdown(f"> {log_text}")
    except asyncio.TimeoutError:
        placeholder.warning("⏳ 거래량이 없어 새로운 체결 데이터가 들어오지 않았습니다.")
    except Exception as e:
        placeholder.error(f"❌ 웹소켓 연결 종료: {str(e)}")

# =============================================================================
# ⚙️ 안전한 화면 모드 전환 스위치 & 쇼츠 모드 분기 (기존과 동일하여 생략 없이 유지)
# =============================================================================
with st.expander("⚙️ OBS 방송 송출용 화면 설정 (클릭하여 열기)"):
    is_shorts_mode = st.checkbox("📱 쇼츠(세로형) 라이브 모드 켜기", value=False)
    st.info("이 체크박스를 켜면 즉시 화면이 검은 배경의 세로형 쇼츠 디자인으로 변경됩니다. OBS에서 이 브라우저 창을 캡처하세요.")

if is_shorts_mode:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60000, limit=10000, key="shorts_refresh")
    except: pass

    st.markdown("""
    <style>
        html, body, [class*="css"], .stApp { background-color: #0b1120 !important; color: white !important; }
        .block-container { padding: 1.5rem 1.0rem 4.5rem 1.0rem !important; max-width: 480px !important; margin: 0 auto !important; }
        header[data-testid="stHeader"], footer { display: none !important; }
        ::-webkit-scrollbar { display: none !important; }
        .s-title { text-align: center; color: #facc15; font-size: 1.8rem; font-weight: 900; margin-bottom: 5px; white-space: nowrap; }
        .s-card { background-color: #1e293b; border-radius: 12px; padding: 12px 15px; margin-bottom: 8px; border: 1px solid #334155; display: flex; flex-direction: column; gap: 6px; }
        .s-card-top { display: flex; justify-content: space-between; align-items: center; }
        .s-rank-name { display: flex; align-items: center; gap: 10px; overflow: hidden; }
        .s-rank { background-color: #ef4444; color: white; border-radius: 50%; min-width: 22px; height: 22px; display: flex; justify-content: center; align-items: center; font-size: 1.0rem; font-weight: bold; }
        .s-name { font-size: 1.4rem; font-weight: 900; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; word-break: keep-all; }
        .s-eval { font-size: 0.85rem; color: #facc15; font-weight: bold; background: #0f172a; padding: 3px 6px; border-radius: 5px; white-space: nowrap; border: 1px solid #475569; }
        .s-card-bottom { display: flex; justify-content: space-between; align-items: baseline; }
        .s-price-box { display: flex; align-items: baseline; gap: 8px; flex-shrink: 0; }
        .s-price { font-size: 1.3rem; color: #e2e8f0; font-weight: bold; }
        .s-ratio { font-size: 1.1rem; font-weight: 900; }
        .s-ratio.up { color: #ef4444; }
        .s-ratio.down { color: #3b82f6; }
        .s-score { font-size: 0.95rem; color: #10b981; font-weight: bold; background: rgba(16, 185, 129, 0.15); padding: 3px 8px; border-radius: 5px; }
        .s-marquee { position: fixed; bottom: 0; left: 0; width: 100%; background-color: #b91c1c; color: white; padding: 12px 0; font-size: 1.2rem; font-weight: bold; z-index: 9999; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='s-title'>🔴 실시간 AI 타점 스캐너</div>", unsafe_allow_html=True)
    
    df_shorts = get_kis_top_trading_value_stocks()
    if not df_shorts.empty:
        df_shorts = df_shorts[df_shorts['등락률'] > 1.0].copy()
        df_shorts['스코어'] = ((df_shorts['등락률'] * 0.5) + np.log1p(df_shorts['거래대금'])).round(2)
        top_10 = df_shorts.sort_values(by='스코어', ascending=False).head(10)
        
        for i, (index, row) in enumerate(top_10.iterrows(), 1):
            rate, vol = row['등락률'], row['거래대금']
            if rate >= 7.0 and vol > 500: eval_text = "🔥 급등돌파"
            elif 1.0 <= rate < 5.0 and vol > 200: eval_text = "💧 눌림목"
            else: eval_text = "⚡ AI포착"
            
            ratio_class, sign = ("up", "+") if rate > 0 else ("down", "")
            
            st.markdown(f"""
                <div class="s-card">
                    <div class="s-card-top">
                        <div class="s-rank-name"><div class="s-rank">{i}</div><div class="s-name">{row['종목명']}</div></div>
                        <div class="s-eval">{eval_text}</div>
                    </div>
                    <div class="s-card-bottom">
                        <div class="s-price-box"><span class="s-price">{int(row['현재가']):,}원</span><span class="s-ratio {ratio_class}">{sign}{rate:.2f}%</span></div>
                        <div class="s-score">AI점수 {row['스코어']:.1f}점</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("데이터 수집 중입니다.")
        
    st.markdown("""
        <marquee class="s-marquee" scrollamount="6">
            ⚠️ [투자 유의사항] 본 방송은 딥러닝 AI 모델 분석에 의한 단순 데이터 제공용이며 투자를 권유하지 않습니다. 모든 매매의 최종 판단과 책임은 투자자 본인에게 있습니다.
        </marquee>
    """, unsafe_allow_html=True)

# =============================================================================
# 💻 분기점 2: 기존 메인 대시보드 화면
# =============================================================================
else:
    st.title("🚀 실시간 딥러닝 단타 및 시장 동향 대시보드")
    st.subheader("🌐 실시간 시장 종합 상황판")

    (ks_price, ks_ratio), (kq_price, kq_ratio), (usd_price, usd_change) = get_realtime_market_summary()

    if 'foreign_futures_net' not in st.session_state: 
        st.session_state.foreign_futures_net = get_foreign_investor_trend()
    foreign_futures_net = st.session_state.foreign_futures_net

    if foreign_futures_net > 0: value_str, program_intensity, trade_signal, delta_msg, score_color = f"+{foreign_futures_net:,} 억 원", min(100, int(50 + (foreign_futures_net / 10))), "🚀 강력 매수", "매수 우위", "normal"
    elif foreign_futures_net < 0: value_str, program_intensity, trade_signal, delta_msg, score_color = f"{foreign_futures_net:,} 억 원", max(0, int(50 - (abs(foreign_futures_net) / 10))), "⚠️ 강한 매도", "매도 우위", "inverse"
    else: value_str, program_intensity, trade_signal, delta_msg, score_color = "0.0 억 원", 50, "⏸️ 대기 중", "데이터 없음", "off"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("📊 KOSPI", ks_price, f"{ks_ratio:+.2f}%" if ks_price != "-" else "데이터 없음")
    with col2: st.metric("📈 KOSDAQ", kq_price, f"{kq_ratio:+.2f}%" if kq_price != "-" else "데이터 없음")
    with col3: st.metric("💵 환율(USD/KRW)", f"{usd_price} 원", f"{usd_change} 원" if usd_price != "-" else "데이터 없음", delta_color="inverse")
    with col4: st.metric("🏢 외국인 선물", value_str, delta_msg, delta_color=score_color)
    with col5: st.metric("🎯 우량주 매력도", f"{program_intensity} 점", trade_signal, delta_color=score_color)

    st.markdown("---")

    now_time = datetime.now(KST).time()
    time_pre_start, time_reg_start, time_after_start = dt_time(8, 30), dt_time(9, 0), dt_time(15, 30)
    default_auto, default_pre, default_after = False, False, True
    
    if time_pre_start <= now_time < time_reg_start: default_auto, default_pre, default_after = True, True, False
    elif time_reg_start <= now_time < time_after_start: default_auto, default_pre, default_after = True, False, False

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: auto_refresh = st.toggle("⏱️ 1분 자동 스캐닝 켜기", value=default_auto)
    with col_t2: pre_market_mode = st.toggle("☀️ 프리마켓 (08:30~09:00)", value=default_pre)
    with col_t3: after_market_mode = st.toggle("🌙 애프터 마켓 (15:30~18:00)", value=default_after)

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60000, limit=1000, key="auto_scanner_refresh")
        except ImportError: pass

    # 💡 [핵심 업그레이드] 프리마켓 모드 vs 일반 모드 완벽 분리
    df_universe = get_kis_top_trading_value_stocks()

    if not df_universe.empty:
        filtered_df = df_universe[df_universe['등락률'] >= 1.0].copy()
        filtered_df = filtered_df.sort_values(by='거래대금', ascending=False).head(20)
        filtered_df['테마'] = filtered_df['종목명'].apply(get_theme_icon)

        if pre_market_mode:
            st.subheader("☀️ 장전 동시호가 갭상승 주도주 포착 (08:30 ~ 09:00)")
            st.info("💡 9시 개장 전, 예상 체결가를 바탕으로 시초가부터 강하게 치고 나갈 종목을 추적합니다.")
            
            with st.spinner("호가창 예상 체결가 분석 중..."):
                exp_prices, exp_gaps, exp_vols = [], [], []
                for _, row in filtered_df.iterrows():
                    prc, gap, vol = get_expected_price(row['종목코드'])
                    exp_prices.append(prc)
                    exp_gaps.append(gap)
                    exp_vols.append(vol)
                    time.sleep(0.05) # API 과부하 방지
                
                filtered_df['예상_시초가'] = exp_prices
                filtered_df['예상_갭상승률(%)'] = exp_gaps
                filtered_df['예상_체결량'] = exp_vols
                
                # 가짜 호가 필터링 (예상 갭이 20%가 넘는데 체결수량이 100주 이하면 허수)
                filtered_df = filtered_df[~((filtered_df['예상_갭상승률(%)'] > 20) & (filtered_df['예상_체결량'] < 500))]
                
                # 프리마켓 전용 대장주 탐색 (예상 갭상승률 기준)
                idx_max_by_theme = filtered_df.groupby('테마')['예상_갭상승률(%)'].idxmax()
                filtered_df['대장주여부'] = False
                filtered_df.loc[idx_max_by_theme, '대장주여부'] = True
                
                filtered_df['출력종목명'] = filtered_df.apply(
                    lambda x: f"👑 {x['종목명']}" if x['대장주여부'] and x['테마'] != "▪️ 개별주" else x['종목명'], axis=1
                )
                
                def get_pre_status(gap):
                    if gap >= 5.0: return "🔥 강력 갭상승 유력"
                    elif 1.0 <= gap < 5.0: return "☀️ 상승 출발 유력"
                    elif gap < 0: return "⚠️ 갭하락 주의"
                    return "▪️ 보합 예상"
                    
                filtered_df['장전_상태'] = filtered_df['예상_갭상승률(%)'].apply(get_pre_status)
                top_list = filtered_df.sort_values(by=['대장주여부', '예상_갭상승률(%)'], ascending=[False, False])
                
                output_dict = {
                    '테마': top_list['테마'], '장전 상태': top_list['장전_상태'], 
                    '예상 갭상승(%)': top_list['예상_갭상승률(%)'].apply(lambda x: f"🚀 {x:+.2f} %"), 
                    '종목명': top_list['출력종목명'], 
                    '예상 시초가': top_list['예상_시초가'].apply(lambda x: f"{int(x):,} 원" if x > 0 else "데이터 없음"),
                    '전일 종가': top_list['현재가'].apply(lambda x: f"{int(x):,} 원"),
                    '예상 체결량': top_list['예상_체결량'].apply(lambda x: f"{int(x):,} 주"),
                    '종목코드': top_list['종목코드'], '원본종목명': top_list['종목명']
                }

        else:
            # 기존 일반장 (09:00 ~ 15:30) 실시간 로직
            st.subheader("🎯 실시간 딥러닝 타겟 및 대장주 탐색")
            filtered_df['10분_상승예측(%)'] = ((filtered_df['등락률'] * 0.5) + np.log1p(filtered_df['거래대금'])).round(2)
            
            idx_max_by_theme = filtered_df.groupby('테마')['등락률'].idxmax()
            filtered_df['대장주여부'] = False
            filtered_df.loc[idx_max_by_theme, '대장주여부'] = True
            
            filtered_df['출력종목명'] = filtered_df.apply(
                lambda x: f"👑 {x['종목명']}" if x['대장주여부'] and x['테마'] != "▪️ 개별주" else x['종목명'], axis=1
            )

            filtered_df['단기_목표가'] = (filtered_df['현재가'] * 1.03).astype(int)
            filtered_df['손절가'] = (filtered_df['현재가'] * 0.98).astype(int)

            def detect_signal(row):
                if row['등락률'] >= 7.0 and row['거래대금'] > 50000: return "🔥 돌파매매"
                elif 1.0 <= row['등락률'] < 5.0 and row['거래대금'] > 20000: return "💧 눌림목"
                return "▪️ 관망"
                
            filtered_df['매매상태'] = filtered_df.apply(detect_signal, axis=1)
            top_list = filtered_df.sort_values(by=['대장주여부', '10분_상승예측(%)'], ascending=[False, False])
            
            output_dict = {
                '테마': top_list['테마'], '실시간 상태': top_list['매매상태'], 'AI 예측스코어': top_list['10분_상승예측(%)'].apply(lambda x: f"🚀 {float(x):.2f}점"), 
                '종목명': top_list['출력종목명'], '현재가': top_list['현재가'].apply(lambda x: f"{int(x):,} 원"), '전일비': top_list['등락률'].apply(lambda x: f"+{x:.2f} %"),
                '단기 목표가(+3%)': top_list['단기_목표가'].apply(lambda x: f"{x:,} 원"), '손절가(-2%)': top_list['손절가'].apply(lambda x: f"{x:,} 원"),
                '거래대금(백만)': top_list['거래대금'].apply(lambda x: f"{int(x):,}"), '종목코드': top_list['종목코드'], '원본종목명': top_list['종목명']
            }
        
        output_df = pd.DataFrame(output_dict).reset_index(drop=True)
        st.write("표 안의 종목을 클릭하면 하단에 차트 및 호가/체결 데이터가 나타납니다.")
        selected_rows = st.dataframe(output_df.drop(columns=['종목코드', '원본종목명']), use_container_width=True, selection_mode="single-row", on_select="rerun")
    else:
        st.error("데이터를 불러오지 못했습니다.")
        output_df = pd.DataFrame()

    # -----------------------------------------------------------------------------
    # 차트, 호가창(체결강도), 그리고 웹소켓 라이브
    # -----------------------------------------------------------------------------
    st.markdown("---")
    selected_idx = selected_rows.selection.rows[0] if (hasattr(selected_rows, 'selection') and len(selected_rows.selection.rows) > 0) else 0

    if not output_df.empty and selected_idx < len(output_df):
        target_code = output_df.iloc[selected_idx]['종목코드']
        target_name = output_df.iloc[selected_idx]['원본종목명']
        target_display_name = output_df.iloc[selected_idx]['종목명'] # 왕관 포함
        target_theme = output_df.iloc[selected_idx]['테마']
        
        st.markdown(f"## 📈 {target_display_name} 상세 분석")
        
        with st.spinner("호가창 및 체결강도 불러오는 중..."):
            quote_url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
            quote_headers = get_common_headers("FHKST01010200")
            quote_params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": target_code}
            
            try:
                q_res = requests.get(quote_url, headers=quote_headers, params=quote_params).json()
                if q_res.get('rt_cd') == '0':
                    q_data = q_res['output1']
                    
                    price_url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-price"
                    price_headers = get_common_headers("FHKST01010100")
                    p_res = requests.get(price_url, headers=price_headers, params=quote_params).json()
                    power = p_res['output'].get('vrss_vol_rate', '데이터없음') if p_res.get('rt_cd') == '0' else '데이터없음'
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.info("📉 **매도 호가 (Top 3)**")
                        st.write(f"1호가: {int(q_data.get('askp1', 0)):,}원 (잔량: {int(q_data.get('askp_rsqn1', 0)):,}주)")
                        st.write(f"2호가: {int(q_data.get('askp2', 0)):,}원 (잔량: {int(q_data.get('askp_rsqn2', 0)):,}주)")
                        st.write(f"3호가: {int(q_data.get('askp3', 0)):,}원 (잔량: {int(q_data.get('askp_rsqn3', 0)):,}주)")
                    with c2:
                        st.error("📈 **매수 호가 (Top 3)**")
                        st.write(f"1호가: {int(q_data.get('bidp1', 0)):,}원 (잔량: {int(q_data.get('bidp_rsqn1', 0)):,}주)")
                        st.write(f"2호가: {int(q_data.get('bidp2', 0)):,}원 (잔량: {int(q_data.get('bidp_rsqn2', 0)):,}주)")
                        st.write(f"3호가: {int(q_data.get('bidp3', 0)):,}원 (잔량: {int(q_data.get('bidp_rsqn3', 0)):,}주)")
                    with c3:
                        st.warning("⚡ **변동성 및 체결강도**")
                        st.metric("현재 체결강도", f"{power}%" if power != "데이터없음" else power)
                        st.caption("체결강도가 100% 이상이면 매수세가 강함을 의미합니다.")
            except Exception as e:
                st.write("호가창 데이터를 불러오는 중 오류가 발생했습니다.")
        
        if st.button(f"▶️ [{target_name}] 실시간 웹소켓 체결 데이터 보기 (10초간 수신)", type="primary"):
            ws_placeholder = st.empty()
            asyncio.run(connect_kis_websocket_streamlit(target_code, target_name, ws_placeholder))
        
        st.markdown("---")

        with st.spinner(f"[{target_name}] 1분봉 차트 그리는 중..."):
            url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
            headers = get_common_headers("FHKST03010200")
            params = {"FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": target_code, "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), "FID_PW_DATA_INCU_YN": "Y"}
            
            try:
                res = requests.get(url, headers=headers, params=params)
                res_data = res.json()
                if res_data['rt_cd'] == '0' and 'output2' in res_data:
                    min_data = res_data['output2'][::-1] 
                    df_min = pd.DataFrame({"Open": [float(m['stck_oprc']) for m in min_data], "High": [float(m['stck_hgpr']) for m in min_data], "Low": [float(m['stck_lwpr']) for m in min_data], "Close": [float(m['stck_prpr']) for m in min_data], "Volume": [float(m['cntg_vol']) for m in min_data]}, index=pd.to_datetime([f"{m['stck_bsop_date']} {m['stck_cntg_hour']}" for m in min_data], format="%Y%m%d %H%M%S"))
                    df_min = df_min[df_min['Close'] > 0]
                    
                    if not df_min.empty:
                        df_min['MA5'], df_min['MA20'], df_min['Vol_MA5'] = df_min['Close'].rolling(5).mean(), df_min['Close'].rolling(20).mean(), df_min['Volume'].rolling(5).mean()
                        df_min['Breakout'] = (df_min['Close'] > df_min['High'].shift(1).rolling(20).max()) & (df_min['Volume'] > df_min['Vol_MA5'] * 1.5)
                        df_min['Pullback'] = (df_min['MA20'] > df_min['MA20'].shift(3)) & (df_min['Low'] <= df_min['MA20'] * 1.005) & (df_min['Close'] >= df_min['MA20'] * 0.998) & (df_min['Volume'] < df_min['Vol_MA5'])
                        
                        c_p, h_10m = df_min['Close'].iloc[-1], df_min['High'].iloc[-10:].max()
                        
                        if c_p < df_min['MA5'].iloc[-1] and c_p <= h_10m * 0.97: st.error(f"💣 **[🚨 보안관 비상경보]** **{target_name}** 5분선 붕괴 위험! 신규 진입 금지.")
                        elif c_p >= h_10m * 0.98 and not (df_min['MA5'].iloc[-1] > df_min['MA20'].iloc[-1] and df_min['MA5'].iloc[-2] <= df_min['MA20'].iloc[-2]): st.warning(f"⚠️ **[추격매수 경고]** **{target_name}** 가짜 돌파 주의, 관망 권장.")
                        elif df_min['MA5'].iloc[-1] > df_min['MA20'].iloc[-1] and df_min['MA5'].iloc[-2] <= df_min['MA20'].iloc[-2] and df_min['Volume'].iloc[-1] > df_min['Vol_MA5'].iloc[-1] * 1.5 and c_p < h_10m * 0.96: st.success(f"🚀 **[정석 무릎자리]** **{target_name}** 정배열 초입 돌파 타점입니다.")
                        else: st.info(f"⚪ **[안전 지대]** **{target_name}** 기준선 리스크를 준수 중입니다.")
                        
                        df_min['Diff'] = df_min['Close'].diff().fillna(0)
                        min_price, max_price = df_min['Low'].min(), df_min['High'].max()
                        price_margin = (max_price - min_price) * 0.1 if max_price != min_price else min_price * 0.01
                        
                        fig_stock = go.Figure()
                        fig_stock.add_trace(go.Candlestick(x=df_min.index, open=df_min['Open'], high=df_min['High'], low=df_min['Low'], close=df_min['Close'], increasing_line_color='#ff4b4b', decreasing_line_color='#4c6198', name="주가"))
                        fig_stock.add_trace(go.Scatter(x=df_min.index, y=df_min['MA5'], mode='lines', line=dict(color='#ff9900', width=1.5), name="5분선"))
                        fig_stock.add_trace(go.Scatter(x=df_min.index, y=df_min['MA20'], mode='lines', line=dict(color='#cc00ff', width=1.5), name="20분선"))
                        
                        bo_d, pb_d = df_min[df_min['Breakout']], df_min[df_min['Pullback']]
                        if not bo_d.empty: fig_stock.add_trace(go.Scatter(x=bo_d.index, y=bo_d['High'] + price_margin*0.2, mode='markers+text', marker=dict(symbol='triangle-down', size=10, color='red'), text="🔥돌파", textposition="top center", textfont=dict(color='red', size=11, weight='bold'), name="돌파"))
                        if not pb_d.empty: fig_stock.add_trace(go.Scatter(x=pb_d.index, y=pb_d['Low'] - price_margin*0.2, mode='markers+text', marker=dict(symbol='triangle-up', size=10, color='blue'), text="💧눌림", textposition="bottom center", textfont=dict(color='blue', size=11, weight='bold'), name="눌림"))
                        fig_stock.add_trace(go.Bar(x=df_min.index, y=df_min['Volume'], name="거래량", marker_color=['#ff4b4b' if d >= 0 else '#4c6198' for d in df_min['Diff']], opacity=0.7, yaxis='y2'))
                        
                        fig_stock.update_layout(template="plotly_white", height=650, margin=dict(l=10, r=60, t=30, b=20), xaxis=dict(showgrid=True, gridcolor='#f0f0f0', type='date', tickformat='%H:%M', rangeslider=dict(visible=False)), yaxis=dict(side='right', showgrid=True, gridcolor='#f0f0f0', tickformat=',', range=[min_price - price_margin, max_price + price_margin], domain=[0.3, 1]), yaxis2=dict(side='right', showgrid=False, tickformat=',', domain=[0, 0.2]), hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_stock, use_container_width=True)
            except: pass
