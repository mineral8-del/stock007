import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import time
from datetime import datetime, timedelta, timezone, time as dt_time
from bs4 import BeautifulSoup
import joblib
import os
import tensorflow as tf

# 📱 1. 페이지 기본 설정 (무조건 최상단)
st.set_page_config(layout="wide", page_title="국내주식 실시간 딥러닝 스캐너", initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# [설정] 한국투자증권 API KEY (Streamlit Secrets 사용)
# -----------------------------------------------------------------------------
try:
    APP_KEY = st.secrets["KIS_APP_KEY"]
    APP_SECRET = st.secrets["KIS_APP_SECRET"]
except KeyError:
    st.error("⚠️ Streamlit secrets에 'KIS_APP_KEY' 또는 'KIS_APP_SECRET'이 설정되지 않았습니다.")
    st.stop()

URL_BASE = "https://openapi.koreainvestment.com:9443" 
KST = timezone(timedelta(hours=9))

THEME_DICT = {
    "🤖 로봇": ["두산로보틱스", "레인보우로보틱스", "뉴로메카", "에스피지", "로보티즈", "이랜시스", "로보틱스"],
    "💾 반도체": ["한미반도체", "SK하이닉스", "삼성전자", "HPSP", "이수페타시스", "제우스", "가온칩스", "리노공업", "디아이"],
    "🔋 2차전지": ["에코프로", "에코프로비엠", "에코프로머티", "포스코홀딩스", "POSCO홀딩스", "LG에너지솔루션", "엘앤에프", "금양"],
    "🧬 바이오": ["알테오젠", "HLB", "삼성바이오로직스", "셀트리온", "삼천당제약", "리가켐바이오", "휴젤"],
    "⚡ 전력기기": ["HD현대일렉트릭", "LS일렉트릭", "효성중공업", "제룡전기", "일진전기"],
    "💄 화장품": ["실리콘투", "브이티", "코스메카코리아", "씨앤씨인터내셔널", "아모레퍼시픽", "클리오"]
}

def get_theme_icon(stock_name):
    for theme, keywords in THEME_DICT.items():
        if any(keyword in stock_name for keyword in keywords):
            return theme
    return "▪️ 개별주"

# 🎯 공통 매매 타이밍 시그널 함수
def detect_signal(row):
    if row['등락률'] >= 7.0 and row['거래대금'] > 50000: 
        return "🔥 돌파매매"
    elif 1.0 <= row['등락률'] < 5.0 and row['거래대금'] > 20000: 
        return "💧 눌림목"
    return "▪️ 관망"

@st.cache_resource(ttl=3600*20)
def get_access_token():
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    url = f"{URL_BASE}/oauth2/tokenP"
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        return res.json()["access_token"]
    except: return None

def get_common_headers(tr_id):
    token = get_access_token()
    if not token: token = get_access_token()
    return {
        "Content-Type": "application/json", "authorization": f"Bearer {token}",
        "appKey": APP_KEY, "appSecret": APP_SECRET, "tr_id": tr_id
    }

@st.cache_data(ttl=15)
def get_kis_top_trading_value_stocks():
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/volume-rank"
    headers = get_common_headers("FHPST01710000")
    
    params_mid = {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171",
        "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "1", 
        "FID_BLNG_CLS_CODE": "0", "FID_TRGT_CLS_CODE": "111111111", 
        "FID_TRGT_EXLS_CLS_CODE": "111111", 
        "FID_INPUT_PRICE_1": "10000", "FID_INPUT_PRICE_2": "80000", 
        "FID_VOL_CNT": "", "FID_INPUT_DATE_1": ""
    }
    params_large = {
        "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171",
        "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "1", 
        "FID_BLNG_CLS_CODE": "0", "FID_TRGT_CLS_CODE": "111111111", 
        "FID_TRGT_EXLS_CLS_CODE": "111111", 
        "FID_INPUT_PRICE_1": "80000", "FID_INPUT_PRICE_2": "2000000", 
        "FID_VOL_CNT": "", "FID_INPUT_DATE_1": ""
    }
    
    df_list = []
    for params in [params_mid, params_large]:
        try:
            res = requests.get(url, headers=headers, params=params, timeout=5)
            data = res.json()
            if data['rt_cd'] == '0' and 'output' in data:
                df_temp = pd.DataFrame(data['output'])[['hts_kor_isnm', 'mksc_shrn_iscd', 'stck_prpr', 'prdy_ctrt', 'acml_tr_pbmn']]
                df_list.append(df_temp)
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

@st.cache_data(ttl=15)
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

# =============================================================================
# 🧠 글로벌 딥러닝 모델 로드
# =============================================================================
@st.cache_resource
def load_lstm_assets():
    try:
        if "stock_lstm_model.h5" not in os.listdir() or "lstm_scaler.pkl" not in os.listdir(): 
            return None, None
        return tf.keras.models.load_model("stock_lstm_model.h5", compile=False), joblib.load("lstm_scaler.pkl")
    except: return None, None

lstm_model, lstm_scaler = load_lstm_assets()

# =============================================================================
# 🔄 뷰(View) 라우터: 사이드바 스위치
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ 화면 모드 설정")
    is_shorts_mode = st.toggle("📱 쇼츠(세로) 방송 모드 켜기", value=False)
    st.info("스위치를 켜면 상단 메뉴가 사라지고 쇼츠용 세로 화면으로 바뀝니다.")


# =============================================================================
# 🎬 분기점 1: 쇼츠 송출용 세로 화면 
# =============================================================================
if is_shorts_mode:
    try:
        from streamlit_autorefresh import st_autorefresh
        # ⏱️ 30초(30000ms)마다 데이터 강제 리셋 및 새로고침
        st_autorefresh(interval=30000, limit=20000, key="shorts_refresh")
    except: pass

    st.markdown("""
    <style>
        .stApp { background-color: #0b1120 !important; }
        header[data-testid="stHeader"], footer, .stToolbar { display: none !important; }
        .block-container { padding: 0 !important; max-width: 100% !important; }
        ::-webkit-scrollbar { display: none !important; }
        
        /* 하단 자막을 위해 padding-bottom 50px 추가 */
        .shorts-container { padding: 12px; padding-bottom: 50px; font-family: 'Pretendard', 'Malgun Gothic', sans-serif; }
        .s-header { text-align: center; padding: 15px 0 5px 0; }
        .s-title { color: #facc15; font-size: 1.8rem; font-weight: 900; margin-bottom: 6px; letter-spacing: -0.5px; }
        .s-time-box { display: inline-block; background-color: #1e293b; color: #cbd5e1; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; border: 1px solid #334155; }
        .s-progress-line { height: 3px; background: linear-gradient(90deg, #3b82f6, #eab308, #ef4444); margin: 12px 0 15px 0; border-radius: 3px; }
        .s-card { background-color: #151e2e; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; border: 1px solid #2a364a; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
        .s-left { display: flex; align-items: center; gap: 10px; }
        .s-rank { background-color: #ef4444; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: 900; font-size: 0.9rem; flex-shrink: 0; box-shadow: 0 2px 4px rgba(239, 68, 68, 0.4); }
        .s-name-group { display: flex; flex-direction: column; justify-content: center; }
        .s-name { font-size: 1.25rem; font-weight: 900; color: white; margin-bottom: 2px; letter-spacing: -0.5px; }
        .s-badge { font-size: 0.75rem; font-weight: bold; display: flex; align-items: center; gap: 4px; }
        .badge-orange { color: #fb923c; } 
        .badge-pink { color: #f472b6; }   
        .badge-gray { color: #94a3b8; }   
        .s-right { display: flex; gap: 10px; align-items: center; }
        .s-price-group { text-align: right; }
        .s-price { font-size: 1.15rem; font-weight: 900; color: white; margin-bottom: 1px; }
        .s-ratio { font-size: 0.95rem; font-weight: 900; }
        .s-ratio.up { color: #ef4444; }
        .s-ratio.down { color: #3b82f6; }
        .s-score-box { background-color: #0f172a; padding: 4px 8px; border-radius: 8px; border: 1px solid #334155; text-align: center; min-width: 60px; display: flex; flex-direction: column; justify-content: center; }
        .s-score-label { font-size: 0.6rem; color: #94a3b8; font-weight: bold; margin-bottom: 1px; }
        .s-score { font-size: 1.1rem; font-weight: 900; color: #22c55e; } 
        
        /* 🚨 하단 스크롤 자막(Ticker) CSS */
        .ticker-wrap { position: fixed; bottom: 0; left: 0; width: 100%; overflow: hidden; background-color: #7f1d1d; height: 38px; display: flex; align-items: center; z-index: 9999; }
        .ticker-text { white-space: nowrap; color: white; font-size: 1.05rem; font-weight: 800; animation: ticker 15s linear infinite; }
        @keyframes ticker {
            0% { transform: translateX(100vw); }
            100% { transform: translateX(-100%); }
        }
    </style>
    """, unsafe_allow_html=True)

    current_time_str = datetime.now(KST).strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="shorts-container">
        <div class="s-header">
            <div class="s-title">🔴 실시간 AI 타점 스캐너</div>
            <div class="s-time-box">{current_time_str} 기준</div>
        </div>
        <div class="s-progress-line"></div>
    """, unsafe_allow_html=True)
    
    df_shorts = get_kis_top_trading_value_stocks()
    if not df_shorts.empty:
        filtered_df = df_shorts.head(30).copy()
        
        ai_scores = []
        for i, (idx, row) in enumerate(filtered_df.iterrows()):
            if lstm_model is not None and lstm_scaler is not None:
                try:
                    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice", headers=get_common_headers("FHKST03010200"), params={"FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": row['종목코드'], "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), "FID_PW_DATA_INCU_YN": "N"}).json()
                    if res.get('rt_cd') == '0' and 'output2' in res:
                        recent_10_mins = res['output2'][:10][::-1] 
                        if len(recent_10_mins) == 10:
                            scaled_min = lstm_scaler.transform(pd.DataFrame({"Open": [float(m['stck_oprc']) for m in recent_10_mins], "High": [float(m['stck_hgpr']) for m in recent_10_mins], "Low": [float(m['stck_lwpr']) for m in recent_10_mins], "Close": [float(m['stck_prpr']) for m in recent_10_mins], "Volume": [float(m['cntg_vol']) for m in recent_10_mins]}))
                            ai_scores.append(np.round(lstm_model.predict(np.expand_dims(scaled_min, axis=0), verbose=0)[0][0], 2))
                        else: ai_scores.append(0.0)
                    else: ai_scores.append(0.0)
                except: ai_scores.append(0.0)
            else:
                ai_scores.append(((row['등락률'] * 0.5) + np.log1p(row['거래대금'])).round(2))
            time.sleep(0.02) 

        filtered_df['10분_상승예측(%)'] = ai_scores
        filtered_df['매매상태'] = filtered_df.apply(detect_signal, axis=1)
        
        signal_order = ["🔥 돌파매매", "💧 눌림목", "▪️ 관망"]
        filtered_df['정렬순서'] = pd.Categorical(filtered_df['매매상태'], categories=signal_order, ordered=True)
        
        # 쇼츠는 10개만 추출
        top_10 = filtered_df.sort_values(by=['정렬순서', '10분_상승예측(%)'], ascending=[True, False]).head(10)
        
        def assign_badge_style(status):
            if status == "🔥 돌파매매": return "🔥 급등 돌파", "badge-orange"
            elif status == "💧 눌림목": return "💕 S급 눌림", "badge-pink"
            return "▪️ 관망 상태", "badge-gray"

        for i, row in top_10.reset_index(drop=True).iterrows():
            rank = i + 1
            ratio_class = "up" if row['등락률'] > 0 else "down"
            sign = "+" if row['등락률'] > 0 else ""
            ai_score_display = f"{float(row['10분_상승예측(%)']):.1f}"
            badge_text, badge_css = assign_badge_style(row['매매상태'])
            
            st.markdown(f"""
            <div class="s-card">
                <div class="s-left">
                    <div class="s-rank">{rank}</div>
                    <div class="s-name-group">
                        <div class="s-name">{row['종목명']}</div>
                        <div class="s-badge {badge_css}">{badge_text}</div>
                    </div>
                </div>
                <div class="s-right">
                    <div class="s-price-group">
                        <div class="s-price">{int(row['현재가']):,}원</div>
                        <div class="s-ratio {ratio_class}">{sign}{row['등락률']:.2f}%</div>
                    </div>
                    <div class="s-score-box">
                        <div class="s-score-label">AI 점수</div>
                        <div class="s-score">{ai_score_display}점</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # 🚨 하단 스크롤 자막 추가 (컨테이너 닫고 외부 고정 배치)
        st.markdown("""
        </div>
        <div class="ticker-wrap">
            <div class="ticker-text">
                [투자 유의사항] 본 방송은 딥러닝 AI 모델(LSTM)에 의한 단순 데이터 제공용이며 투자를 권유하지 않습니다. 모든 투자의 책임은 투자자 본인에게 있습니다.
            </div>
        </div>
        """, unsafe_allow_html=True) 

    else:
        st.warning("데이터 수집 중입니다.")

# =============================================================================
# 💻 분기점 2: 기존 메인 대시보드 화면 (일반 PC 접속 시)
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

    col_empty, col_btn = st.columns([8, 2])
    with col_btn:
        if st.button("🔄 실시간 동기화", use_container_width=True):
            st.session_state.foreign_futures_net = get_foreign_investor_trend()
            get_realtime_market_summary.clear()
            get_kis_top_trading_value_stocks.clear()
            st.rerun()

    st.markdown("---")

    now_time = datetime.now(KST).time()
    time_pre_start, time_reg_start, time_after_start, time_after_end = dt_time(8, 30), dt_time(9, 0), dt_time(15, 30), dt_time(18, 0)
    default_auto, default_pre, default_after = False, False, True
    
    if time_pre_start <= now_time < time_reg_start: default_auto, default_pre, default_after = True, True, False
    elif time_reg_start <= now_time < time_after_start: default_auto, default_pre, default_after = True, False, False

    st.info("🤖 **오토 파일럿 작동 중:** 현재 시각에 맞춰 최적의 모드가 자동으로 켜집니다.")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: auto_refresh = st.toggle("⏱️ 1분 자동 스캐닝 켜기", value=default_auto)
    with col_t2: pre_market_mode = st.toggle("☀️ 프리마켓 (08:30~09:00)", value=default_pre)
    with col_t3: after_market_mode = st.toggle("🌙 애프터 마켓 (15:30~18:00)", value=default_after)

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60000, limit=1000, key="auto_scanner_refresh")
            st.toast("🔄 스캐너 감시 중!", icon="👀")
            get_kis_top_trading_value_stocks.clear()
        except ImportError: pass

    if pre_market_mode: st.subheader("🎯 오늘 아침 시초가 타겟 (장전 예상 갭상승)")
    elif after_market_mode: st.subheader("🎯 시간외 단일가 및 내일 시초가 타겟")
    else: st.subheader("🎯 실시간 단타 타겟 (딥러닝 스코어)")

    df_universe = get_kis_top_trading_value_stocks()

    if not df_universe.empty:
        filtered_df = df_universe[df_universe['등락률'] >= 1.0].copy()
        
        # 💻 대시보드는 30개로 확장 추출 (.head(30))
        filtered_df = filtered_df.sort_values(by='거래대금', ascending=False).head(30)

        if lstm_model is not None and lstm_scaler is not None:
            my_bar = st.progress(0, text="🧠 딥러닝 모델 분석 중...")
            ai_scores = []
            for i, (idx, row) in enumerate(filtered_df.iterrows()):
                try:
                    res = requests.get(f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice", headers=get_common_headers("FHKST03010200"), params={"FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": row['종목코드'], "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), "FID_PW_DATA_INCU_YN": "N"}).json()
                    if res.get('rt_cd') == '0' and 'output2' in res:
                        recent_10_mins = res['output2'][:10][::-1] 
                        if len(recent_10_mins) == 10:
                            scaled_min = lstm_scaler.transform(pd.DataFrame({"Open": [float(m['stck_oprc']) for m in recent_10_mins], "High": [float(m['stck_hgpr']) for m in recent_10_mins], "Low": [float(m['stck_lwpr']) for m in recent_10_mins], "Close": [float(m['stck_prpr']) for m in recent_10_mins], "Volume": [float(m['cntg_vol']) for m in recent_10_mins]}))
                            ai_scores.append(np.round(lstm_model.predict(np.expand_dims(scaled_min, axis=0), verbose=0)[0][0], 2))
                        else: ai_scores.append(0.0)
                    else: ai_scores.append(0.0)
                except: ai_scores.append(0.0)
                time.sleep(0.1)
                my_bar.progress((i + 1) / len(filtered_df))
            my_bar.empty()
            filtered_df['10분_상승예측(%)'] = ai_scores
        else:
            st.warning("⚠️ LSTM 모델이 없어 기본 수식으로 대체합니다.")
            filtered_df['10분_상승예측(%)'] = ((filtered_df['등락률'] * 0.5) + np.log1p(filtered_df['거래대금'])).round(2)

        filtered_df['테마'] = filtered_df['종목명'].apply(get_theme_icon)
        filtered_df['단기_목표가'] = (filtered_df['현재가'] * 1.03).astype(int)
        filtered_df['손절가'] = (filtered_df['현재가'] * 0.98).astype(int)

        filtered_df['매매상태'] = filtered_df.apply(detect_signal, axis=1)
        top_list = filtered_df.sort_values(by='10분_상승예측(%)', ascending=False)
        
        output_dict = {
            '테마': top_list['테마'], '실시간 상태': top_list['매매상태'], 'AI 예측스코어': top_list['10분_상승예측(%)'].apply(lambda x: f"🚀 {float(x):.2f}점"), 
            '종목명': top_list['종목명'], '전일 종가(현재가)': top_list['현재가'].apply(lambda x: f"{int(x):,} 원"), '전일 상승률': top_list['등락률'].apply(lambda x: f"+{x:.2f} %"),
            '단기 목표가(+3%)': top_list['단기_목표가'].apply(lambda x: f"{x:,} 원"), '손절가(-2%)': top_list['손절가'].apply(lambda x: f"{x:,} 원"),
            '거래대금(백만)': top_list['거래대금'].apply(lambda x: f"{int(x):,}"), '종목코드': top_list['종목코드']
        }
        
        output_df = pd.DataFrame(output_dict).reset_index(drop=True)
        st.dataframe(output_df, use_container_width=True)
        
        st.info("💡 행별 상세 분석(차트 및 보안관) 기능은 구버전 호환성 유지를 위해 현재 비활성화되어 있습니다. 최신 Streamlit 버전으로 업데이트하시면 복구 가능합니다.")

    else:
        st.error("데이터를 불러오지 못했습니다.")
