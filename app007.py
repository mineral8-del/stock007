import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import time
from datetime import datetime, timedelta, timezone, time as dt_time
import FinanceDataReader as fdr
from bs4 import BeautifulSoup
import joblib
import os
import tensorflow as tf

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

st.set_page_config(layout="wide", page_title="국내주식 실시간 딥러닝 스캐너")
st.title("🚀 실시간 딥러닝 단타 및 시장 동향 대시보드")

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

@st.cache_resource(ttl=3600*20)
def get_access_token():
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    url = f"{URL_BASE}/oauth2/tokenP"
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        return res.json()["access_token"]
    except Exception as e:
        st.error(f"⚠️ 토큰 발급 실패: {e}")
        return None

def get_common_headers(tr_id):
    token = get_access_token()
    if not token:
        get_access_token.clear()
        token = get_access_token()
    return {
        "Content-Type": "application/json", "authorization": f"Bearer {token}",
        "appKey": APP_KEY, "appSecret": APP_SECRET, "tr_id": tr_id
    }

@st.cache_data(ttl=30)
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
            res = requests.get(url, headers=headers, params=params)
            data = res.json()
            if data['rt_cd'] == '0' and 'output' in data:
                df_temp = pd.DataFrame(data['output'])[['hts_kor_isnm', 'mksc_shrn_iscd', 'stck_prpr', 'prdy_ctrt', 'acml_tr_pbmn']]
                df_list.append(df_temp)
        except: 
            continue
            
    if not df_list: return pd.DataFrame()
        
    df = pd.concat(df_list, ignore_index=True)
    df.columns = ['종목명', '종목코드', '현재가', '등락률', '거래대금']
    
    exclude_keywords = ['KODEX', 'TIGER', 'KBSTAR', 'ACE', 'ARIRANG', 'HANARO', 'KOSEF', 'SOL', 'TIMEFOLIO', 'WOORI', '히어로즈', '마이티', '스팩', 'ETN']
    pattern = '|'.join(exclude_keywords)
    df = df[~df['종목명'].str.contains(pattern, case=False, regex=True)]
    
    df['현재가'] = pd.to_numeric(df['현재가'], errors='coerce')
    df['등락률'] = pd.to_numeric(df['등락률'], errors='coerce')
    df['거래대금'] = pd.to_numeric(df['거래대금'], errors='coerce') / 1000000 
    
    return df.sort_values(by='거래대금', ascending=False).drop_duplicates(subset=['종목코드']).dropna()

@st.cache_data(ttl=15)
def get_foreign_investor_trend():
    try:
        url = "https://finance.naver.com/sise/sise_index.naver?code=KOSPI"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        for dd in soup.find_all('dd'):
            text = dd.get_text(strip=True)
            if text.startswith("외국인") and "억" in text:
                clean_str = text.replace("외국인", "").replace("억", "").replace(",", "").strip()
                return float(clean_str)
                
    except Exception as e:
        st.error(f"⚠️ 실시간 수급 스크래핑 오류: {e}")
        
    return 0.0

# -----------------------------------------------------------------------------
# 🌐 실시간 지수 스크래핑 함수 (API 구조 무관형 완벽 계산 로직)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def get_realtime_market_summary():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    # 1. 코스피/코스닥 호출 (리스트 응답)
    def fetch_index(code):
        url = f"https://m.stock.naver.com/api/index/{code}/price?pageSize=20&page=1"
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json() 
        df = pd.DataFrame(data)
        df['Close'] = df['closePrice'].str.replace(',', '').astype(float)
        df['Date'] = pd.to_datetime(df['localTradedAt'])
        df = df.sort_values('Date').set_index('Date')
        
        now_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else now_price
        change_ratio = ((now_price - prev_price) / prev_price) * 100
        return df, now_price, change_ratio

    # 2. 환율 호출 (딕셔너리 내 'result' 응답)
    def fetch_exchange():
        url = "https://m.stock.naver.com/front-api/v1/marketIndex/prices?category=exchange&reutersCode=FX_USDKRW&page=1"
        res = requests.get(url, headers=headers, timeout=5)
        data = res.json().get('result', []) 
        df = pd.DataFrame(data)
        df['Close'] = df['closePrice'].str.replace(',', '').astype(float)
        df['Date'] = pd.to_datetime(df['localTradedAt'])
        df = df.sort_values('Date').set_index('Date')
        
        # 💡 API 등락률 항목 누락으로 인한 에러 방지용 직접 계산!
        now_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else now_price
        change_ratio = ((now_price - prev_price) / prev_price) * 100
        return df, now_price, change_ratio

    try: ks_data = fetch_index("KOSPI")
    except: ks_data = (pd.DataFrame(), 0.0, 0.0)
        
    try: kq_data = fetch_index("KOSDAQ")
    except: kq_data = (pd.DataFrame(), 0.0, 0.0)
        
    try: usd_data = fetch_exchange()
    except: usd_data = (pd.DataFrame(), 0.0, 0.0)

    return ks_data, kq_data, usd_data
# -----------------------------------------------------------------------------
# 🎨 [수정됨] 차트 그리기 함수 (지연 데이터로 계산하지 않고 실시간 값을 직접 텍스트로 박음)
# -----------------------------------------------------------------------------
def create_pro_chart(df, title, color_hex, now_price, change_ratio):
    if df.empty: return go.Figure().update_layout(title="데이터 로드 실패")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], mode='lines', 
        line=dict(color=color_hex, width=3), fill='tozeroy', 
        fillcolor=f"rgba({int(color_hex[1:3],16)}, {int(color_hex[3:5],16)}, {int(color_hex[5:7],16)}, 0.1)", 
        name=title
    ))
    
    sign = "+" if change_ratio > 0 else ""
    color_text = '#ff4b4b' if change_ratio > 0 else ('#0068c9' if change_ratio < 0 else '#ffffff')
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b> <span style='font-size:14px; color:{color_text}'>{now_price:,.2f} ({sign}{change_ratio:.2f}%)</span>", 
            x=0.05, y=0.85
        ), 
        height=280, margin=dict(l=10, r=10, t=50, b=10), template="plotly_dark", 
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', side='right'), 
        hovermode="x unified"
    )
    return fig

# -----------------------------------------------------------------------------
# 메인 화면 렌더링 부분 (여기도 교체해주세요)
# -----------------------------------------------------------------------------
st.subheader("🌐 글로벌 시장 및 주요 지수 실시간 모니터링")
ks_data, kq_data, usd_data = get_realtime_market_summary()

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1: st.plotly_chart(create_pro_chart(ks_data[0], "KOSPI", "#FF4B4B", ks_data[1], ks_data[2]), use_container_width=True)
with m_col2: st.plotly_chart(create_pro_chart(kq_data[0], "KOSDAQ", "#00CC96", kq_data[1], kq_data[2]), use_container_width=True)
with m_col3: st.plotly_chart(create_pro_chart(usd_data[0], "USD/KRW", "#636EFA", usd_data[1], usd_data[2]), use_container_width=True)

st.markdown("---")
st.subheader("💼 외국인 선물 수급 및 시장 주도 상태")

if 'foreign_futures_net' not in st.session_state: 
    st.session_state.foreign_futures_net = get_foreign_investor_trend()
foreign_futures_net = st.session_state.foreign_futures_net

if foreign_futures_net > 0:
    value_str = f"+{foreign_futures_net:,} 억 원"
    program_intensity = min(100, int(50 + (foreign_futures_net / 10)))
    trade_signal = "🚀 외국인 선물 대량 매수 중!"
    delta_msg = "매수 우위"
    score_color = "normal"
elif foreign_futures_net < 0:
    value_str = f"{foreign_futures_net:,} 억 원"
    program_intensity = max(0, int(50 - (abs(foreign_futures_net) / 10)))
    trade_signal = "⚠️ 외국인 선물 강한 매도세!"
    delta_msg = "매도 우위"
    score_color = "inverse"
else:
    value_str = "0.0 억 원"
    program_intensity = 50
    trade_signal = "⏸️ 수급 데이터 대기 중"
    delta_msg = "데이터 없음"
    score_color = "off"

col_m1, col_m2 = st.columns(2)
col_m1.metric(label="외국인 주식선물 순매수 금액", value=value_str, delta=delta_msg, delta_color=score_color)
col_m2.metric(label="시장 전체 우량주 매력도 환경 (100점 만점)", value=f"{program_intensity} 점", delta=trade_signal, delta_color=score_color)

if st.button("🔄 실시간 데이터 업데이트 (수동)"):
    get_foreign_investor_trend.clear() 
    st.session_state.foreign_futures_net = get_foreign_investor_trend()
    get_kis_top_trading_value_stocks.clear()
    get_market_indices_v2.clear()
    st.rerun()

st.markdown("---")

now_time = datetime.now(KST).time()
time_pre_start = dt_time(8, 30)
time_reg_start = dt_time(9, 0)
time_after_start = dt_time(15, 30)
time_after_end = dt_time(18, 0)

default_auto = False
default_pre = False
default_after = True

if time_pre_start <= now_time < time_reg_start:
    default_auto = True
    default_pre = True
    default_after = False
elif time_reg_start <= now_time < time_after_start:
    default_auto = True
    default_pre = False
    default_after = False
elif time_after_start <= now_time < time_after_end:
    default_auto = True
    default_pre = False
    default_after = True

st.info("🤖 **오토 파일럿 작동 중:** 현재 시각에 맞춰 최적의 모드(프리마켓/정규장/애프터마켓)가 자동으로 켜집니다.")

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1: auto_refresh = st.toggle("⏱️ 1분 자동 스캐닝 켜기", value=default_auto)
with col_t2: pre_market_mode = st.toggle("☀️ 프리마켓 (08:30~09:00 동시호가)", value=default_pre)
with col_t3: after_market_mode = st.toggle("🌙 애프터 마켓 (15:30~18:00 시간외)", value=default_after)

if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60000, limit=1000, key="auto_scanner_refresh")
        st.toast("🔄 스캐너가 실시간(1분 단위)으로 감시 중입니다!", icon="👀")
        get_kis_top_trading_value_stocks.clear()
    except ImportError: 
        pass

if pre_market_mode: 
    st.subheader("🎯 오늘 아침 시초가 타겟 (장전 예상 갭상승)")
elif after_market_mode: 
    st.subheader("🎯 시간외 단일가 및 내일 시초가 타겟")
else: 
    st.subheader("🎯 실시간 단타 타겟 (딥러닝 10분 상승 예측 스코어)")

df_universe = get_kis_top_trading_value_stocks()

if not df_universe.empty:
    # -------------------------------------------------------------------------
    # 🚀 [딥러닝 통합] 타이트 필터링 및 LSTM 예측 로직
    # -------------------------------------------------------------------------
    # 조건: 상승 흐름을 탄 거래대금 최상위 15개 종목만 추출 (API 서버 밴 방어)
    filtered_df = df_universe[df_universe['등락률'] >= 1.0].copy()
    filtered_df = filtered_df.sort_values(by='거래대금', ascending=False).head(15)

    @st.cache_resource
    def load_lstm_assets():
        try:
            import os
            # 파일이 진짜 서버에 존재하긴 하는지 디버깅용 확인
            files = os.listdir()
            if "stock_lstm_model.h5" not in files:
                st.error("❌ 'stock_lstm_model.h5' 파일이 리포지토리에 없습니다!")
            if "lstm_scaler.pkl" not in files:
                st.error("❌ 'lstm_scaler.pkl' 파일이 리포지토리에 없습니다!")

            model = tf.keras.models.load_model("stock_lstm_model.h5", compile=False)
            scaler = joblib.load("lstm_scaler.pkl")
            return model, scaler
        except Exception as e:
            st.error(f"🚨 모델 로드 중 치명적 오류 발생: {e}")
            return None, None

    lstm_model, lstm_scaler = load_lstm_assets()
    if lstm_model is not None and lstm_scaler is not None:
        my_bar = st.progress(0, text="🧠 딥러닝 모델이 대장주 15개의 1분봉 패턴을 분석 중입니다...")
        ai_scores = []
        
        url_min = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        headers_min = get_common_headers("FHKST03010200")
        
        for i, (idx, row) in enumerate(filtered_df.iterrows()):
            code = row['종목코드']
            params_min = {
                "FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", 
                "FID_INPUT_ISCD": code, 
                "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), 
                "FID_PW_DATA_INCU_YN": "N"
            }
            
            try:
                res = requests.get(url_min, headers=headers_min, params=params_min)
                res_data = res.json()
                
                if res_data['rt_cd'] == '0' and 'output2' in res_data:
                    # KIS API는 최신순으로 오므로 [::-1]로 뒤집어서 과거->현재 순으로 맞춤
                    recent_10_mins = res_data['output2'][:10][::-1] 
                    
                    if len(recent_10_mins) == 10:
                        df_min_temp = pd.DataFrame({
                            "Open": [float(m['stck_oprc']) for m in recent_10_mins],
                            "High": [float(m['stck_hgpr']) for m in recent_10_mins],
                            "Low": [float(m['stck_lwpr']) for m in recent_10_mins],
                            "Close": [float(m['stck_prpr']) for m in recent_10_mins],
                            "Volume": [float(m['cntg_vol']) for m in recent_10_mins]
                        })
                        
                        scaled_min = lstm_scaler.transform(df_min_temp)
                        X_live_input = np.expand_dims(scaled_min, axis=0) # (1, 10, 5) 형태로 변환
                        
                        pred = lstm_model.predict(X_live_input, verbose=0)
                        ai_scores.append(np.round(pred[0][0], 2))
                    else:
                        ai_scores.append(0.0)
                else:
                    ai_scores.append(0.0)
            except Exception as e:
                ai_scores.append(0.0)
                
            time.sleep(0.1) # KIS API 호출 제한 방어
            my_bar.progress((i + 1) / len(filtered_df))
            
        my_bar.empty()
        filtered_df['10분_상승예측(%)'] = ai_scores
    else:
        st.warning("⚠️ LSTM 모델이 없어 기본 수식으로 대체합니다. (오프라인 학습을 먼저 진행하세요)")
        filtered_df['10분_상승예측(%)'] = ((filtered_df['등락률'] * 0.5) + np.log1p(filtered_df['거래대금'])).round(2)

    # 데이터 가공
    filtered_df['테마'] = filtered_df['종목명'].apply(get_theme_icon)
    filtered_df['단기_목표가'] = (filtered_df['현재가'] * 1.03).astype(int)
    filtered_df['손절가'] = (filtered_df['현재가'] * 0.98).astype(int)

    def detect_signal(row):
        if row['등락률'] >= 7.0 and row['거래대금'] > 50000: return "🔥 돌파매매"
        elif 1.0 <= row['등락률'] < 5.0 and row['거래대금'] > 20000: return "💧 눌림목"
        return "▪️ 관망"
        
    filtered_df['매매상태'] = filtered_df.apply(detect_signal, axis=1)
    
    top_30 = filtered_df.sort_values(by='10분_상승예측(%)', ascending=False)
    
    if pre_market_mode:
        extra_df = fetch_pre_market_data(top_30)
        top_30 = pd.merge(top_30, extra_df, on='종목코드', how='left').sort_values(by='_sort_ratio_num', ascending=False)
    elif after_market_mode:
        extra_df = fetch_after_market_data(top_30)
        top_30 = pd.merge(top_30, extra_df, on='종목코드', how='left').sort_values(by='_sort_ratio_num', ascending=False)

    output_dict = {
        '테마': top_30['테마'],
        '실시간 상태': top_30['매매상태'],
        'AI 예측스코어': top_30['10분_상승예측(%)'].apply(lambda x: f"🚀 {float(x):.2f}점"), 
        '종목명': top_30['종목명'],
        '전일 종가(현재가)': top_30['현재가'].apply(lambda x: f"{int(x):,} 원"),
        '전일 상승률': top_30['등락률'].apply(lambda x: f"+{x:.2f} %"),
    }
    
    if pre_market_mode:
        output_dict['☀️ 예상 갭상승률'] = top_30['☀️ 예상 갭상승률']
        output_dict['☀️ 예상 체결가'] = top_30['☀️ 예상 체결가']
        output_dict['☀️ 예상 거래량'] = top_30['☀️ 예상 거래량']
    elif after_market_mode:
        output_dict['🌙 시간외 등락률'] = top_30['시간외 등락률']
        output_dict['🌙 시간외 현재가'] = top_30['시간외 현재가']
        output_dict['🌙 시간외 거래량'] = top_30['시간외 거래량']
    else:
        output_dict['단기 목표가(+3%)'] = top_30['단기_목표가'].apply(lambda x: f"{x:,} 원")
        output_dict['손절가(-2%)'] = top_30['손절가'].apply(lambda x: f"{x:,} 원")
        
    output_dict['거래대금(백만)'] = top_30['거래대금'].apply(lambda x: f"{int(x):,}")
    output_dict['종목코드'] = top_30['종목코드']
    
    output_df = pd.DataFrame(output_dict).reset_index(drop=True)

    selected_rows = st.dataframe(output_df, use_container_width=True, selection_mode="single-row", on_select="rerun")
else:
    st.error("데이터를 불러오지 못했습니다. 장외 시간이거나 API 호출 초과일 수 있습니다.")
    output_df = pd.DataFrame()

# -----------------------------------------------------------------------------
# 종목 클릭 시 차트 및 보안관 연동 (기존 로직 유지)
# -----------------------------------------------------------------------------
st.markdown("---")
selected_idx = selected_rows.selection.rows[0] if (hasattr(selected_rows, 'selection') and len(selected_rows.selection.rows) > 0) else 0

if not output_df.empty and selected_idx < len(output_df):
    target_code = output_df.iloc[selected_idx]['종목코드']
    target_name = output_df.iloc[selected_idx]['종목명']
    target_theme = output_df.iloc[selected_idx]['테마']
    target_price = output_df.iloc[selected_idx]['전일 종가(현재가)']
    target_change = output_df.iloc[selected_idx]['전일 상승률']
    target_vol = output_df.iloc[selected_idx]['거래대금(백만)']
    
    st.markdown(f"<div style='padding:10px 0; border-bottom:1px solid #ddd; margin-bottom:15px;'><span style='font-size:20px; font-weight:bold;'>{target_name}</span> <span style='font-size:14px; color:#555;'>[{target_theme}]</span><span style='font-size:14px; font-weight:bold; margin-left:15px;'>{target_price}</span><span style='font-size:14px; color:#e12929; margin-left:5px;'>{target_change}</span><span style='font-size:14px; color:#888; margin-left:10px;'>누적 거래대금 {target_vol}백만</span></div>", unsafe_allow_html=True)
    
    with st.spinner(f"[{target_name}] 1분봉 데이터 및 고점 폭락 위험성을 정밀 진단 중입니다..."):
        url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        headers = get_common_headers("FHKST03010200")
        params = {
            "FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", 
            "FID_INPUT_ISCD": target_code, 
            "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), 
            "FID_PW_DATA_INCU_YN": "Y"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params)
            res_data = res.json()
            if res_data['rt_cd'] == '0' and 'output2' in res_data:
                min_data = res_data['output2'][::-1] 
                df_min = pd.DataFrame({
                    "Open": [float(m['stck_oprc']) for m in min_data], 
                    "High": [float(m['stck_hgpr']) for m in min_data], 
                    "Low": [float(m['stck_lwpr']) for m in min_data], 
                    "Close": [float(m['stck_prpr']) for m in min_data], 
                    "Volume": [float(m['cntg_vol']) for m in min_data]
                }, index=pd.to_datetime([f"{m['stck_bsop_date']} {m['stck_cntg_hour']}" for m in min_data], format="%Y%m%d %H%M%S"))
                
                df_min = df_min[df_min['Close'] > 0]
                
                if not df_min.empty:
                    df_min['MA5'] = df_min['Close'].rolling(5).mean()
                    df_min['MA20'] = df_min['Close'].rolling(20).mean()
                    df_min['Vol_MA5'] = df_min['Volume'].rolling(5).mean()
                    
                    df_min['Breakout'] = (df_min['Close'] > df_min['High'].shift(1).rolling(20).max()) & (df_min['Volume'] > df_min['Vol_MA5'] * 1.5)
                    df_min['Pullback'] = (df_min['MA20'] > df_min['MA20'].shift(3)) & (df_min['Low'] <= df_min['MA20'] * 1.005) & (df_min['Close'] >= df_min['MA20'] * 0.998) & (df_min['Volume'] < df_min['Vol_MA5'])
                    
                    c_p = df_min['Close'].iloc[-1]
                    h_10m = df_min['High'].iloc[-10:].max()
                    
                    if c_p < df_min['MA5'].iloc[-1] and c_p <= h_10m * 0.97: 
                        st.error(f"💣 **[🚨 보안관 비상경보]** **{target_name}** 종목은 5분선이 붕괴되어 급락 위험이 큽니다. 신규 진입 금지!")
                    elif c_p >= h_10m * 0.98 and not (df_min['MA5'].iloc[-1] > df_min['MA20'].iloc[-1] and df_min['MA5'].iloc[-2] <= df_min['MA20'].iloc[-2]): 
                        st.warning(f"⚠️ **[추격매수 경고]** **{target_name}** 종목은 가짜 돌파에 걸릴 확률이 높으니 관망하십시오.")
                    elif df_min['MA5'].iloc[-1] > df_min['MA20'].iloc[-1] and df_min['MA5'].iloc[-2] <= df_min['MA20'].iloc[-2] and df_min['Volume'].iloc[-1] > df_min['Vol_MA5'].iloc[-1] * 1.5 and c_p < h_10m * 0.96: 
                        st.success(f"🚀 **[정석 무릎자리]** **{target_name}** 정배열 초입 돌파가 확인된 타점입니다.")
                    else: 
                        st.info(f"⚪ **[안전 지대]** **{target_name}** 기준선 리스크를 준수 중입니다.")
                    
                    df_min['Diff'] = df_min['Close'].diff().fillna(0)
                    min_price, max_price = df_min['Low'].min(), df_min['High'].max()
                    price_margin = (max_price - min_price) * 0.1 if max_price != min_price else min_price * 0.01
                    
                    fig_stock = go.Figure()
                    fig_stock.add_trace(go.Candlestick(x=df_min.index, open=df_min['Open'], high=df_min['High'], low=df_min['Low'], close=df_min['Close'], increasing_line_color='#ff4b4b', decreasing_line_color='#4c6198', name="주가"))
                    fig_stock.add_trace(go.Scatter(x=df_min.index, y=df_min['MA5'], mode='lines', line=dict(color='#ff9900', width=1.5), name="5분선", hoverinfo='skip'))
                    fig_stock.add_trace(go.Scatter(x=df_min.index, y=df_min['MA20'], mode='lines', line=dict(color='#cc00ff', width=1.5), name="20분선", hoverinfo='skip'))
                    
                    bo_d = df_min[df_min['Breakout']]
                    if not bo_d.empty: 
                        fig_stock.add_trace(go.Scatter(x=bo_d.index, y=bo_d['High'] + price_margin*0.2, mode='markers+text', marker=dict(symbol='triangle-down', size=10, color='red'), text="🔥돌파", textposition="top center", textfont=dict(color='red', size=11, weight='bold'), name="돌파"))
                        
                    pb_d = df_min[df_min['Pullback']]
                    if not pb_d.empty: 
                        fig_stock.add_trace(go.Scatter(x=pb_d.index, y=pb_d['Low'] - price_margin*0.2, mode='markers+text', marker=dict(symbol='triangle-up', size=10, color='blue'), text="💧눌림", textposition="bottom center", textfont=dict(color='blue', size=11, weight='bold'), name="눌림"))
                    
                    fig_stock.add_trace(go.Bar(x=df_min.index, y=df_min['Volume'], name="거래량", marker_color=['#ff4b4b' if d >= 0 else '#4c6198' for d in df_min['Diff']], opacity=0.7, yaxis='y2'))
                    
                    fig_stock.update_layout(
                        template="plotly_white", height=650, margin=dict(l=10, r=60, t=30, b=20), 
                        xaxis=dict(showgrid=True, gridcolor='#f0f0f0', type='date', tickformat='%H:%M', rangeslider=dict(visible=False)), 
                        yaxis=dict(side='right', showgrid=True, gridcolor='#f0f0f0', tickformat=',', range=[min_price - price_margin, max_price + price_margin], domain=[0.3, 1]), 
                        yaxis2=dict(side='right', showgrid=False, tickformat=',', domain=[0, 0.2]), 
                        hovermode='x unified', showlegend=True, 
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_stock, use_container_width=True)
        except: 
            pass
