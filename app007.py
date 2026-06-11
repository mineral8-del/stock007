import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import time
from datetime import datetime, timedelta, timezone, time as dt_time
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import urllib.parse
import joblib
import os
import base64
import asyncio
import edge_tts
import tensorflow as tf
import streamlit.components.v1 as components
from google import genai

# 📱 1. 페이지 기본 설정 (무조건 최상단)
st.set_page_config(layout="wide", page_title="국내주식 실시간 딥러닝 스캐너", initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# [설정] 한국투자증권 및 구글 API KEY
# -----------------------------------------------------------------------------
try:
    KIS_APP_KEY = st.secrets["KIS_APP_KEY"]
    KIS_APP_SECRET = st.secrets["KIS_APP_SECRET"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    
    APP_KEY = KIS_APP_KEY
    APP_SECRET = KIS_APP_SECRET
except KeyError:
    st.error("⚠️ Streamlit secrets에 KIS_APP_KEY, KIS_APP_SECRET 또는 GEMINI_API_KEY 설정이 누락되었습니다.")
    st.stop()

URL_BASE = "https://openapi.koreainvestment.com:9443" 
KST = timezone(timedelta(hours=9))
client = genai.Client(api_key=GEMINI_API_KEY)

THEME_DICT = {
    "🤖 로봇": ["두산로보틱스", "레인보우로보틱스", "뉴로메카", "에스피지", "로보티즈", "이랜시스", "로보틱스"],
    "💾 반도체": ["한미반도체", "SK하이닉스", "삼성전자", "HPSP", "이수페타시스", "제우스", "가온칩스", "리노공업", "디아이"],
    "🔋 2차전지": ["에코프로", "エ코프로비엠", "에코프로머티", "포스코홀딩스", "POSCO홀딩스", "LG에너지솔루션", "엘앤에프", "금양"],
    "🧬 바이오": ["알테오젠", "HLB", "삼성바이오로직스", "셀트리온", "삼천당제약", "리가켐바이오", "휴젤"],
    "⚡ 전력기기": ["HD현대일렉트릭", "LS일렉트릭", "효성중공업", "제룡전기", "일진전기"],
    "💄 화장품": ["실리콘투", "브이티", "코스메카코리아", "씨앤씨인터내셔널", "아모레퍼시픽", "클리오"]
}

def get_theme_icon(stock_name):
    for theme, keywords in THEME_DICT.items():
        if any(keyword in stock_name for keyword in keywords):
            return theme
    return "▪️ 개별주"

# -----------------------------------------------------------------------------
# 🤖 쇼츠 라이브용 AI 및 초경량 데이터 함수 모음
# -----------------------------------------------------------------------------
def get_latest_news():
    url = "https://news.google.com/rss/search?q=주식+특징주+OR+증시+시황+when:1d&hl=ko&gl=KR&ceid=KR:ko"
    try:
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.content)
        return [item.find('title').text.rsplit('-', 1)[0].strip() for item in root.findall('.//item')[:5]]
    except:
        return ["현재 시각 국내 증시 주요 특징주 시황을 전해드립니다."] * 5

def generate_shorts_script(news_list):
    news_text = "\n".join([f"- {news}" for news in news_list])
    prompt = f"다음 뉴스 목록을 바탕으로 유투브 쇼츠 대본 5줄을 증권 전문가 톤으로 글머리기호 없이 작성하세요:\n{news_text}"
    for attempt in range(3):
        try:
            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            return response.text.strip()
        except:
            time.sleep(5)
    return "현재 AI 서버 접속이 지연되고 있습니다. 화면의 실시간 데이터를 참고해 주십시오."

def download_ai_image(korean_sentence):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        translation_prompt = f"Translate into short english keywords for image generation: {korean_sentence}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=translation_prompt)
        safe_keywords = urllib.parse.quote(response.text.strip())
        img_res = requests.get(f"https://image.pollinations.ai/prompt/{safe_keywords}?width=1080&height=1920&nologo=true", headers=headers, timeout=15)
        if img_res.status_code == 200:
            return base64.b64encode(img_res.content).decode()
    except: pass
    return ""

def create_audio_b64(text):
    filename = "temp_briefing.mp3"
    try:
        communicate = edge_tts.Communicate(text, "ko-KR-SunHiNeural")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(communicate.save(filename))
        with open(filename, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        os.remove(filename)
        return b64
    except: return ""

# 🚀 [핵심 추가] 쇼츠 카드 내부에 들어갈 초경량 미니 차트용 데이터 수집 함수
@st.cache_data(ttl=30)
def get_mini_chart_path(stock_code, stroke_color):
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = get_common_headers("FHKST03010200")
    params = {"FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code, "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), "FID_PW_DATA_INCU_YN": "N"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=3).json()
        if res.get('rt_cd') == '0' and 'output2' in res:
            prices = [float(m['stck_prpr']) for m in res['output2'][:12][::-1]] # 최근 12분 추이
            if len(prices) >= 2:
                min_p, max_p = min(prices), max(prices)
                rng = (max_p - min_p) if max_p != min_p else 1
                # SVG 좌표 계산 (가로 400, 세로 45 크기에 맞춤)
                points = [f"{(idx / (len(prices) - 1)) * 380 + 10},{40 - ((p - min_p) / rng) * 35}" for idx, p in enumerate(prices)]
                return f"""
                <div class="s-mini-chart">
                    <svg width="100%" height="45" viewBox="0 0 400 45">
                        <polyline fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{" ".join(points)}"/>
                    </svg>
                </div>
                """
    except: pass
    return "<div style='height:10px;'></div>"

# -----------------------------------------------------------------------------
# 📊 기존 한국투자증권 연동 공통 함수
# -----------------------------------------------------------------------------
@st.cache_resource(ttl=3600*20)
def get_access_token():
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    try:
        res = requests.post(f"{URL_BASE}/oauth2/tokenP", headers=headers, data=json.dumps(body))
        return res.json()["access_token"]
    except: return None

def get_common_headers(tr_id):
    token = get_access_token()
    if not token: token = get_access_token()
    return {"Content-Type": "application/json", "authorization": f"Bearer {token}", "appKey": APP_KEY, "appSecret": APP_SECRET, "tr_id": tr_id}

@st.cache_data(ttl=30)
def get_kis_top_trading_value_stocks():
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/volume-rank"
    headers = get_common_headers("FHPST01710000")
    params_mid = {"FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171", "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "1", "FID_BLNG_CLS_CODE": "0", "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "111111", "FID_INPUT_PRICE_1": "10000", "FID_INPUT_PRICE_2": "80000", "FID_VOL_CNT": "", "FID_INPUT_DATE_1": ""}
    params_large = {"FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171", "FID_INPUT_ISCD": "0000", "FID_DIV_CLS_CODE": "1", "FID_BLNG_CLS_CODE": "0", "FID_TRGT_CLS_CODE": "111111111", "FID_TRGT_EXLS_CLS_CODE": "111111", "FID_INPUT_PRICE_1": "80000", "FID_INPUT_PRICE_2": "2000000", "FID_VOL_CNT": "", "FID_INPUT_DATE_1": ""}
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


# =============================================================================
# ⚙️ 안전한 화면 모드 전환 스위치
# =============================================================================
with st.expander("⚙️ OBS 방송 송출용 화면 설정 (클릭하여 열기)"):
    is_shorts_mode = st.checkbox("📱 쇼츠(세로형) 라이브 모드 켜기", value=False)
    st.info("체크박스를 켜면 대시보드가 검은 배경의 세로형 라이브 방송 전용 스크린으로 자동 변환됩니다.")

# =============================================================================
# 🎬 분기점 1: 쇼츠 송출용 세로 화면 (AI 브리핑 + 30초 로딩바 + 전종목 미니차트)
# =============================================================================
if is_shorts_mode:
    now = datetime.now(KST)
    is_weekday = now.weekday() < 5
    is_market_open = is_weekday and (dt_time(8, 30) <= now.time() <= dt_time(15, 30))

    if 'bg_image_b64' not in st.session_state: st.session_state.bg_image_b64 = ""
    if 'last_briefing_hour' not in st.session_state: st.session_state.last_briefing_hour = -1

    try:
        from streamlit_autorefresh import st_autorefresh
        refresh_interval = 30000 if is_market_open else 600000
        st_autorefresh(interval=refresh_interval, limit=10000, key="shorts_refresh")
    except: pass

    # 🛑 장 마감 대기 화면 정책
    if not is_market_open:
        st.markdown("""
        <style>
            html, body, .stApp { background-color: #0b1120 !important; color: white !important; display: flex; justify-content: center; align-items: center; height: 100vh; }
            [data-testid="collapsedControl"], section[data-testid="stSidebar"], header[data-testid="stHeader"] { display: none !important; }
        </style>
        <div style='text-align: center;'>
            <h1 style='color:#facc15; font-size:4rem;'>🔴 실시간 AI 타점 스캐너</h1>
            <h2 style='color:#94a3b8; font-size:2.5rem; margin-top:20px;'>정규 방송 시간이 아닙니다.</h2>
            <p style='color:#cbd5e1; font-size:1.8rem;'>방송 시간: 평일 08:30 ~ 15:30</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # 🎤 정각 AI 방송 제어 영역
    current_hour, current_minute = now.hour, now.minute
    audio_html = ""
    if (current_hour == 9 or current_hour == 15) and (0 <= current_minute < 5):
        if st.session_state.last_briefing_hour != current_hour:
            with st.spinner("AI가 시장 요약 정보를 브리핑하는 중입니다..."):
                news = get_latest_news()
                script_body = generate_shorts_script(news)
                full_script = f"현재 시각, {now.month}월 {now.day}일 {now.hour}시 주식시장 상황입니다. {script_body}"
                audio_b64 = create_audio_b64(full_script)
                if audio_b64:
                    audio_html = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_b64}"></audio>'
                st.session_state.bg_image_b64 = download_ai_image(script_body.split('\n')[0])
                st.session_state.last_briefing_hour = current_hour

    # 🎨 무인 라이브 방송 맞춤형 스킨 패키지 (미니 차트 스타일 포함)
    bg_css = ""
    if st.session_state.get('bg_image_b64'):
        bg_css = f"""[data-testid="stAppViewContainer"], .stApp {{ background-image: linear-gradient(rgba(11, 17, 32, 0.4), rgba(11, 17, 32, 0.4)), url("data:image/jpeg;base64,{st.session_state.bg_image_b64}") !important; background-size: cover !important; background-position: center !important; }}"""
    else:
        bg_css = """[data-testid="stAppViewContainer"], .stApp { background-image: linear-gradient(rgba(11, 17, 32, 0.6), rgba(11, 17, 32, 0.6)), url("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1080&auto=format&fit=crop") !important; background-size: cover !important; background-position: center !important; }"""

    st.markdown(f"""
    <style>
        html, body {{ background-color: #0b1120 !important; color: white !important; }}
        {bg_css}
        [data-testid="collapsedControl"], section[data-testid="stSidebar"], header[data-testid="stHeader"], .stApp > header {{ display: none !important; }}
        .block-container {{ padding: 0.5rem 1.5rem 4.5rem 1.5rem !important; max-width: 1000px !important; margin: 0 auto !important; }}
        ::-webkit-scrollbar {{ display: none !important; }}
        
        .s-title {{ text-align: center; color: #facc15; font-size: 2.9rem; font-weight: 900; margin-bottom: 6px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
        .s-market-board {{ background-color: rgba(30, 41, 59, 0.85); border-radius: 12px; padding: 12px 20px; margin-bottom: 10px; border: 2px solid #334155; display: flex; justify-content: space-between; box-shadow: 0 2px 4px rgba(0,0,0,0.3); backdrop-filter: blur(5px); }}
        .s-market-col {{ display: flex; flex-direction: column; gap: 6px; }}
        .s-market-right {{ text-align: right; align-items: flex-end; }}
        .s-m-item {{ display: flex; align-items: baseline; gap: 8px; }}
        .s-m-label {{ font-size: 1.6rem; color: #94a3b8; font-weight: bold; }}
        .s-m-val {{ font-size: 2.1rem; font-weight: 900; }}
        .s-m-sub {{ font-size: 1.5rem; margin-left: 4px; }}
        
        .m-up {{ color: #ef4444; }} .m-down {{ color: #3b82f6; }} .m-off {{ color: #e2e8f0; }}
        
        /* 📦 카드 내부 패딩 조정 및 미니차트 경계선 확보 */
        .s-card {{ background-color: rgba(30, 41, 59, 0.80); border-radius: 12px; padding: 11px 18px 6px 18px; margin-bottom: 9px; border: 1px solid #334155; display: flex; flex-direction: column; gap: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); backdrop-filter: blur(5px); }}
        .s-card-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .s-rank-name {{ display: flex; align-items: center; gap: 12px; }}
        .s-rank {{ background-color: #ef4444; color: white; border-radius: 50%; min-width: 32px; height: 32px; display: flex; justify-content: center; align-items: center; font-size: 1.5rem; font-weight: bold; }}
        .s-name {{ font-size: 2.2rem; font-weight: 900; }}
        .s-eval {{ font-size: 1.3rem; color: #facc15; font-weight: bold; background: #0f172a; padding: 4px 8px; border-radius: 6px; border: 1px solid #475569; }}
        .s-card-bottom {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px dashed rgba(255,255,
