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
# 💡 [수정됨] 여기서 텐서플로우를 무조건 부르지 않고, 아래쪽 필요할 때만 부르도록 내렸습니다.
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

@st.cache_data(ttl=30)
def get_mini_chart_path(stock_code, stroke_color):
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = get_common_headers("FHKST03010200")
    params = {"FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code, "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), "FID_PW_DATA_INCU_YN": "N"}
    try:
        res = requests.get(url, headers=headers, params=params, timeout=3).json()
        if res.get('rt_cd') == '0' and 'output2' in res:
            prices = [float(m['stck_prpr']) for m in res['output2'][:12][::-1]] 
            if len(prices) >= 2:
                min_p, max_p = min(prices), max(prices)
                rng = (max_p - min_p) if max_p != min_p else 1
                points_list = [f"{(idx / (len(prices) - 1)) * 380 + 10},{40 - ((p - min_p) / rng) * 35}" for idx, p in enumerate(prices)]
                points_str = " ".join(points_list)
                return f"""
                <div class="s-mini-chart">
                    <svg width="100%" height="45" viewBox="0 0 400 45">
                        <polyline fill="none" stroke="{stroke_color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{points_str}"/>
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
        
        .s-card {{ background-color: rgba(30, 41, 59, 0.80); border-radius: 12px; padding: 11px 18px 6px 18px; margin-bottom: 9px; border: 1px solid #334155; display: flex; flex-direction: column; gap: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); backdrop-filter: blur(5px); }}
        .s-card-top {{ display: flex; justify-content: space-between; align-items: center; }}
        .s-rank-name {{ display: flex; align-items: center; gap: 12px; }}
        .s-rank {{ background-color: #ef4444; color: white; border-radius: 50%; min-width: 32px; height: 32px; display: flex; justify-content: center; align-items: center; font-size: 1.5rem; font-weight: bold; }}
        .s-name {{ font-size: 2.2rem; font-weight: 900; }}
        .s-eval {{ font-size: 1.3rem; color: #facc15; font-weight: bold; background: #0f172a; padding: 4px 8px; border-radius: 6px; border: 1px solid #475569; }}
        .s-card-bottom {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px dashed rgba(255,255,255,0.1); padding-bottom: 5px; margin-bottom: 3px; }}
        .s-price-box {{ display: flex; align-items: baseline; gap: 10px; }}
        .s-price {{ font-size: 2.0rem; color: #e2e8f0; font-weight: bold; }}
        .s-ratio {{ font-size: 1.9rem; font-weight: 900; }}
        .s-score {{ font-size: 1.4rem; color: #10b981; font-weight: bold; background: rgba(16, 185, 129, 0.15); padding: 4px 8px; border-radius: 6px; }}
        
        .s-mini-chart {{ width: 100%; height: 45px; display: flex; align-items: center; justify-content: center; opacity: 0.95; }}
        .s-marquee {{ position: fixed; bottom: 0; left: 0; width: 100%; background-color: #b91c1c; color: white; padding: 14px 0; font-size: 1.9rem; font-weight: bold; box-shadow: 0 -3px 10px rgba(0,0,0,0.5); z-index: 9999; }}
    </style>
    {audio_html}
    """, unsafe_allow_html=True)

    st.markdown("<div class='s-title'>🔴 실시간 AI 타점 스캐너 라이브 방송</div>", unsafe_allow_html=True)

    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { margin: 0; padding: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; }
        .s-time { color: #94a3b8; font-family: 'Malgun Gothic', sans-serif; font-size: 1.4rem; font-weight: bold; background-color: rgba(30, 41, 59, 0.9); padding: 4px 15px; border-radius: 20px; border: 1px solid #334155; margin-bottom: 15px; }
        .progress-container { width: 96%; max-width: 980px; height: 15px; background-color: rgba(15, 23, 42, 0.8); border-radius: 8px; border: 1px solid #334155; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6); }
        .progress-bar { height: 100%; width: 0%; background-image: linear-gradient(-45deg, rgba(255, 255, 255, 0.25) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.25) 50%, rgba(255, 255, 255, 0.25) 75%, transparent 75%, transparent); background-size: 30px 30px; animation: fillBar 30s linear infinite, moveStripes 1s linear infinite; }
        @keyframes fillBar { 0% { width: 0%; background-color: #3b82f6; } 25% { background-color: #06b6d4; } 50% { background-color: #10b981; } 75% { background-color: #f59e0b; } 99% { width: 100%; background-color: #ef4444; } 100% { width: 100%; background-color: #ef4444; } }
        @keyframes moveStripes { 0% { background-position: 0 0; } 100% { background-position: 30px 0; } }
    </style>
    </head>
    <body>
        <div id="live-clock" class="s-time">🕒 시간 동기화 중...</div>
        <div class="progress-container"><div class="progress-bar"></div></div>
        <script>
            function updateClock() {
                const now = new Date();
                document.getElementById('live-clock').innerText = '🕒 ' + now.getFullYear() + '년 ' + String(now.getMonth() + 1).padStart(2, '0') + '월 ' + String(now.getDate()).padStart(2, '0') + '일 ' + String(now.getHours()).padStart(2, '0') + '시 ' + String(now.getMinutes()).padStart(2, '0') + '분 ' + String(now.getSeconds()).padStart(2, '0') + '초';
            }
            setInterval(updateClock, 1000); updateClock();
        </script>
    </body>
    </html>
    """, height=85)

    (ks_price, ks_ratio), (kq_price, kq_ratio), (usd_price, usd_change) = get_realtime_market_summary()
    foreign_futures_net = get_foreign_investor_trend()
    usd_price_clean = str(usd_price).split('.')[0]
    ks_class = "m-up" if ks_ratio > 0 else ("m-down" if ks_ratio < 0 else "m-off")
    kq_class = "m-up" if kq_ratio > 0 else ("m-down" if kq_ratio < 0 else "m-off")
    usd_class = "m-up" if "+" in str(usd_change) else ("m-down" if "-" in str(usd_change) else "m-off")
    foreign_val = f"+{foreign_futures_net:,.0f}억" if foreign_futures_net > 0 else f"{foreign_futures_net:,.0f}억"
    foreign_class = "m-up" if foreign_futures_net > 0 else ("m-down" if foreign_futures_net < 0 else "m-off")

    st.markdown(f"""
        <div class="s-market-board">
            <div class="s-market-col">
                <div class="s-m-item"><span class="s-m-label">코스피</span><span class="s-m-val {ks_class}">{ks_price}<span class="s-m-sub">{ks_ratio:+.2f}%</span></span></div>
                <div class="s-m-item"><span class="s-m-label">코스닥</span><span class="s-m-val {kq_class}">{kq_price}<span class="s-m-sub">{kq_ratio:+.2f}%</span></span></div>
            </div>
            <div class="s-market-col s-market-right">
                <div class="s-m-item"><span class="s-m-label">환율(원)</span><span class="s-m-val {usd_class}">{usd_price_clean}</span></div>
                <div class="s-m-item"><span class="s-m-label">외인선물</span><span class="s-m-val {foreign_class}">{foreign_val}</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    df_shorts = get_kis_top_trading_value_stocks()
    if not df_shorts.empty:
        df_shorts = df_shorts[df_shorts['등락률'] > 1.0].copy()
        df_shorts['스코어'] = ((df_shorts['등락률'] * 0.5) + np.log1p(df_shorts['거래대금'])).round(2)
        top_10 = df_shorts.sort_values(by='스코어', ascending=False).head(10)
        
        for i, (index, row) in enumerate(top_10.iterrows(), 1):
            rate = row['등락률']
            vol = row['거래대금']
            
            if rate >= 7.0 and vol > 500: eval_text = "🔥 급등돌파"
            elif 1.0 <= rate < 5.0 and vol > 200: eval_text = "💧 눌림목"
            else: eval_text = "⚡ AI포착"
            
            stroke_color = "#ef4444" if rate > 0 else "#3b82f6"
            sign = "+" if rate > 0 else ""
            
            mini_chart_svg = get_mini_chart_path(row['종목코드'], stroke_color)
            
            st.markdown(f"""
                <div class="s-card">
                    <div class="s-card-top">
                        <div class="s-rank-name">
                            <div class="s-rank">{i}</div>
                            <div class="s-name">{row['종목명']}</div>
                        </div>
                        <div class="s-eval">{eval_text}</div>
                    </div>
                    <div class="s-card-bottom">
                        <div class="s-price-box">
                            <span class="s-price">{int(row['현재가']):,}원</span>
                            <span class="s-ratio m-up">{sign}{rate:.2f}%</span>
                        </div>
                        <div class="s-score">AI {row['스코어']:.1f}점</div>
                    </div>
                    {mini_chart_svg}
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("데이터 수집 중입니다.")
        
    st.markdown("""<marquee class="s-marquee" scrollamount="8">⚠️ [투자 유의사항] 본 방송은 데이터 제공용이며 투자를 권유하지 않습니다. 최종 책임은 투자자 본인에게 있습니다.</marquee>""", unsafe_allow_html=True)

# =============================================================================
# 💻 분기점 2: 기존 메인 대시보드 화면 (데스크톱 와이드 모드)
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
        filtered_df = filtered_df.sort_values(by='거래대금', ascending=False).head(15)

        @st.cache_resource
        def load_lstm_assets():
            # 💡 [안전장치] 무료 서버 메모리 폭파를 막기 위해 함수 안에서만 몰래 불러옵니다.
            try:
                import tensorflow as tf
                if "stock_lstm_model.h5" not in os.listdir() or "lstm_scaler.pkl" not in os.listdir(): return None, None
                return tf.keras.models.load_model("stock_lstm_model.h5", compile=False), joblib.load("lstm_scaler.pkl")
            except: return None, None

        lstm_model, lstm_scaler = load_lstm_assets()
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
            st.warning("⚠️ 텐서플로우 메모리 절약 모드 작동: 기본 예측 수식으로 안전하게 대체합니다.")
            filtered_df['10분_상승예측(%)'] = ((filtered_df['등락률'] * 0.5) + np.log1p(filtered_df['거래대금'])).round(2)

        filtered_df['테마'] = filtered_df['종목명'].apply(get_theme_icon)
        filtered_df['단기_목표가'] = (filtered_df['현재가'] * 1.03).astype(int)
        filtered_df['손절가'] = (filtered_df['현재가'] * 0.98).astype(int)

        def detect_signal(row):
            if row['등락률'] >= 7.0 and row['거래대금'] > 50000: return "🔥 돌파매매"
            elif 1.0 <= row['등락률'] < 5.0 and row['거래대금'] > 20000: return "💧 눌림목"
            return "▪️ 관망"
            
        filtered_df['매매상태'] = filtered_df.apply(detect_signal, axis=1)
        top_30 = filtered_df.sort_values(by='10분_상승예측(%)', ascending=False)
        
        output_dict = {
            '테마': top_30['테마'], '실시간 상태': top_30['매매상태'], 'AI 예측스코어': top_30['10분_상승예측(%)'].apply(lambda x: f"🚀 {float(x):.2f}점"), 
            '종목명': top_30['종목명'], '전일 종가(현재가)': top_30['현재가'].apply(lambda x: f"{int(x):,} 원"), '전일 상승률': top_30['등락률'].apply(lambda x: f"+{x:.2f} %"),
            '단기 목표가(+3%)': top_30['단기_목표가'].apply(lambda x: f"{x:,} 원"), '손절가(-2%)': top_30['손절가'].apply(lambda x: f"{x:,} 원"),
            '거래대금(백만)': top_30['거래대금'].apply(lambda x: f"{int(x):,}"), '종목코드': top_30['종목코드']
        }
        
        output_df = pd.DataFrame(output_dict).reset_index(drop=True)
        selected_rows = st.dataframe(output_df, use_container_width=True, selection_mode="single-row", on_select="rerun")
    else:
        st.error("데이터를 불러오지 못했습니다.")
        output_df = pd.DataFrame()

    # -----------------------------------------------------------------------------
    # 데스크톱 하단 개별 1분봉 분석 차트 영역
    # -----------------------------------------------------------------------------
    st.markdown("---")
    selected_idx = selected_rows.selection.rows[0] if (hasattr(selected_rows, 'selection') and len(selected_rows.selection.rows) > 0) else 0

    if not output_df.empty and selected_idx < len(output_df):
        target_code, target_name, target_theme, target_price, target_change, target_vol = output_df.iloc[selected_idx]['종목코드'], output_df.iloc[selected_idx]['종목명'], output_df.iloc[selected_idx]['테마'], output_df.iloc[selected_idx]['전일 종가(현재가)'], output_df.iloc[selected_idx]['전일 상승률'], output_df.iloc[selected_idx]['거래대금(백만)']
        
        st.markdown(f"<div style='padding:10px 0; border-bottom:1px solid #ddd; margin-bottom:15px;'><span style='font-size:20px; font-weight:bold;'>{target_name}</span> <span style='font-size:14px; color:#555;'>[{target_theme}]</span><span style='font-size:14px; font-weight:bold; margin-left:15px;'>{target_price}</span><span style='font-size:14px; color:#e12929; margin-left:5px;'>{target_change}</span><span style='font-size:14px; color:#888; margin-left:10px;'>누적 거래대금 {target_vol}백만</span></div>", unsafe_allow_html=True)
        
        with st.spinner(f"[{target_name}] 1분봉 데이터 및 위험성 진단 중..."):
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
                        
                        if c_p < df_min['MA5'].iloc[-1] and c_p <= h_10m * 0.97: st.error(f"💣 **[🚨 보안관 비상경보]** **{target_name}** 종목은 5분선이 붕괴되어 급락 위험이 큽니다. 신규 진입 금지!")
                        elif c_p >= h_10m * 0.98 and not (df_min['MA5'].iloc[-1] > df_min['MA20'].iloc[-1] and df_min['MA5'].iloc[-2] <= df_min['MA20'].iloc[-2]): st.warning(f"⚠️ **[추격매수 경고]** **{target_name}** 종목은 가짜 돌파에 걸릴 확률이 높으니 관망하십시오.")
                        elif df_min['MA5'].iloc[-1] > df_min['MA20'].iloc[-1] and df_min['MA5'].iloc[-2] <= df_min['MA20'].iloc[-2] and df_min['Volume'].iloc[-1] > df_min['Vol_MA5'].iloc[-1] * 1.5 and c_p < h_10m * 0.96: st.success(f"🚀 **[정석 무릎자리]** **{target_name}** 정배열 초입 돌파가 확인된 타점입니다.")
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

    # -----------------------------------------------------------------------------
    # 🌟 [추가 기능] 하단 돌파매매 Top 10 미니 차트 갤러리 (데스크톱 전용)
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔥 실시간 돌파매매 Top 10 차트 갤러리")

    if not output_df.empty:
        # '돌파매매' 상태인 종목을 최우선으로, 부족하면 전체 순위 순으로 10개 채우기
        breakout_df = output_df[output_df['실시간 상태'].str.contains("돌파", na=False)]
        rest_df = output_df[~output_df['종목코드'].isin(breakout_df['종목코드'])]
        gallery_df = pd.concat([breakout_df, rest_df]).head(10)

        cols = st.columns(5) # 5개씩 가로로 배치 (자동으로 2줄 형성)
        with st.spinner("돌파매매 상위 10개 종목의 실시간 틱 차트를 불러오는 중입니다..."):
            for i, (idx, row) in enumerate(gallery_df.iterrows()):
                col = cols[i % 5]
                t_code, t_name, t_rate, t_state = row['종목코드'], row['종목명'], row['전일 상승률'], row['실시간 상태']
                
                with col:
                    # 각 종목 카드 UI
                    st.markdown(f"""
                        <div style="background-color:rgba(30,41,59,0.5); padding:10px; border-radius:10px; border:1px solid #334155; margin-bottom:5px;">
                            <div style="font-size:16px; font-weight:bold; color:white;">{t_name}</div>
                            <div style="font-size:13px; color:{'#ef4444' if '+' in t_rate else '#3b82f6'};">{t_rate} <span style="font-size:11px; color:#facc15; margin-left:5px;">{t_state}</span></div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
                    headers = get_common_headers("FHKST03010200")
                    params = {"FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": t_code, "FID_INPUT_HOUR_1": datetime.now(KST).strftime("%H%M%S"), "FID_PW_DATA_INCU_YN": "N"}
                    
                    try:
                        res = requests.get(url, headers=headers, params=params, timeout=3).json()
                        if res.get('rt_cd') == '0' and 'output2' in res:
                            min_data = res['output2'][:30][::-1] # 최근 30분 데이터
                            if len(min_data) > 2:
                                prices = [float(m['stck_prpr']) for m in min_data]
                                times = [m['stck_cntg_hour'] for m in min_data]
                                color = "#ef4444" if prices[-1] >= prices[0] else "#3b82f6"
                                
                                # 미니 스파크라인 차트 생성
                                fig = go.Figure(go.Scatter(x=times, y=prices, mode='lines', line=dict(color=color, width=2.5)))
                                fig.update_layout(
                                    margin=dict(l=0, r=0, t=5, b=5),
                                    height=80,
                                    xaxis=dict(visible=False),
                                    yaxis=dict(visible=False),
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    hovermode=False
                                )
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                            else:
                                st.caption("데이터 부족")
                    except:
                        st.caption("차트 로드 실패")
                        
                    time.sleep(0.05) # 한국투자증권 API 초당 20건 제한 회피 (안전장치)
