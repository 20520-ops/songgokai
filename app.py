import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="교통사고 사망자 수 예측 및 안전 가이드",
    page_icon="🚦",
    layout="wide"
)

# 1. 모델 로드 함수 (캐싱 적용)
@st.cache_resource
def load_orange_model(model_path="traffic_casualty_model.pkcls"):
    if not os.path.exists(model_path):
        st.error(f"모델 파일 '{model_path}'을(를) 찾을 수 없습니다. 경로를 확인해 주세요.")
        st.stop()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_orange_model()

# 메인 타이틀
st.title("🚦 교통사고 예상 사망자 수 실시간 예측 시스템")
st.caption("Orange3 예측 모델 기반 실시간 위험도 모니터링 및 어린이 보호구역 안전 수칙 카드")

st.markdown("---")

# 레이아웃 분할: 좌측 입력 폼, 우측 실시간 결과
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("⚙️ 조건 입력")
    
    # 2. 사용자 입력 조절 (시도 선택 및 사고건수, 경상자수 슬라이더)
    sido = st.selectbox(
        "📍 시/도 지역 선택",
        options=["서울", "경기", "부산", "기타 지역"],
        help="학습 모델 특성에 맞게 지역 선택"
    )
    
    accidents = st.slider("🚗 사고건수", min_value=0, max_value=500, value=50, step=1)
    minor_injured = st.slider("🩹 경상자수", min_value=0, max_value=500, value=30, step=1)
    serious_injured = st.slider("🏥 중상자수 (추가 변수)", min_value=0, max_value=200, value=5, step=1)
    reported_injured = st.slider("📋 부상신고자수 (추가 변수)", min_value=0, max_value=100, value=2, step=1)

    # 입력 데이터를 모델의 피처 형식으로 변환
    # 모델 입력 차원: [시도=경기, 시도=부산, 시도=서울, 사고건수, 중상자수, 경상자수, 부상신고자수]
    is_gyeonggi = 1.0 if sido == "경기" else 0.0
    is_busan = 1.0 if sido == "부산" else 0.0
    is_seoul = 1.0 if sido == "서울" else 0.0

    input_data = np.array([[
        is_gyeonggi,
        is_busan,
        is_seoul,
        float(accidents),
        float(serious_injured),
        float(minor_injured),
        float(reported_injured)
    ]])

    # 모델 예측 수행
    try:
        # Orange3 모델 개체에 따라 predict 또는 __call__ 지원
        if hasattr(model, "predict"):
            pred_raw = model.predict(input_data)
        else:
            pred_raw = model(input_data)
        
        # 예측값 정제 (음수 방지)
        predicted_deaths = float(np.ravel(pred_raw)[0])
        predicted_deaths = max(0.0, predicted_deaths)
    except Exception as e:
        # scikit-learn inner model fallback
        if hasattr(model, "skl_model"):
            pred_raw = model.skl_model.predict(input_data)
            predicted_deaths = max(0.0, float(pred_raw[0]))
        else:
            st.error(f"예측 수행 중 오류 발생: {e}")
            predicted_deaths = 0.0

with col2:
    st.subheader("📊 실시간 예측 결과")
    
    # 예측 사망자 수 표시
    st.metric(
        label="예상 사망자 수",
        value=f"{predicted_deaths:.2f} 명"
    )
    
    # 3. 위험도 등급 판정 및 신호등 UI
    if predicted_deaths < 0.5:
        status_color = "green"
        status_icon = "🟢"
        status_text = "안전 (Green)"
        bg_color = "#e8f5e9"
        border_color = "#2e7d32"
    elif predicted_deaths < 1.5:
        status_color = "orange"
        status_icon = "🟡"
        status_text = "주의 (Yellow)"
        bg_color = "#fffde7"
        border_color = "#fbc02d"
    else:
        status_color = "red"
        status_icon = "🔴"
        status_text = "위험 (Red)"
        bg_color = "#ffebee"
        border_color = "#c62828"

    # 대형 신호등 카드 UI
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            border: 3px solid {border_color};
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
        ">
            <h1 style="font-size: 70px; margin: 0;">{status_icon}</h1>
            <h2 style="color: {border_color}; margin: 10px 0 0 0;">위험도 등급: {status_text}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# 하단: 어린이 보호구역 안전 수칙 카드
st.markdown("---")
st.subheader("🚸 어린이 보호구역(스쿨존) 안전 수칙 카드")

card_col1, card_col2, card_col3 = st.columns(3)

with card_col1:
    st.markdown(
        """
        <div style="background-color: #f0f4f8; padding: 18px; border-radius: 10px; border-left: 5px solid #1976d2;">
            <h4>🚗 운전자 준수 수칙</h4>
            <ul>
                <li>시속 <b>30km/h 이하</b> 서행 운행</li>
                <li>횡단보도 앞 <b>일시정지</b> (신호등 유무 무관)</li>
                <li>주·정차 절대 금지 (불법 주정차 단속 대상)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with card_col2:
    st.markdown(
        """
        <div style="background-color: #fff9c4; padding: 18px; border-radius: 10px; border-left: 5px solid #fbc02d;">
            <h4>🎒 어린이 지도 수칙</h4>
            <ul>
                <li>길을 건널 땐 <b>우선 멈추고 좌우 살펴보기</b></li>
                <li>횡단보도에서는 <b>손을 들고 건너기</b></li>
                <li>차도 근처에서 공놀이나 장난치지 않기</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with card_col3:
    st.markdown(
        """
        <div style="background-color: #e8ecef; padding: 18px; border-radius: 10px; border-left: 5px solid #455a64;">
            <h4>⚙️ 모니터링 행동 지침</h4>
            <ul>
                <li>🟡 <b>주의 단계</b>: 보행 신호 준수 및 서행 가이드 강화</li>
                <li>🔴 <b>위험 단계</b>: 교통경찰·안전요원 배치 및 집중 단속 필요</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
