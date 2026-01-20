"""
차량 매뉴얼 전용 챗봇 - Streamlit UI

URL 파라미터로 car_class_id를 받아 해당 차종 매뉴얼 기반 질문 답변
예: ?car_class_id=695 → GV70 매뉴얼로 답변

인증: Google OAuth (@socar.kr 도메인 제한)
"""
import streamlit as st
import pandas as pd
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 이미지 경로
SOCAR_SYMBOL = "img/Socar_Symbol_RGB.png"
SOCAR_SIGNATURE = "img/Socar_Signature_WhiteBG_RGB.png"

# 페이지 설정
st.set_page_config(
    page_title="차량 매뉴얼 챗봇",
    page_icon=SOCAR_SYMBOL,
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    /* ========================================
       Streamlit 기본 UI 숨김 (웹/모바일, 라이트/다크 모드 대응)
       사이드바 토글 버튼만 유지
       ======================================== */

    /* 헤더 내 모든 버튼 숨기기 */
    header[data-testid="stHeader"] button {
        visibility: hidden !important;
    }

    /* 사이드바 토글 버튼만 보이기 */
    header[data-testid="stHeader"] [data-testid="baseButton-header"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] button {
        visibility: visible !important;
    }

    /* 햄버거 메뉴 숨김 */
    #MainMenu {
        visibility: hidden !important;
    }

    /* Deploy 버튼 숨김 */
    .stAppDeployButton {
        visibility: hidden !important;
    }

    /* 푸터 숨김 */
    footer {
        visibility: hidden !important;
    }

    /* 상단 데코레이션 라인 제거 */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* 헤더 배경 투명 */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ========================================
       앱 스타일
       ======================================== */

    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #0078FF;
        margin-bottom: 1rem;
    }
    .vehicle-info {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0078FF;
    }
    .source-link {
        font-size: 0.85rem;
        color: #666;
    }
    .stChatMessage {
        padding: 1rem;
    }
    /* 자주 묻는 질문 칩 스타일 */
    .suggestion-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
        justify-content: center;
    }
    .suggestion-label {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    /* 모바일에서 columns 가로 유지 */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
    }
    /* Streamlit 버튼을 칩처럼 보이게 */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
        color: #333;
        transition: all 0.2s;
        white-space: nowrap;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background-color: #e9ecef;
        border-color: #0078FF;
        color: #0078FF;
    }
    /* 모바일 버튼 크기 조정 */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] .stButton > button {
            padding: 0.3rem 0.7rem;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chatbot():
    """차량 매뉴얼 전용 챗봇 인스턴스 로드 (캐시)"""
    from config import PROJECT_ROOT, OPENAI_API_KEY
    from src.vehicle_indexing import load_vehicle_index
    from src.vehicle_retriever import VehicleRetriever
    from src.vehicle_chatbot import VehicleChatbot

    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "vehicle_manuals_chroma"

    try:
        # 차종 매뉴얼 인덱스 로드
        index = load_vehicle_index(persist_dir=VECTORSTORE_DIR)
        retriever = VehicleRetriever(index=index, default_top_k=5)

        # Chatbot 생성 (Help Center 없이)
        chatbot = VehicleChatbot(
            retriever=retriever,
            help_center_retriever=None,  # 차량 매뉴얼만 사용
            api_key=OPENAI_API_KEY
        )

        return chatbot, True
    except Exception as e:
        logger.error(f"Failed to load chatbot: {e}")
        return None, False


@st.cache_data
def load_car_class_mapping():
    """car_class_id → manual_vehicle_name 매핑 테이블 로드"""
    from config import PROJECT_ROOT

    MAPPING_PATH = PROJECT_ROOT / "data" / "car_class_manual_mapping.csv"

    try:
        df = pd.read_csv(MAPPING_PATH)
        # car_class_id를 key로 하는 딕셔너리 생성
        mapping = {}
        for _, row in df.iterrows():
            car_class_id = int(row['car_class_id'])
            mapping[car_class_id] = {
                'car_name': row['car_name'],
                'manual_vehicle_name': row['manual_vehicle_name'],
                'match_type': row['match_type'],
                'vehicle_type': row.get('vehicle_type', '')
            }
        logger.info(f"Loaded {len(mapping)} car_class mappings")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load car_class mapping: {e}")
        return {}


def get_vehicle_from_car_class_id(car_class_id: int, mapping: dict) -> tuple:
    """
    car_class_id로 매뉴얼 차종명 조회

    Returns:
        (manual_vehicle_name, car_name, vehicle_type, found)
    """
    if car_class_id in mapping:
        info = mapping[car_class_id]
        return info['manual_vehicle_name'], info['car_name'], info.get('vehicle_type', ''), True
    return None, None, None, False


@st.cache_data
def get_vehicle_data():
    """차종 데이터 로드 (유형별 분류)"""
    from config import PROJECT_ROOT
    from src.vehicle_data_loader import load_vehicle_data, get_vehicle_list

    DATA_PATH = PROJECT_ROOT / "data" / "vehicle_manual_data.csv"

    try:
        df = load_vehicle_data(DATA_PATH)
        vehicle_by_type = get_vehicle_list(df)
        return vehicle_by_type
    except Exception as e:
        logger.error(f"Failed to load vehicle data: {e}")
        return {}


def _get_auth_config():
    """Get auth config from secrets or environment."""
    import os

    # Try st.secrets first (Streamlit Cloud)
    try:
        auth_enabled = st.secrets.get("AUTH_ENABLED", "true")
        allowed_domains = st.secrets.get("ALLOWED_EMAIL_DOMAINS", "socar.kr")
    except Exception:
        # Fallback to environment variables (local)
        auth_enabled = os.getenv("AUTH_ENABLED", "true")
        allowed_domains = os.getenv("ALLOWED_EMAIL_DOMAINS", "socar.kr")

    # Convert to proper types
    auth_enabled = str(auth_enabled).lower() == "true"
    allowed_domains = str(allowed_domains).split(",")

    return auth_enabled, allowed_domains


def main():
    # ========================================
    # 인증 체크 (AUTH_ENABLED=true일 때만)
    # ========================================
    AUTH_ENABLED, ALLOWED_EMAIL_DOMAINS = _get_auth_config()

    if AUTH_ENABLED:
        from src.auth import require_auth, render_user_info

        if not require_auth(ALLOWED_EMAIL_DOMAINS):
            st.stop()

    # ========================================
    # URL 파라미터에서 car_class_id 확인
    # ========================================
    query_params = st.query_params
    car_class_id_param = query_params.get("car_class_id")

    # 매핑 테이블 로드
    car_class_mapping = load_car_class_mapping()

    # car_class_id로 차종 결정
    selected_vehicle = None
    display_car_name = None  # 사용자에게 표시할 차종명 (원본)
    vehicle_type = None  # 차량 유형 (전기차, SUV 등)
    is_from_url = False

    if car_class_id_param:
        try:
            car_class_id = int(car_class_id_param)
            manual_name, car_name, v_type, found = get_vehicle_from_car_class_id(car_class_id, car_class_mapping)

            if found:
                selected_vehicle = manual_name
                display_car_name = car_name
                vehicle_type = v_type
                is_from_url = True
                logger.info(f"car_class_id={car_class_id} → {car_name} ({v_type}) → 매뉴얼: {manual_name}")
            else:
                st.warning(f"car_class_id={car_class_id}에 해당하는 차종을 찾을 수 없습니다.")
                logger.warning(f"Unknown car_class_id: {car_class_id}")
        except ValueError:
            st.error(f"잘못된 car_class_id 형식입니다: {car_class_id_param}")

    # ========================================
    # 사이드바 (파라미터 없을 때만 표시)
    # ========================================
    if not is_from_url:
        with st.sidebar:
            st.header("🚙 차종 선택")

            # 차종 데이터 로드
            vehicle_by_type = get_vehicle_data()

            if not vehicle_by_type:
                st.error("차종 데이터를 불러올 수 없습니다.")
                st.info("먼저 Vector Store 인덱싱을 실행해 주세요.")
                st.code("python build_index.py --target vehicle", language="bash")
                return

            # 차량 유형 선택
            vehicle_types = list(vehicle_by_type.keys())
            selected_type = st.selectbox(
                "차량 유형",
                options=vehicle_types,
                index=0,
                help="차량 유형을 선택하세요"
            )

            # 해당 유형의 차종 목록
            vehicles_in_type = vehicle_by_type.get(selected_type, [])

            # 차종 선택
            selected_vehicle = st.selectbox(
                "차종",
                options=vehicles_in_type,
                index=0 if vehicles_in_type else None,
                help="질문할 차종을 선택하세요"
            )
            display_car_name = selected_vehicle  # 수동 선택 시 동일
            vehicle_type = selected_type  # 선택한 유형이 곧 vehicle_type

            st.divider()

            # 선택된 차종 정보
            if selected_vehicle:
                st.markdown(f"""
                <div class="vehicle-info">
                    <strong>선택된 차종:</strong><br>
                    🚙 {selected_vehicle}<br>
                    📂 {selected_type}
                </div>
                """, unsafe_allow_html=True)

            # 사용자 정보 & 로그아웃 (인증 활성화 시)
            if AUTH_ENABLED:
                from src.auth import render_user_info
                render_user_info()

    # ========================================
    # 헤더 (차종명 반영) + 우측 상단 시그니처
    # ========================================
    # 시그니처를 CSS로 우측 상단 고정
    import base64
    with open(SOCAR_SIGNATURE, "rb") as f:
        sig_base64 = base64.b64encode(f.read()).decode()
    st.markdown(f'''
        <style>
            .socar-signature {{
                position: fixed;
                top: 2.5rem;
                right: 4rem;
                z-index: 999999;
            }}
            .socar-signature img {{
                height: 28px;
            }}
            @media (max-width: 768px) {{
                .socar-signature {{
                    top: 2rem;
                    right: 3rem;
                }}
                .socar-signature img {{
                    height: 22px;
                }}
            }}
        </style>
        <div class="socar-signature">
            <img src="data:image/png;base64,{sig_base64}" alt="SOCAR">
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="main-header">차량 매뉴얼 챗봇</div>', unsafe_allow_html=True)
    if display_car_name:
        st.markdown(f"**{display_car_name}**의 조작 방법에 대해 질문해 주세요!")
    else:
        st.markdown("쏘카 차량의 **조작 방법**에 대해 질문해 주세요!")

    # 메인 영역: 채팅 인터페이스
    if not selected_vehicle:
        st.warning("사이드바에서 차종을 선택해 주세요.")
        return

    # 챗봇 로드
    chatbot, is_loaded = load_chatbot()

    if not is_loaded:
        st.error("챗봇을 초기화할 수 없습니다.")
        st.info("Vector Store 인덱싱을 먼저 실행해 주세요:")
        st.code("python build_index.py --target vehicle", language="bash")
        return

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_vehicle" not in st.session_state:
        st.session_state.current_vehicle = None

    # 차종 변경 감지 → 대화 초기화
    if st.session_state.current_vehicle != selected_vehicle:
        st.session_state.messages = []
        st.session_state.current_vehicle = selected_vehicle

    # 아바타 설정
    USER_AVATAR = "👤"
    ASSISTANT_AVATAR = SOCAR_SYMBOL

    # 대화 기록 표시
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            # 차종 매뉴얼 출처
            if message.get("sources") and message["sources"][0].get('url'):
                with st.expander("📖 출처 보기"):
                    st.markdown(f"🔗 [{display_car_name} 이용 안내]({message['sources'][0]['url']})")

    # 채팅 입력 (먼저 받아서 suggestions 조건에 사용)
    user_input = st.chat_input(f"{display_car_name}에 대해 질문하세요...")

    # ========================================
    # 자주 묻는 질문 (첫 대화 전에만 표시)
    # ========================================
    # 메시지가 없고, pending_question도 없고, 직접 입력도 없을 때만 표시
    show_suggestions = (
        not st.session_state.messages
        and "pending_question" not in st.session_state
        and not user_input
    )

    if show_suggestions:
        st.markdown('<p class="suggestion-label">💡 이런 질문을 해보세요</p>', unsafe_allow_html=True)

        # 전기차 여부에 따라 다른 질문 표시
        is_ev = vehicle_type == "전기차"

        if is_ev:
            vehicle_questions = [
                "시동 거는 방법",
                "충전구 위치",
                "충전 방법",
                "블루투스 연결",
                "트렁크 열기",
                "주차 브레이크",
            ]
        else:
            vehicle_questions = [
                "시동 거는 방법",
                "주유구 위치",
                "기어 조작법",
                "블루투스 연결",
                "트렁크 열기",
                "주차 브레이크",
            ]

        # 3개씩 2줄로 배치
        cols = st.columns(3)
        for i, q in enumerate(vehicle_questions):
            with cols[i % 3]:
                if st.button(q, key=f"suggest_{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    # 예제 질문 처리 (버튼 클릭 시)
    if "pending_question" in st.session_state:
        user_input = st.session_state.pending_question
        del st.session_state.pending_question

    # 사용자 입력 처리
    if user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(user_input)

        # 어시스턴트 응답 생성
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("답변을 생성하고 있습니다..."):
                try:
                    # 챗봇 호출
                    result = chatbot.chat(
                        query=user_input,
                        vehicle_name=selected_vehicle,
                        top_k=5,
                        include_sources=True
                    )

                    answer = result["answer"]
                    sources = result.get("sources", [])

                    # 답변 표시
                    st.markdown(answer)

                    # 차종 매뉴얼 출처 표시
                    if sources and sources[0].get('url'):
                        with st.expander("📖 출처 보기"):
                            st.markdown(f"🔗 [{display_car_name} 이용 안내]({sources[0]['url']})")

                    # 세션에 저장
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Chat error: {e}")

    # 푸터
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
