"""
FAQ 질문에 대한 AI 답변 생성 스크립트
"""
import sys
import re
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from config import PROJECT_ROOT, OPENAI_API_KEY
from src.vehicle_indexing import load_vehicle_index
from src.vehicle_retriever import VehicleRetriever
from src.vehicle_chatbot import VehicleChatbot

logging.basicConfig(
    level=logging.WARNING,  # INFO 로그 숨김
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 질문에서 차종 추출하는 매핑
VEHICLE_KEYWORDS = {
    # 전기차
    '니로EV': '니로 EV',
    '니로 EV': '니로 EV',
    '아이오닉9': '아이오닉 9',
    '아이오닉 9': '아이오닉 9',
    '아이오닉6': '아이오닉 6',
    '아이오닉 6': '아이오닉 6',
    '아이오닉5': '아이오닉 5',
    '아이오닉 5': '아이오닉 5',
    'EV6': 'EV6',
    'EV9': 'EV9',
    '코나EV': '코나EV',
    '코나 EV': '코나EV',
    '볼트EV': '볼트 EV',
    '볼트 EV': '볼트 EV',
    '레이EV': '레이 EV',
    '레이 EV': '레이 EV',

    # 특정 차종
    'K8': 'K8',
    'K5': 'K5',
    'K7': 'K7',
    'GV80': 'GV80',
    'GV70': 'GV70',
    'G80': 'G80',

    # 한글 차종
    '더 뉴 기아 레이 루프탑': '더 뉴 기아 레이 루프탑',
    '더 뉴 기아 레이': '더 뉴 기아 레이 루프탑',  # 매칭되는 차종으로
    '기아 레이': '레이',
    '레이 루프탑': '더 뉴 기아 레이 루프탑',
    '더 뉴 셀토스': '셀토스',  # 더 뉴 셀토스 매뉴얼 없음, 셀토스로 매핑
    '셀토스': '셀토스',
    '더 뉴 아반떼': '더 뉴 아반떼',
    '아반떼': '더 뉴 아반떼',
    '더 뉴 코나': '더 뉴 코나',
    '코나': '더 뉴 코나',
    '더 뉴 팰리세이드': '디 올 뉴 팰리세이드',  # 매칭되는 차종
    '팰리세이드': '디 올 뉴 팰리세이드',
    '스타리아 캠퍼4': '스타리아 라운지 캠퍼 4',
    '스타리아 캠퍼 4': '스타리아 라운지 캠퍼 4',
    '스타리아': '스타리아',
    '더 뉴 쏘렌토': '더 뉴 쏘렌토',
    '쏘렌토': '더 뉴 쏘렌토',
    '더 뉴 카니발': '더 뉴 카니발',
    '카니발': '더 뉴 카니발',
    '더 뉴 캐스퍼': '더 뉴 캐스퍼',
    '캐스퍼': '더 뉴 캐스퍼',
    '그랜저': '디 올 뉴 그랜저',
    '싼타페': '디 올 뉴 싼타페',
}

# 기본 차종 (질문에 차종이 명시되지 않은 경우)
DEFAULT_VEHICLE = 'K5'  # 일반 질문용
DEFAULT_EV_VEHICLE = '아이오닉 5'  # 전기차 관련 일반 질문용

# 전기차 관련 키워드
EV_KEYWORDS = ['충전', '전기', 'EV', '배터리', 'V2L', 'V2H']


def extract_vehicle_from_question(question: str) -> str:
    """질문에서 차종 추출"""
    # 긴 키워드부터 먼저 매칭 시도 (더 정확한 매칭)
    sorted_keywords = sorted(VEHICLE_KEYWORDS.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in question:
            return VEHICLE_KEYWORDS[keyword]

    # 전기차 관련 질문인지 확인
    for ev_kw in EV_KEYWORDS:
        if ev_kw in question:
            return DEFAULT_EV_VEHICLE

    return DEFAULT_VEHICLE


def check_zendesk_source(sources: list) -> str:
    """출처에 Zendesk URL이 포함되어 있는지 확인"""
    if not sources:
        return 'N'

    for source in sources:
        url = source.get('url', '')
        if 'zendesk' in url.lower() or 'socar.zendesk' in url.lower():
            return 'Y'
    return 'N'


def main():
    # 경로 설정
    FAQ_PATH = PROJECT_ROOT / "data" / "FAQ_260121.xlsx"
    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "vehicle_manuals_chroma"

    print("=" * 60)
    print("FAQ AI 답변 생성 시작")
    print("=" * 60)

    # FAQ 로드
    df = pd.read_excel(FAQ_PATH)
    print(f"총 {len(df)}개 질문 로드")

    # 챗봇 초기화
    print("\n챗봇 초기화 중...")
    index = load_vehicle_index(persist_dir=VECTORSTORE_DIR)
    retriever = VehicleRetriever(index=index, default_top_k=5)
    chatbot = VehicleChatbot(
        retriever=retriever,
        help_center_retriever=None,
        api_key=OPENAI_API_KEY
    )
    print("챗봇 초기화 완료")

    # 결과 저장 리스트
    ai_answers = []
    zendesk_flags = []

    print("\n답변 생성 중...")
    for idx, row in df.iterrows():
        question = row['질문']
        original_answer = row['답변']

        # 차종 추출
        vehicle = extract_vehicle_from_question(question)

        print(f"\n[{idx+1}/{len(df)}] {question[:40]}...")
        print(f"  → 차종: {vehicle}")

        try:
            # 챗봇 호출
            result = chatbot.chat(
                query=question,
                vehicle_name=vehicle,
                top_k=5,
                include_sources=True
            )

            answer = result['answer']
            sources = result.get('sources', [])

            # Zendesk 출처 확인
            zendesk_flag = check_zendesk_source(sources)

            ai_answers.append(answer)
            zendesk_flags.append(zendesk_flag)

            # 미리보기
            print(f"  → AI 답변: {answer[:80]}...")
            print(f"  → Zendesk: {zendesk_flag}")

        except Exception as e:
            print(f"  → 오류: {e}")
            ai_answers.append(f"오류 발생: {e}")
            zendesk_flags.append('')

    # 결과를 DataFrame에 추가
    df['AI 답변 내용'] = ai_answers
    df['Zendesk 포함 여부'] = zendesk_flags
    # 정상 답변 여부는 수동 검토 필요

    # Excel 저장
    df.to_excel(FAQ_PATH, index=False)
    print(f"\n\n결과가 {FAQ_PATH}에 저장되었습니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
