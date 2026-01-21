"""
FAQ 답변 검증 스크립트 v2
- E열: Zendesk에 정보가 없는 경우 N으로 변경
- F열: AI 답변이 실제 매뉴얼 Fact와 일치하는지 (Hallucination 체크)
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from openai import OpenAI
from config import PROJECT_ROOT, OPENAI_API_KEY
from src.vehicle_indexing import load_vehicle_index
from src.vehicle_retriever import VehicleRetriever

# OpenAI 클라이언트
client = OpenAI(api_key=OPENAI_API_KEY)

# "정보 없음" 패턴 (Zendesk에 해당 내용이 없는 경우)
NO_INFO_PATTERNS = [
    r'정보[가는]?\s*(제공된\s*)?문서에\s*포함되어\s*있지\s*않',
    r'제공된\s*(정보|문서)에[는]?\s*(해당\s*)?(내용이\s*)?포함되어\s*있지\s*않',
    r'관련\s*정보를\s*찾을\s*수\s*없',
    r'정확한\s*답변을\s*드리기\s*어렵',
    r'정보가\s*없',
    r'문서에\s*포함되어\s*있지\s*않',
    r'내용이\s*포함되어\s*있지\s*않',
    r'정보는\s*제공된\s*문서에',
]

# 차종 추출 매핑 (faq_test.py에서 복사)
VEHICLE_KEYWORDS = {
    '니로EV': '니로 EV', '니로 EV': '니로 EV',
    '아이오닉9': '아이오닉 9', '아이오닉 9': '아이오닉 9',
    '아이오닉6': '아이오닉 6', '아이오닉 6': '아이오닉 6',
    '아이오닉5': '아이오닉 5', '아이오닉 5': '아이오닉 5',
    'EV6': 'EV6', 'EV9': 'EV9',
    'K8': 'K8', 'K5': 'K5', 'K7': 'K7',
    'GV80': 'GV80', 'GV70': 'GV70', 'G80': 'G80',
    '더 뉴 기아 레이 루프탑': '더 뉴 기아 레이 루프탑',
    '더 뉴 기아 레이': '더 뉴 기아 레이 루프탑',
    '기아 레이': '레이', '레이 루프탑': '더 뉴 기아 레이 루프탑',
    '더 뉴 셀토스': '셀토스', '셀토스': '셀토스',
    '더 뉴 아반떼': '더 뉴 아반떼', '아반떼': '더 뉴 아반떼',
    '더 뉴 코나': '더 뉴 코나', '코나': '더 뉴 코나',
    '더 뉴 팰리세이드': '디 올 뉴 팰리세이드', '팰리세이드': '디 올 뉴 팰리세이드',
    '스타리아 캠퍼4': '스타리아 라운지 캠퍼 4', '스타리아 캠퍼 4': '스타리아 라운지 캠퍼 4',
    '스타리아': '스타리아',
    '더 뉴 쏘렌토': '더 뉴 쏘렌토', '쏘렌토': '더 뉴 쏘렌토',
    '더 뉴 카니발': '더 뉴 카니발', '카니발': '더 뉴 카니발',
    '더 뉴 캐스퍼': '더 뉴 캐스퍼', '캐스퍼': '더 뉴 캐스퍼',
}
DEFAULT_VEHICLE = 'K5'
DEFAULT_EV_VEHICLE = '아이오닉 5'
EV_KEYWORDS = ['충전', '전기', 'EV', '배터리', 'V2L', 'V2H']


def extract_vehicle_from_question(question: str) -> str:
    """질문에서 차종 추출"""
    sorted_keywords = sorted(VEHICLE_KEYWORDS.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in question:
            return VEHICLE_KEYWORDS[keyword]
    for ev_kw in EV_KEYWORDS:
        if ev_kw in question:
            return DEFAULT_EV_VEHICLE
    return DEFAULT_VEHICLE


def has_no_info(answer: str) -> bool:
    """AI 답변에서 '정보 없음' 패턴 감지"""
    if not answer:
        return True
    for pattern in NO_INFO_PATTERNS:
        if re.search(pattern, answer):
            return True
    return False


def check_hallucination(question: str, ai_answer: str, manual_context: str) -> str:
    """
    AI 답변이 매뉴얼 Fact와 일치하는지 확인 (Hallucination 체크)

    Returns:
        Y: 매뉴얼과 일치 (정상)
        N: 매뉴얼에 없는 정보 또는 잘못된 정보 (Hallucination)
    """
    if not manual_context or not ai_answer:
        return ''

    prompt = f"""AI 답변이 매뉴얼 내용과 일치하는지 판단하세요.

## 질문
{question}

## 매뉴얼 원문 (실제 Fact)
{manual_context[:3000]}

## AI 답변
{ai_answer}

## 판단 기준
1. AI 답변의 핵심 정보(위치, 방법, 절차, 버튼명 등)가 매뉴얼에 있으면 → Y
2. AI가 매뉴얼에 없는 정보를 지어냈으면 → N (Hallucination)
3. AI가 매뉴얼 정보를 잘못 해석했으면 → N
4. AI가 "정보 없음"이라고 정직하게 답했으면 → Y (거짓말 안 함)
5. 세부 표현 차이는 허용 (의미만 맞으면 Y)

Y 또는 N 중 하나만 답하세요:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip().upper()
        return 'Y' if 'Y' in result else 'N'
    except Exception as e:
        print(f"  → LLM 체크 오류: {e}")
        return ''


def main():
    FAQ_PATH = PROJECT_ROOT / "data" / "FAQ_260121.xlsx"
    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "vehicle_manuals_chroma"

    print("=" * 60)
    print("FAQ Hallucination 검증 시작")
    print("=" * 60)

    # FAQ 로드
    df = pd.read_excel(FAQ_PATH)
    print(f"총 {len(df)}개 질문 로드")

    # Retriever 초기화
    print("매뉴얼 인덱스 로드 중...")
    index = load_vehicle_index(persist_dir=VECTORSTORE_DIR)
    retriever = VehicleRetriever(index=index, default_top_k=10)
    print("인덱스 로드 완료\n")

    # E열, F열 초기화
    zendesk_flags = []
    hallucination_flags = []

    for idx, row in df.iterrows():
        question = row['질문']
        ai_answer = row['AI 답변 내용']

        print(f"[{idx+1}/{len(df)}] {question[:40]}...")

        # 1. E열: Zendesk 정보 유무 판단
        if has_no_info(str(ai_answer)):
            zendesk_flag = 'N'
            print(f"  → Zendesk 정보: N (정보 없음 패턴 감지)")
        else:
            zendesk_flag = 'Y'
            print(f"  → Zendesk 정보: Y")

        zendesk_flags.append(zendesk_flag)

        # 2. F열: Hallucination 체크 (E=Y인 경우만)
        if zendesk_flag == 'Y':
            # 해당 차종의 매뉴얼 컨텍스트 가져오기
            vehicle = extract_vehicle_from_question(question)
            chunks = retriever.get_vehicle_document(
                vehicle_name=vehicle,
                query=question,
                max_chunks=10
            )

            if chunks:
                # 매뉴얼 텍스트 결합
                manual_context = "\n\n".join([
                    f"[{c.get('section', '')}]\n{c.get('text', '')}"
                    for c in chunks
                ])

                hallucination_flag = check_hallucination(question, ai_answer, manual_context)
                print(f"  → Fact 일치: {hallucination_flag}")
            else:
                hallucination_flag = ''
                print(f"  → Fact 일치: (매뉴얼 컨텍스트 없음)")
        else:
            hallucination_flag = ''
            print(f"  → Fact 일치: (Zendesk 정보 없음으로 스킵)")

        hallucination_flags.append(hallucination_flag)

    # 결과 저장
    df['Zendesk 포함 여부'] = zendesk_flags
    df['정상 답변 여부'] = hallucination_flags

    df.to_excel(FAQ_PATH, index=False)

    # 통계 출력
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)
    print(f"E열 - Zendesk 정보 있음 (Y): {zendesk_flags.count('Y')}개")
    print(f"E열 - Zendesk 정보 없음 (N): {zendesk_flags.count('N')}개")
    print(f"F열 - Fact 일치 (Y): {hallucination_flags.count('Y')}개")
    print(f"F열 - Hallucination (N): {hallucination_flags.count('N')}개")
    print(f"F열 - 체크 불가: {hallucination_flags.count('')}개")
    print(f"\n결과가 {FAQ_PATH}에 저장되었습니다.")


if __name__ == "__main__":
    main()
