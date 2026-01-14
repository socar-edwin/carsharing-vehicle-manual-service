"""
Vehicle Manual Chatbot with LLM

차종별 매뉴얼 기반 RAG 챗봇 + Help Center 일반 문의 지원
"""
from openai import OpenAI
from typing import List, Dict, Any, Optional
from src.vehicle_retriever import VehicleRetriever, format_context_for_llm
import logging
import re

logger = logging.getLogger(__name__)

# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 쏘카(SOCAR) 카셰어링 서비스의 전문 상담사입니다.

## 역할
- 고객이 선택한 차종에 대한 질문에 친절하고 정확하게 답변합니다.
- 차량 외 일반 서비스 문의(예약, 결제, 취소, 보험, 사고 등)에도 도움을 드립니다.
- **반드시 제공된 문서 내용을 기반으로 답변합니다.**

## 답변 지침
1. **제공된 문서의 내용을 직접 인용하거나 요약하여 답변합니다.**
2. 핵심 내용을 먼저 간결하게 설명합니다.
3. 단계별 안내가 필요한 경우 번호를 매겨 설명합니다.
4. 안전 관련 주의사항이 있으면 반드시 언급합니다.
5. 문서에 없는 내용은 추측하지 말고 "자세한 내용은 쏘카 고객센터(1661-3315)로 문의해 주세요"라고 안내합니다.
6. 친근하고 이해하기 쉬운 말투를 사용합니다.

## 정보 우선순위
- **차량별 구체적인 정보**(주유구 위치, 버튼 위치, 조작법 등)를 **가장 우선**합니다.
- "쏘카 차량 공통 안내"는 차량별 정보가 없을 때만 참조합니다.
- 예: "주유구 위치"를 물으면 "오른쪽/왼쪽" 등 구체적 위치를 먼저 답하고, 주유카드 안내는 부가 정보로만 제공합니다.

## 제약사항
- **제공된 컨텍스트에 없는 정보는 답변하지 않습니다.**
- 다른 차종의 정보를 혼동하지 않습니다.
- 일반적인 상식이 아닌 쏘카 서비스 관련 정보는 반드시 문서를 참조합니다.
"""

# 일반 문의 키워드 (Help Center 검색 트리거)
GENERAL_INQUIRY_KEYWORDS = [
    '예약', '취소', '결제', '환불', '크레딧', '쿠폰', '포인트',
    '보험', '사고', '과태료', '범칙금', '면허', '자격',
    '가격', '요금', '비용', '수수료', '주행료', '휴차보상',
    '회원', '가입', '탈퇴', '계정', '인증',
    '앱', '어플', '로그인', '비밀번호',
    '고객센터', '연락처', '문의', '신고',
    '반납', '연장', '지연', '패널티',
    '존', '주차', '반납장소',
    # 사고/손상 관련 구어체
    '반파', '파손', '부서', '박살', '망가', '깨졌', '찌그러', '긁혔', '찍혔',
    '펑크', '타이어', '접촉', '충돌', '추돌', '사이드미러', '범퍼',
]

# 구어체 → 공식 키워드 매핑 (Help Center 문서 검색용) - fallback용
KEYWORD_MAPPING = {
    '반파': '사고', '파손': '사고', '부서': '사고', '박살': '사고', '망가': '사고',
    '깨졌': '사고', '찌그러': '사고', '긁혔': '사고', '찍혔': '사고',
    '접촉': '사고', '충돌': '사고', '추돌': '사고',
    '펑크': '사고', '타이어': '사고',
    '사이드미러': '사고', '범퍼': '사고',
}

# 쿼리 분석 프롬프트
QUERY_ANALYSIS_PROMPT = """사용자의 질문을 분석하여 다음을 JSON으로 반환하세요:

1. is_service_inquiry: 차량 조작이 아닌 서비스 관련 문의인지 (true/false)
2. keywords: 검색에 사용할 공식 키워드 리스트 (구어체를 공식 용어로 변환)
3. is_action_query: "어떻게 해야 하나요?" 형태의 방법/대처 질문인지 (true/false)

서비스 문의 예시: 예약, 취소, 결제, 환불, 사고, 보험, 과태료, 반납, 크레딧, 요금 등
차량 조작 예시: 시동, 주유, 충전, 블루투스, 트렁크, 에어컨, 네비게이션 등

예시:
- "차가 박살났어 어떻게 해?" → {{"is_service_inquiry": true, "keywords": ["사고"], "is_action_query": true}}
- "블루투스 연결 방법" → {{"is_service_inquiry": false, "keywords": [], "is_action_query": true}}
- "예약 취소하고 싶어요" → {{"is_service_inquiry": true, "keywords": ["예약", "취소"], "is_action_query": false}}

사용자 질문: {query}

JSON만 반환하세요:"""

# 출처 선택 프롬프트
# 텍스트 하이라이트용 핵심 키워드 (차량 관련)
HIGHLIGHT_KEYWORDS = [
    '시동', '주유', '충전', '블루투스', '트렁크', '에어컨', '히터',
    '사이드미러', '미러', '핸들', '기어', '브레이크', '파킹',
    '네비게이션', '오디오', '하이패스', '스마트키', '열선',
    '와이퍼', '전조등', '라이트', '시트', '안전벨트', '에어백',
    'USB', '무선충전', '주유구', '충전구', '보닛', '본넷'
]


def extract_highlight_keyword(query: str) -> str:
    """
    질문에서 텍스트 하이라이트용 핵심 키워드 추출

    Args:
        query: 사용자 질문

    Returns:
        하이라이트할 키워드 (없으면 빈 문자열)
    """
    query_lower = query.lower()

    # 우선순위: HIGHLIGHT_KEYWORDS에서 매칭
    for keyword in HIGHLIGHT_KEYWORDS:
        if keyword in query_lower:
            return keyword

    # Fallback: 질문에서 명사 추출 (간단한 휴리스틱)
    # "~은", "~는", "~이", "~가" 앞의 단어, 또는 "~ 방법", "~ 위치" 앞의 단어
    import re
    patterns = [
        r'(\w+)\s*(?:은|는|이|가)\s*(?:어떻게|어디|뭐)',
        r'(\w+)\s*(?:방법|위치|하는\s*법)',
        r'(\w+)\s*(?:열기|닫기|걸기|끄기|켜기)',
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            keyword = match.group(1)
            if len(keyword) >= 2:  # 최소 2글자
                return keyword

    return ""


def extract_heading_from_body(body: str, keyword: str) -> str:
    """
    body에서 키워드와 관련된 heading 텍스트 추출

    Heading 패턴:
    - "▶ 주유구는 어떻게 여나요?"
    - "[주요 기능 안내]"
    - "1. 전자식 사이드 브레이크"

    Args:
        body: 매뉴얼 본문 텍스트
        keyword: 검색할 키워드

    Returns:
        매칭된 heading 텍스트 (없으면 빈 문자열)
    """
    if not body or not keyword:
        return ""

    import re

    # Heading 패턴들
    heading_patterns = [
        r'▶\s*([^\n▶\[\]]{5,50})',           # ▶ 시동은 어떻게 거나요?
        r'\[([^\[\]]{3,30})\]',               # [주요 기능 안내]
        r'(\d+\.\s*[^\n\d]{3,40})',           # 1. 전자식 사이드 브레이크
        r'([가-힣]+\s*(?:안내|방법|사용)[^\n]{0,20})',  # 충전 안내, 블루투스 연결 방법
    ]

    keyword_lower = keyword.lower()
    best_match = ""
    best_score = 0

    for pattern in heading_patterns:
        matches = re.findall(pattern, body)
        for match in matches:
            match_lower = match.lower().strip()
            # 키워드가 heading에 포함되어 있는지 확인
            if keyword_lower in match_lower:
                # 더 짧은 매칭이 더 정확함 (너무 긴 건 제외)
                score = 1 / (len(match) + 1)
                if score > best_score and len(match) < 50:
                    best_match = match.strip()
                    best_score = score

    return best_match


def find_heading_id_by_keyword(heading_map: Dict[str, str], keyword: str) -> Optional[str]:
    """
    heading_map에서 키워드와 매칭되는 heading ID 찾기
    더 구체적인 heading을 우선 (예: "주유구 개폐" > "쏘카 차량 공통 안내(주유)")

    Args:
        heading_map: heading 텍스트 -> heading ID 매핑
        keyword: 검색할 키워드

    Returns:
        매칭된 heading ID (없으면 None)
    """
    if not heading_map or not keyword:
        return None

    keyword_lower = keyword.lower()

    # 구체적인 키워드 우선 매핑 (예: "주유" 검색 시 "주유구"가 포함된 heading 우선)
    specific_keywords = {
        '주유': ['주유구'],
        '충전': ['충전구'],
    }

    # 낮은 우선순위 패턴 (공통 안내 등)
    low_priority_patterns = ['공통 안내', '공통안내', '쏘카 전용']

    def score_heading(heading_text: str, search_keyword: str) -> int:
        """heading의 매칭 점수 계산 (높을수록 좋음)"""
        text_lower = heading_text.lower()
        score = 0

        # 키워드 포함 여부 기본 점수
        if search_keyword.lower() in text_lower:
            score = 100

        # 구체적인 키워드 포함 시 보너스 (예: "주유구" 포함)
        if search_keyword.lower() in specific_keywords:
            for specific_kw in specific_keywords[search_keyword.lower()]:
                if specific_kw in text_lower:
                    score += 50

        # 낮은 우선순위 패턴 포함 시 감점
        for pattern in low_priority_patterns:
            if pattern in text_lower:
                score -= 30

        # 짧은 heading 선호 (더 구체적일 가능성)
        if score > 0:
            score -= len(heading_text) // 10

        return score

    # 모든 매칭되는 heading 수집 및 점수 계산
    candidates = []
    for heading_text, heading_id in heading_map.items():
        if keyword_lower in heading_text.lower():
            score = score_heading(heading_text, keyword)
            candidates.append((heading_text, heading_id, score))

    # 점수 순으로 정렬하여 최고 점수 반환
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_match = candidates[0]
        logger.info(f"Found heading ID for '{keyword}': {best_match[0]} -> #{best_match[1]} (score: {best_match[2]})")
        return best_match[1]

    # 2차: 유사 키워드 매칭 (주유 -> 주유구, 충전 -> 충전구 등)
    related_keywords = {
        '주유': ['주유구', '주유는', '주유 방법'],
        '충전': ['충전구', '충전은', '충전 방법', '충전 안내'],
        '시동': ['시동은', '시동 걸기', '시동 방법'],
        '블루투스': ['블루투스 연결', '블루투스 설정'],
        '트렁크': ['트렁크 사용', '트렁크 열기'],
        '에어컨': ['에어컨 사용', '냉방'],
        '히터': ['히터 사용', '난방'],
        '미러': ['사이드 미러', '미러 조절'],
        '기어': ['기어 조절', '기어 변속'],
        '브레이크': ['파킹 브레이크', '사이드 브레이크', '전자식'],
    }

    search_keywords = []
    for main_kw, related in related_keywords.items():
        if keyword_lower == main_kw or keyword_lower in main_kw:
            search_keywords.extend(related)

    for search_kw in search_keywords:
        candidates = []
        for heading_text, heading_id in heading_map.items():
            if search_kw.lower() in heading_text.lower():
                score = score_heading(heading_text, search_kw)
                candidates.append((heading_text, heading_id, score))

        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_match = candidates[0]
            logger.info(f"Found heading ID for '{keyword}' (related: {search_kw}): {best_match[0]} -> #{best_match[1]} (score: {best_match[2]})")
            return best_match[1]

    return None


def add_text_highlight_to_url(
    url: str,
    keyword: str,
    body: str = "",
    heading_map: Optional[Dict[str, str]] = None
) -> str:
    """
    URL에 heading ID 또는 텍스트 하이라이트 fragment 추가

    우선순위:
    1. heading_map에서 heading ID 찾기 (Zendesk anchor)
    2. body에서 키워드 관련 heading 텍스트 찾기 (Text Fragment)
    3. 키워드만 사용 (Text Fragment)

    Args:
        url: 원본 URL
        keyword: 하이라이트할 키워드
        body: 매뉴얼 본문 (heading 추출용, fallback)
        heading_map: heading 텍스트 -> heading ID 매핑 (우선)

    Returns:
        heading ID 또는 텍스트 하이라이트가 추가된 URL
    """
    if not url or not keyword:
        return url

    from urllib.parse import quote

    # 기존 fragment 제거
    if '#' in url:
        url = url.split('#')[0]

    # 1. heading_map에서 heading ID 찾기 (최우선)
    if heading_map:
        heading_id = find_heading_id_by_keyword(heading_map, keyword)
        if heading_id:
            return f"{url}#{heading_id}"

    # 2. heading ID 없으면 텍스트 하이라이트 사용 (fallback)
    highlight_text = keyword
    if body:
        heading = extract_heading_from_body(body, keyword)
        if heading:
            highlight_text = heading
            logger.info(f"Found heading text for '{keyword}': '{heading}'")

    encoded_text = quote(highlight_text)
    return f"{url}#:~:text={encoded_text}"


SOURCE_SELECTION_PROMPT = """사용자 질문에 가장 직접적으로 답변하는 문서를 선택하세요.

사용자 질문: {query}
핵심 키워드: {keywords}

후보 문서:
{candidates}

선택 기준 (우선순위):
1. "어떻게 해야 하나요?" / "어떻게 하나요?" 형태의 대처법/절차 문서 최우선
2. 사고/손상 관련 질문 → "사고를 당하면" 또는 "사고 발생" 문서 우선 (면책/보상 문서 X)
3. 제목이 질문과 가장 유사한 문서

중요: Q&A, 안내, 면책제도 등 정보성 문서보다 직접적인 대처법 문서를 우선하세요.

가장 적합한 문서 번호만 반환하세요 (예: 1, 2, 3...). 적합한 문서가 없으면 0을 반환하세요."""


class VehicleChatbot:
    """
    차종별 매뉴얼 RAG 챗봇 + Help Center 일반 문의 지원
    """

    def __init__(
        self,
        retriever: VehicleRetriever,
        help_center_retriever: Optional[Any] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.3
    ):
        """
        Initialize VehicleChatbot

        Args:
            retriever: VehicleRetriever 인스턴스 (차종 매뉴얼)
            help_center_retriever: HybridRetriever 인스턴스 (일반 문의, 선택)
            api_key: OpenAI API 키 (없으면 환경변수 사용)
            model: OpenAI 모델명
            max_tokens: 최대 토큰 수
            temperature: 생성 온도 (0~1)
        """
        self.retriever = retriever
        self.help_center_retriever = help_center_retriever
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # OpenAI 클라이언트 초기화
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # 환경변수에서 API 키 사용

        logger.info(f"VehicleChatbot initialized")
        logger.info(f"  - Model: {model}")
        logger.info(f"  - Max tokens: {max_tokens}")
        logger.info(f"  - Help Center: {'Enabled' if help_center_retriever else 'Disabled'}")

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """LLM을 사용하여 쿼리 의도 분석"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                temperature=0,
                messages=[
                    {"role": "user", "content": QUERY_ANALYSIS_PROMPT.format(query=query)}
                ]
            )
            result_text = response.choices[0].message.content.strip()
            # JSON 파싱
            import json
            # ```json ... ``` 형식 처리
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            analysis = json.loads(result_text)
            logger.info(f"Query analysis: {analysis}")
            return analysis
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}, falling back to keyword matching")
            return None

    def _is_general_inquiry(self, query: str, analysis: Dict = None) -> bool:
        """일반 문의 여부 판단 (LLM 분석 결과 우선 사용)"""
        if analysis and 'is_service_inquiry' in analysis:
            return analysis['is_service_inquiry']
        # Fallback: 키워드 매칭
        query_lower = query.lower()
        for keyword in GENERAL_INQUIRY_KEYWORDS:
            if keyword in query_lower:
                return True
        return False

    def _select_best_source(self, query: str, candidates: List[Dict], analysis: Dict = None) -> Optional[Dict]:
        """LLM을 사용하여 가장 적절한 출처 선택"""
        if not candidates:
            return None

        # 키워드 기반 사전 필터링
        keywords = analysis.get('keywords', []) if analysis else []
        if keywords:
            # 키워드가 제목에 포함된 문서 우선
            keyword_matched = []
            others = []
            for doc in candidates:
                title = doc.get('title', '').lower()
                if any(kw in title for kw in keywords):
                    keyword_matched.append(doc)
                else:
                    others.append(doc)
            # 키워드 매칭된 문서를 앞에 배치
            candidates = keyword_matched + others

        # 후보 문서 포맷팅 (최대 7개)
        candidate_texts = []
        for i, doc in enumerate(candidates[:7], 1):
            title = doc.get('title', '')
            candidate_texts.append(f"{i}. {title}")

        candidates_str = "\n".join(candidate_texts)

        # 키워드 추출
        keywords = []
        if analysis and analysis.get('keywords'):
            keywords = analysis['keywords']
        keywords_str = ", ".join(keywords) if keywords else "없음"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=10,
                temperature=0,
                messages=[
                    {"role": "user", "content": SOURCE_SELECTION_PROMPT.format(
                        query=query,
                        keywords=keywords_str,
                        candidates=candidates_str
                    )}
                ]
            )
            result_text = response.choices[0].message.content.strip()
            logger.info(f"LLM source selection response: {result_text}")
            selected_idx = int(result_text) - 1  # 0-indexed

            if 0 <= selected_idx < len(candidates[:7]):
                logger.info(f"LLM selected source: {candidates[selected_idx].get('title', '')}")
                return candidates[selected_idx]
            else:
                # LLM이 0을 반환하거나 범위 밖이면 첫 번째 문서 사용
                logger.info(f"LLM returned {result_text}, using first candidate as fallback")
                return candidates[0] if candidates else None
        except Exception as e:
            logger.warning(f"Source selection failed: {e}")
            return candidates[0] if candidates else None

    def _format_help_center_results(self, results: List[Dict], query: str, analysis: Dict = None) -> str:
        """Help Center 검색 결과 포맷팅 (LLM 분석 결과 우선 사용)"""
        if not results:
            return ""

        # LLM 분석 결과에서 키워드 추출 (우선)
        if analysis and analysis.get('keywords'):
            query_keywords = analysis['keywords']
            is_action_query = analysis.get('is_action_query', False)
        else:
            # Fallback: 기존 키워드 매칭
            query_keywords = []
            for kw in GENERAL_INQUIRY_KEYWORDS:
                if kw in query.lower():
                    query_keywords.append(kw)
                    if kw in KEYWORD_MAPPING:
                        mapped_kw = KEYWORD_MAPPING[kw]
                        if mapped_kw not in query_keywords:
                            query_keywords.append(mapped_kw)
            action_words = ['대처', '어떻게', '방법', '해야', '하면']
            is_action_query = any(w in query.lower() for w in action_words)

        # 키워드가 포함된 결과만 필터링
        keyword_matched_results = []
        for result in results:
            title = result.get('title', '').lower()
            body = result.get('body', '').lower()
            keyword_match = any(kw in title or kw in body for kw in query_keywords)
            if keyword_match:
                keyword_matched_results.append(result)

        if not keyword_matched_results:
            return ""

        # 액션 쿼리인 경우 "어떻게"가 제목에 있는 문서 우선
        if is_action_query:
            for result in keyword_matched_results:
                title = result.get('title', '').lower()
                if any(w in title for w in ['어떻게', '방법']):
                    return f"[관련 도움말] {result.get('title', '')}\n{result.get('body', '')[:500]}"

        # 기본: 첫 번째 키워드 매칭 결과 반환
        result = keyword_matched_results[0]
        return f"[관련 도움말] {result.get('title', '')}\n{result.get('body', '')[:500]}"

    def chat(
        self,
        query: str,
        vehicle_name: str,
        top_k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        차종별 질문 응답

        Args:
            query: 사용자 질문
            vehicle_name: 선택된 차종
            top_k: 검색 결과 수
            include_sources: 출처 포함 여부

        Returns:
            응답 딕셔너리 (answer, sources, vehicle_name)
        """
        logger.info(f"Chat request - Vehicle: {vehicle_name}, Query: {query}")

        # 0. LLM으로 쿼리 의도 분석 (Help Center가 활성화된 경우만)
        analysis = None
        if self.help_center_retriever:
            analysis = self._analyze_query(query)

        # 1. 차종 매뉴얼 로드 (쿼리 기반 관련 청크 우선)
        # 차종당 매뉴얼이 1개이므로, 해당 차종의 청크를 가져오되 관련 청크 우선
        vehicle_chunks = self.retriever.get_vehicle_document(
            vehicle_name=vehicle_name,
            query=query,  # 사용자 질문으로 관련 청크 우선 로드
            max_chunks=15  # 충분한 컨텍스트 확보
        )

        # 2. 일반 문의인 경우 Help Center도 검색
        help_center_results = []
        help_center_context = ""
        is_general = self._is_general_inquiry(query, analysis)

        if is_general and self.help_center_retriever:
            logger.info(f"General inquiry detected, searching Help Center...")
            try:
                search_response = self.help_center_retriever.search(query, top_k=10)
                help_center_results = search_response.get('results', [])
                help_center_context = self._format_help_center_results(help_center_results, query, analysis)
                logger.info(f"Found {len(help_center_results)} Help Center results")
            except Exception as e:
                logger.warning(f"Help Center search failed: {e}")

        # 3. 컨텍스트 구성 (전체 문서 사용, score 필터링 제거)
        vehicle_context = format_context_for_llm(vehicle_chunks) if vehicle_chunks else ""

        # 4. 검색 결과 없음 처리 (차종 매뉴얼도 없고 도움말도 없는 경우)
        if not vehicle_chunks and not help_center_context:
            return {
                'answer': f"죄송합니다. '{query}'에 대한 관련 정보를 찾을 수 없습니다. 다른 질문을 해주시거나, 쏘카 고객센터(1661-3315)로 문의해 주세요.",
                'sources': [],
                'help_center_sources': [],
                'vehicle_name': vehicle_name
            }

        # 5. 프롬프트 구성
        user_prompt = f"""## 차종
{vehicle_name}
"""
        # 관련 차량 매뉴얼이 있을 때만 포함
        if vehicle_context:
            user_prompt += f"""
## 차량 매뉴얼 정보
{vehicle_context}
"""
        # 서비스 도움말이 있을 때만 포함
        if help_center_context:
            user_prompt += f"""
## 쏘카 서비스 도움말
{help_center_context}
"""
        user_prompt += f"""
## 고객 질문
{query}

## 답변
위 정보를 참고하여 고객의 질문에 답변해 주세요. 차량 조작 관련 질문이면 매뉴얼을, 서비스/사고/예약 관련 질문이면 도움말을 참고하세요. 관련 정보가 없으면 솔직히 모른다고 답변하세요."""

        # 4. LLM 호출
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )

            answer = response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.\n\n오류: {str(e)}"

        # 7. 출처 정보 구성
        sources = []
        help_center_sources = []

        # 텍스트 하이라이트용 키워드 추출
        highlight_keyword = extract_highlight_keyword(query)
        logger.info(f"Highlight keyword: '{highlight_keyword}'")

        if include_sources:
            # 차종 매뉴얼 출처 (전체 문서에서 첫 번째 URL 사용 + heading ID)
            seen_urls = set()

            # body 텍스트 수집 (heading 추출용 - fallback)
            full_body = "\n".join([chunk.get('text', '') for chunk in vehicle_chunks])

            # heading_map 추출 (첫 번째 chunk에서)
            heading_map = None
            if vehicle_chunks:
                heading_map_str = vehicle_chunks[0].get('heading_map', '')
                if heading_map_str:
                    try:
                        import json
                        heading_map = json.loads(heading_map_str)
                    except (json.JSONDecodeError, TypeError):
                        pass

            for chunk in vehicle_chunks:
                url = chunk.get('html_url', '')
                if url and url not in seen_urls:
                    # heading ID 추가 (없으면 텍스트 하이라이트)
                    highlighted_url = add_text_highlight_to_url(
                        url, highlight_keyword, full_body, heading_map
                    )
                    sources.append({
                        'section': chunk.get('section', ''),
                        'url': highlighted_url,
                        'score': 1.0
                    })
                    seen_urls.add(url)  # 원본 URL로 중복 체크
                    break  # 차종 매뉴얼은 1개이므로 첫 번째만

            # Help Center 출처 (LLM이 가장 적절한 문서 선택 + heading ID)
            if help_center_results:
                # 중복 URL 제외
                unique_results = [r for r in help_center_results if r.get('html_url', '') not in seen_urls]

                # LLM이 가장 적절한 출처 선택
                selected_result = self._select_best_source(query, unique_results, analysis)

                if selected_result:
                    hc_url = selected_result.get('html_url', '')
                    hc_body = selected_result.get('body', '')

                    # heading_map 추출
                    hc_heading_map = None
                    hc_heading_map_str = selected_result.get('heading_map', '')
                    if hc_heading_map_str:
                        try:
                            import json
                            hc_heading_map = json.loads(hc_heading_map_str)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # heading ID 추가 (analysis에서 키워드 사용)
                    hc_keyword = analysis.get('keywords', [None])[0] if analysis else highlight_keyword
                    highlighted_hc_url = add_text_highlight_to_url(
                        hc_url, hc_keyword or highlight_keyword, hc_body, hc_heading_map
                    )

                    help_center_sources.append({
                        'title': selected_result.get('title', ''),
                        'url': highlighted_hc_url,
                        'category': selected_result.get('category_name', ''),
                        'score': selected_result.get('relevance_score', 0)
                    })
                    seen_urls.add(hc_url)

        return {
            'answer': answer,
            'sources': sources,
            'help_center_sources': help_center_sources,
            'vehicle_name': vehicle_name,
            'search_results_count': len(vehicle_chunks),
            'is_general_inquiry': is_general
        }

    def chat_stream(
        self,
        query: str,
        vehicle_name: str,
        top_k: int = 5
    ):
        """
        스트리밍 응답 생성

        Args:
            query: 사용자 질문
            vehicle_name: 선택된 차종
            top_k: 검색 결과 수

        Yields:
            응답 텍스트 청크
        """
        logger.info(f"Stream chat - Vehicle: {vehicle_name}, Query: {query}")

        # 1. 관련 문서 검색
        search_results = self.retriever.retrieve(
            query=query,
            vehicle_name=vehicle_name,
            top_k=top_k
        )

        if not search_results:
            yield f"죄송합니다. '{vehicle_name}' 차량에 대한 관련 정보를 찾을 수 없습니다."
            return

        # 2. 컨텍스트 구성
        context = format_context_for_llm(search_results)

        user_prompt = f"""## 차종
{vehicle_name}

## 매뉴얼 정보
{context}

## 고객 질문
{query}

## 답변
위 매뉴얼 정보를 참고하여 고객의 질문에 답변해 주세요."""

        # 3. 스트리밍 LLM 호출
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"\n\n죄송합니다. 오류가 발생했습니다: {str(e)}"


if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from config import PROJECT_ROOT, OPENAI_API_KEY
    from src.vehicle_indexing import load_vehicle_index
    from src.vehicle_retriever import VehicleRetriever

    # 경로 설정
    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "vehicle_manuals_chroma"

    print("=" * 60)
    print("Testing Vehicle Chatbot")
    print("=" * 60)

    # 인덱스 로드
    try:
        index = load_vehicle_index(persist_dir=VECTORSTORE_DIR)
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Please run vehicle_indexing.py first to create the index.")
        sys.exit(1)

    # Retriever 생성
    retriever = VehicleRetriever(index=index, default_top_k=5)

    # Chatbot 생성
    chatbot = VehicleChatbot(
        retriever=retriever,
        api_key=OPENAI_API_KEY
    )

    # 테스트
    test_cases = [
        ("아이오닉 5", "충전은 어떻게 하나요?"),
        ("캐스퍼", "시동 거는 방법 알려주세요"),
        ("K5", "블루투스 연결 방법"),
    ]

    for vehicle, query in test_cases:
        print(f"\n{'='*60}")
        print(f"Vehicle: {vehicle}")
        print(f"Query: {query}")
        print("=" * 60)

        result = chatbot.chat(query=query, vehicle_name=vehicle)

        print(f"\n답변:\n{result['answer']}")

        if result['sources']:
            print(f"\n출처:")
            for src in result['sources']:
                print(f"  - [{src['section']}] {src['url']}")
