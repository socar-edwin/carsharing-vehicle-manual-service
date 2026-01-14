# 차량 정보 RAG 챗봇 설계 제안서

> 카셰어링 서비스 차량별 이용 정보를 제공하는 RAG 기반 챗봇 시스템

**작성일**: 2026년 1월 8일
**목적**: Zendesk 고객 지원을 위한 차량별 특수 이용 정보 제공
**구현**: Streamlit 기반 웹 애플리케이션

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [기술 스택](#기술-스택)
4. [핵심 RAG 기술 적용](#핵심-rag-기술-적용)
5. [데이터 구조](#데이터-구조)
6. [구현 단계](#구현-단계)
7. [코드 구조](#코드-구조)
8. [프로젝트 타임라인](#프로젝트-타임라인)
9. [예상 성과](#예상-성과)

---

## 프로젝트 개요

### 문제 정의
카셰어링 서비스 이용자가 차량별 특수한 이용 정보(주유구 버튼, 사이드브레이크, 트렁크 개폐 등)를 찾는 데 어려움을 겪고 있음

### 솔루션
차량별로 구조화된 문서를 RAG 시스템에 입력하고, 사용자가 차종을 선택한 후 자연어로 질문하면 정확한 답변을 제공하는 챗봇

### 핵심 기능
1. **차종 선택 인터페이스**: Streamlit 첫 페이지에서 차종 선택 → 챗봇 페이지 이동
2. **FAQ 빠른 질문**: 채팅 입력창 상단에 차종별 자주 묻는 질문 3개 표시, 클릭 시 즉시 질문
3. **자연어 질의응답**: "주유구는 어디 있어?", "사이드브레이크 어떻게 풀어?" 등
4. **정확한 검색**: Hybrid search (키워드 + 의미 검색)
5. **출처 표시**: 답변의 신뢰성을 위한 문서 출처 표시
6. **대화 컨텍스트 유지**: 이전 질문 기억

---

## 시스템 아키텍처

### UI/UX Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Page 1: 차종 선택 페이지                                   │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  🚗 차량 이용 가이드 챗봇                            │  │ │
│  │  │                                                       │  │ │
│  │  │  이용하실 차종을 선택해주세요:                        │  │ │
│  │  │                                                       │  │ │
│  │  │  [ 아이오닉5 선택 ]                                   │  │ │
│  │  │  [ 코나일렉트릭 선택 ]                                │  │ │
│  │  │  [ 니로EV 선택 ]                                      │  │ │
│  │  │  [ 테슬라모델3 선택 ]                                 │  │ │
│  │  │  ...                                                  │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               ↓ (차종 선택 시)                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Page 2: 챗봇 페이지 (선택된 차종: 아이오닉5)            │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  [← 차종 변경]       🚗 아이오닉5 이용 가이드        │  │ │
│  │  ├──────────────────────────────────────────────────────┤  │ │
│  │  │  💬 자주 묻는 질문:                                  │  │ │
│  │  │  [주유구는 어디에 있나요?]                           │  │ │
│  │  │  [사이드브레이크는 어떻게 푸나요?]                   │  │ │
│  │  │  [트렁크는 어떻게 여나요?]                           │  │ │
│  │  ├──────────────────────────────────────────────────────┤  │ │
│  │  │  Assistant: 안녕하세요! 아이오닉5 이용에...         │  │ │
│  │  │  User: 주유구는 어디에 있나요?                       │  │ │
│  │  │  Assistant: 주유구 버튼은 운전석 좌측...            │  │ │
│  │  ├──────────────────────────────────────────────────────┤  │ │
│  │  │  💬 궁금하신 점을 질문해주세요...  [전송]           │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Pipeline Layer                             │
│                                                                   │
│  1. Query Processing                                             │
│     └─→ User Question + Selected Car Model                       │
│                                                                   │
│  2. Metadata Filtering                                           │
│     └─→ Filter by car_model metadata                             │
│                                                                   │
│  3. Hybrid Retrieval                                             │
│     ├─→ BM25 (Keyword Search)                                    │
│     ├─→ Vector Search (Semantic)                                 │
│     └─→ Reciprocal Rank Fusion                                   │
│                                                                   │
│  4. Reranking                                                    │
│     └─→ Top 3-5 most relevant chunks                             │
│                                                                   │
│  5. Generation                                                   │
│     └─→ LLM generates answer with citations                      │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Data Layer                                     │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Raw Docs   │ →  │   Chunked    │ →  │   Vector DB  │      │
│  │  (Markdown)  │    │   + Headers  │    │ (ChromaDB/   │      │
│  │              │    │              │    │  Qdrant)     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                   │
│  차종별 문서:                                                     │
│  - 아이오닉5/주유_충전.md                                         │
│  - 아이오닉5/주차브레이크.md                                      │
│  - 코나/주유_충전.md                                              │
│  - ...                                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

#### Flow 1: 차종 선택 및 페이지 이동
```
사용자가 Page 1 접속
           ↓
차종 리스트 표시 (아이오닉5, 코나, 니로EV...)
           ↓
사용자가 "아이오닉5" 선택
           ↓
Session state에 car_model = "아이오닉5" 저장
           ↓
Page 2 (챗봇 페이지)로 자동 이동
           ↓
해당 차종의 FAQ 3개 표시
```

#### Flow 2: FAQ 클릭을 통한 질문
```
사용자가 FAQ 버튼 클릭: "주유구는 어디에 있나요?"
           ↓
FAQ 질문이 자동으로 chat_input에 입력됨
           ↓
[아래 일반 질문 Flow와 동일하게 진행]
```

#### Flow 3: 일반 질문 및 답변 생성
```
사용자 입력: "주유구 버튼은 어디에 있어?"
선택 차종: "아이오닉5" (Session state에서 가져옴)
           ↓
[Query Processing]
- 쿼리 정규화 및 전처리
           ↓
[Metadata Filtering]
- car_model == "아이오닉5" 필터 적용
           ↓
[Hybrid Retrieval]
- BM25: "주유구", "버튼", "위치" 키워드 매칭
- Vector: 의미적으로 유사한 청크 검색
- RRF: 두 결과 융합
           ↓
[Reranking]
- Cross-encoder로 상위 3-5개 청크 선별
           ↓
[Generation]
- LLM에 선별된 청크 + 쿼리 전달
- 답변 생성 + 출처 표시
           ↓
[Output]
"아이오닉5의 주유구 버튼은 운전석 좌측 하단에 위치합니다.
버튼을 누르면 '딸깍' 소리와 함께 주유구 커버가 열립니다.

출처: 아이오닉5 > 주유_충전.md"
           ↓
대화 히스토리에 추가 → 맥락 유지
```

---

## 기술 스택

### 1. 핵심 프레임워크

#### A. LlamaIndex (데이터 인덱싱 및 검색)
**선택 이유**:
- 문서 중심 RAG에 최적화
- 차량별 문서 관리에 이상적
- 간단한 API, 빠른 프로토타입
- 2025년 검색 정확도 35% 향상

**사용 영역**:
- 문서 인제스션
- 인덱싱 및 저장
- 메타데이터 필터링
- 기본 쿼리 엔진

#### B. LangChain (오케스트레이션 - 선택적)
**선택 이유**:
- 복잡한 질의 처리
- 대화 히스토리 관리
- 멀티 스텝 추론

**사용 영역** (Phase 2 이후):
- 대화 메모리 관리
- 복잡한 쿼리 분해
- 에이전트 기반 검색

#### C. Streamlit (UI)
**선택 이유**:
- 빠른 프로토타입
- Python 네이티브
- 채팅 UI 기본 제공 (st.chat_input, st.chat_message)

### 2. Vector Store

#### 초기: ChromaDB
**장점**:
- 로컬 실행, 설치 간단
- 프로토타입에 최적
- 메타데이터 필터링 지원

**단점**:
- 대규모 데이터 처리 제한
- 프로덕션 스케일링 어려움

#### 프로덕션: Qdrant
**장점**:
- 뛰어난 메타데이터 필터링 성능
- 스케일링 우수
- Hybrid search 기본 지원
- 한국어 검색 최적화

**마이그레이션 시점**: Phase 3 (프로덕션 준비)

### 3. Embedding Model

#### 옵션 1: OpenAI text-embedding-3-large (권장)
**장점**:
- 최고 수준의 정확도
- 한국어 지원 우수
- API 기반, 관리 편의성

**단점**:
- API 비용 발생
- 외부 의존성

**비용 예상**:
- 1,000 문서 (차량 50종 × 20 문서) 인덱싱: ~$0.05
- 월 10,000 쿼리: ~$0.10
- **총 월 비용: ~$0.15** (매우 저렴)

#### 옵션 2: Nomic Embed v1.5 (비용 절감)
**장점**:
- 오픈소스, 무료
- 로컬 실행 가능
- 다국어 지원

**단점**:
- OpenAI보다 낮은 정확도
- 로컬 GPU 필요 (CPU는 느림)

#### 권장: OpenAI text-embedding-3-large
- 비용이 매우 저렴하고 정확도가 높음
- 한국어 지원 우수

### 4. LLM (생성 모델)

#### 옵션 1: Claude 3.5 Sonnet (강력 권장)
**장점**:
- 뛰어난 한국어 이해 및 생성
- 안전성 및 환각(hallucination) 최소화
- 긴 컨텍스트 지원 (200K 토큰)

**비용**:
- Input: $3 / 1M 토큰
- Output: $15 / 1M 토큰
- 월 10,000 쿼리 예상 비용: ~$5-10

#### 옵션 2: GPT-4o
**장점**:
- 빠른 응답 속도
- 안정적인 API
- 좋은 한국어 지원

**비용**:
- Input: $2.5 / 1M 토큰
- Output: $10 / 1M 토큰

#### 권장: Claude 3.5 Sonnet
- 고객 지원용으로 안전성과 정확도가 중요
- 한국어 품질이 우수

### 5. 추가 도구

#### Reranker (Phase 2)
- **Cohere Rerank**: API 기반, 효과적
- **Cross-Encoder (로컬)**: ms-marco-MiniLM-L-12-v2

#### BM25 구현
- **Rank-BM25**: Python 라이브러리
- LlamaIndex BM25Retriever 내장

---

## 핵심 RAG 기술 적용

### 1. Hybrid Search (30% 성능 향상)

#### 작동 방식
```python
# 동일 쿼리에 대해 두 가지 검색 수행

# 1. BM25 (키워드 기반)
bm25_results = bm25_retriever.retrieve("주유구 버튼 위치")
# → "주유구", "버튼", "위치" 키워드 정확 매칭

# 2. Vector Search (의미 기반)
vector_results = vector_retriever.retrieve("주유구 버튼 위치")
# → "연료 주입구 열기", "기름 넣는 곳" 등 의미적 유사 매칭

# 3. Reciprocal Rank Fusion
final_results = fuse_results(bm25_results, vector_results)
```

#### 장점
- 정확한 키워드 매칭 + 의미 이해
- 다양한 표현 방식 커버
- 사용자가 "기름 넣는 곳"이라고 해도 "주유구" 문서 검색

### 2. Semantic Chunking + Contextual Headers

#### 청킹 전략
```markdown
# 원본 문서: 아이오닉5/주유_충전.md

## 주유구 열기
주유구 버튼은 운전석 좌측 하단에 위치합니다.
버튼을 누르면 '딸깍' 소리와 함께 주유구 커버가 열립니다.

## 충전 포트 위치
급속 충전 포트는 차량 앞쪽 우측에 있습니다.
완속 충전 포트는 차량 앞쪽 좌측에 있습니다.
```

#### 청킹 결과 (Contextual Headers 추가)
```
Chunk 1:
[차종: 아이오닉5] [카테고리: 주유/충전] [주제: 주유구 열기]
주유구 버튼은 운전석 좌측 하단에 위치합니다.
버튼을 누르면 '딸깍' 소리와 함께 주유구 커버가 열립니다.

Chunk 2:
[차종: 아이오닉5] [카테고리: 주유/충전] [주제: 충전 포트 위치]
급속 충전 포트는 차량 앞쪽 우측에 있습니다.
완속 충전 포트는 차량 앞쪽 좌측에 있습니다.
```

#### 장점
- 청크에 맥락 정보 포함 → 검색 정확도 15-25% 향상
- 메타데이터로 차종 필터링 가능
- 관련 주제끼리 그룹화

### 3. Metadata 필터링

#### 구조
```python
metadata = {
    "car_model": "아이오닉5",        # 차종
    "category": "주유/충전",          # 카테고리
    "section": "주유구 열기",         # 섹션
    "source_file": "주유_충전.md"     # 원본 파일
}
```

#### 검색 시 활용
```python
# 사용자가 "아이오닉5" 선택 → 해당 차종만 검색
query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            MetadataFilter(key="car_model", value="아이오닉5"),
        ]
    )
)
```

#### 장점
- 관련 없는 차량 정보 완전 제외
- 검색 속도 향상
- 정확도 극대화

### 4. Reranking (Phase 2)

#### 프로세스
```
Initial Retrieval (Hybrid):
- 10-20개 후보 청크 검색

Reranking (Cross-Encoder):
- 쿼리와 각 청크의 관련성 점수 재계산
- 상위 3-5개만 선별

LLM Generation:
- 고품질 청크만 사용 → 정확한 답변
```

#### 장점
- 최종 답변의 관련성 향상
- LLM 컨텍스트 비용 절감 (적은 청크 사용)

---

## 데이터 구조

### 디렉토리 구조

```
vehicle-info-chatbot/
├── data/
│   ├── raw/                          # 원본 마크다운 문서
│   │   ├── 아이오닉5/
│   │   │   ├── 주유_충전.md
│   │   │   ├── 주차브레이크.md
│   │   │   ├── 트렁크.md
│   │   │   ├── 와이퍼_라이트.md
│   │   │   └── 기타_기능.md
│   │   ├── 코나일렉트릭/
│   │   │   ├── 주유_충전.md
│   │   │   ├── 주차브레이크.md
│   │   │   └── ...
│   │   ├── 니로EV/
│   │   ├── 테슬라모델3/
│   │   └── ...
│   ├── processed/                    # 전처리된 청크 (캐시)
│   │   ├── chunks.json
│   │   └── metadata.json
│   └── faqs/                         # 차종별 FAQ 데이터
│       └── faq_config.json
│
├── vectorstore/                      # Vector DB 저장소
│   └── chroma_db/                    # ChromaDB (초기)
│       └── ...
│
├── src/
│   ├── data_loader.py               # 문서 로딩
│   ├── chunking.py                  # Semantic chunking
│   ├── indexing.py                  # LlamaIndex 인덱싱
│   ├── retrieval.py                 # Hybrid search
│   ├── reranking.py                 # Reranker (Phase 2)
│   ├── generation.py                # LLM 생성
│   └── faq_manager.py               # FAQ 관리
│
├── app.py                           # Streamlit 메인 앱
├── config.py                        # 설정 (API keys, 모델 등)
├── requirements.txt
└── README.md
```

### 문서 작성 가이드

#### 마크다운 템플릿
```markdown
# [차종명] - [카테고리]

## [기능 1]

### 위치/방법
구체적인 설명...

### 주의사항
- 주의사항 1
- 주의사항 2

### 관련 정보
추가 정보...

---

## [기능 2]
...
```

#### 예시: 아이오닉5/주유_충전.md
```markdown
# 아이오닉5 - 주유 및 충전

## 주유구 열기

### 위치
주유구 버튼은 운전석 좌측 하단, 도어 열림 버튼 옆에 위치합니다.

### 사용 방법
1. 차량 시동을 끈 상태에서 버튼을 누릅니다
2. '딸깍' 소리와 함께 주유구 커버가 열립니다
3. 주유구 캡을 반시계 방향으로 돌려 엽니다

### 주의사항
- 주유 중에는 차량에서 내리지 마세요
- 정전기 방지를 위해 금속 부분을 먼저 터치하세요

---

## 급속 충전

### 포트 위치
급속 충전 포트는 차량 전면 우측(운전석 방향)에 위치합니다.

### 사용 방법
1. 충전기 커넥터를 포트에 삽입합니다
2. 충전이 시작되면 대시보드에 충전 상태가 표시됩니다
3. 충전 완료 후 커넥터 버튼을 눌러 분리합니다

### 충전 시간
- 10% → 80%: 약 18분 (350kW 급속충전기 기준)
```

### 메타데이터 자동 추출

```python
# 파일 경로: data/raw/아이오닉5/주유_충전.md
# 자동 추출되는 메타데이터:
{
    "car_model": "아이오닉5",           # 폴더명에서 추출
    "category": "주유/충전",            # 파일명에서 추출
    "source_file": "주유_충전.md",
    "last_updated": "2026-01-08"
}
```

### FAQ 데이터 구조

#### data/faqs/faq_config.json
```json
{
  "아이오닉5": [
    "주유구는 어디에 있나요?",
    "사이드브레이크는 어떻게 푸나요?",
    "트렁크는 어떻게 여나요?"
  ],
  "코나일렉트릭": [
    "충전 포트 위치가 어디인가요?",
    "주차 브레이크 해제 방법은?",
    "후방 카메라는 어떻게 켜나요?"
  ],
  "니로EV": [
    "급속 충전은 어떻게 하나요?",
    "와이퍼 작동 방법은?",
    "비상등은 어디 있나요?"
  ],
  "기본": [
    "차량 시동은 어떻게 거나요?",
    "에어컨 사용법을 알려주세요",
    "내비게이션 설정 방법은?"
  ]
}
```

**특징**:
- 차종별로 맞춤화된 FAQ 3개
- 실제 사용자가 가장 많이 문의하는 항목 기반
- "기본" 카테고리: 차종 정보가 없을 때 표시되는 기본 질문

---

## 구현 단계

### Phase 1: MVP (1-2주)
**목표**: 기본 RAG 챗봇 동작

**구현 항목**:
- [ ] Streamlit 2페이지 UI
  - Page 1: 차종 선택 페이지 (버튼 그리드)
  - Page 2: 챗봇 인터페이스 (FAQ + 채팅)
- [ ] FAQ 기능
  - 차종별 FAQ 3개 표시
  - 클릭 시 자동 질문 입력
- [ ] LlamaIndex 기본 인덱싱
  - Vector search only (Hybrid 제외)
  - 메타데이터 필터링
- [ ] ChromaDB vector store
- [ ] OpenAI embedding + Claude 3.5 Sonnet
- [ ] 3-5개 차종 샘플 데이터
- [ ] 기본 대화 UI

**기대 결과**:
- 직관적인 차종 선택 → 챗봇 페이지 전환
- FAQ 클릭으로 빠른 질문
- 관련 답변 생성 및 출처 표시

### Phase 2: 고도화 (2-3주)
**목표**: 검색 정확도 및 사용자 경험 향상

**구현 항목**:
- [ ] Hybrid Search 구현
  - BM25 + Vector search
  - Reciprocal Rank Fusion
- [ ] Reranker 추가
  - Cohere Rerank API
- [ ] Contextual Chunking
  - 의미 기반 청킹
  - 자동 헤더 생성
- [ ] 대화 히스토리 관리
  - 이전 질문 기억
  - 맥락 있는 대화
- [ ] 전체 차종 데이터 추가 (10-20종)
- [ ] 답변 품질 개선
  - 프롬프트 엔지니어링
  - Citation 포맷 개선

**기대 결과**:
- 30% 검색 정확도 향상 (Hybrid search)
- 자연스러운 대화 흐름
- 전체 차종 커버

### Phase 3: 프로덕션 준비 (2-3주)
**목표**: 실제 서비스 배포 가능 수준

**구현 항목**:
- [ ] Qdrant로 마이그레이션
  - 스케일링 대비
  - 성능 최적화
- [ ] 캐싱 레이어
  - 자주 묻는 질문 캐시
  - 응답 속도 개선
- [ ] 사용자 피드백 수집
  - 👍 / 👎 버튼
  - 피드백 데이터 저장
- [ ] 모니터링 및 로깅
  - 쿼리 로그
  - 성능 메트릭
- [ ] A/B 테스트 프레임워크
  - 프롬프트 변경 테스트
  - 검색 알고리즘 비교
- [ ] 평가 파이프라인
  - 자동 평가 데이터셋
  - 정확도, Recall, Precision 측정
- [ ] 보안 및 인증
  - 사용자 인증 (선택)
  - Rate limiting

**기대 결과**:
- 프로덕션 배포 가능
- 안정적인 성능
- 지속적 개선 프레임워크

### Phase 4: 고급 기능 (선택, 3-4주)
**목표**: 차별화된 사용자 경험

**구현 항목**:
- [ ] 멀티모달 지원
  - 이미지 업로드 (사용자가 찍은 차량 사진)
  - 이미지에서 부품 인식 → 해당 정보 제공
- [ ] 음성 인터페이스
  - STT (Speech-to-Text)
  - TTS (Text-to-Speech)
- [ ] Self-RAG 적용
  - 답변 품질 자체 평가
  - 필요시 재검색
- [ ] 다국어 지원
  - 영어, 중국어 등
- [ ] Zendesk 통합
  - API 연동
  - 티켓 자동 생성

---

## 코드 구조

### 1. app.py (Streamlit 메인)

```python
import streamlit as st
import json
from src.retrieval import HybridRetriever
from src.generation import generate_response
from config import AVAILABLE_CARS

# 페이지 설정
st.set_page_config(
    page_title="차량 이용 가이드 챗봇",
    page_icon="🚗",
    layout="centered"
)

# Session state 초기화
if "page" not in st.session_state:
    st.session_state.page = "selection"  # "selection" or "chat"
if "car_model" not in st.session_state:
    st.session_state.car_model = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# FAQ 로드
def load_faqs():
    """FAQ 설정 파일 로드"""
    try:
        with open("data/faqs/faq_config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # 기본 FAQ
        return {
            "기본": [
                "차량 시동은 어떻게 거나요?",
                "에어컨 사용법을 알려주세요",
                "내비게이션 설정 방법은?"
            ]
        }

faqs = load_faqs()

# ============================================================
# Page 1: 차종 선택 페이지
# ============================================================
def show_selection_page():
    st.title("🚗 차량 이용 가이드 챗봇")
    st.markdown("---")
    st.subheader("이용하실 차종을 선택해주세요")
    st.write("")

    # 차종 버튼을 2열 그리드로 표시
    cols = st.columns(2)

    for idx, car in enumerate(AVAILABLE_CARS):
        col = cols[idx % 2]
        with col:
            if st.button(
                f"🚙 {car}",
                key=f"car_{car}",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.car_model = car
                st.session_state.page = "chat"
                st.session_state.messages = []  # 새 차종 선택 시 대화 초기화
                st.rerun()

# ============================================================
# Page 2: 챗봇 페이지
# ============================================================
def show_chat_page():
    # 헤더 (차종 변경 버튼 포함)
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← 차종 변경"):
            st.session_state.page = "selection"
            st.rerun()
    with col2:
        st.title(f"🚗 {st.session_state.car_model} 이용 가이드")

    st.markdown("---")

    # FAQ 섹션
    st.markdown("### 💬 자주 묻는 질문")
    faq_questions = faqs.get(st.session_state.car_model, faqs.get("기본", []))

    faq_cols = st.columns(len(faq_questions))
    for idx, question in enumerate(faq_questions):
        with faq_cols[idx]:
            if st.button(
                question,
                key=f"faq_{idx}",
                use_container_width=True,
                help="클릭하여 질문하기"
            ):
                # FAQ 클릭 시 해당 질문을 처리
                process_question(question)

    st.markdown("---")

    # 대화 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 출처 표시 (assistant 메시지에만)
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 출처 보기"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")

    # 사용자 입력
    if prompt := st.chat_input("궁금하신 점을 질문해주세요..."):
        process_question(prompt)

def process_question(question: str):
    """질문 처리 및 답변 생성"""
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            response = generate_response(
                query=question,
                car_model=st.session_state.car_model,
                chat_history=st.session_state.messages
            )
            st.markdown(response["answer"])

            # 출처 표시
            with st.expander("📄 출처 보기"):
                for source in response["sources"]:
                    st.markdown(f"- {source}")

    # 응답 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"]
    })

# ============================================================
# 메인 라우팅
# ============================================================
if st.session_state.page == "selection":
    show_selection_page()
elif st.session_state.page == "chat":
    show_chat_page()
```

**주요 특징**:
1. **페이지 라우팅**: `st.session_state.page`로 "selection"과 "chat" 페이지 구분
2. **차종 선택 UI**: 버튼 그리드로 직관적인 선택
3. **FAQ 버튼**: 차종별 맞춤 FAQ 3개를 버튼으로 표시
4. **FAQ 클릭 동작**: 클릭 시 `process_question()` 호출하여 즉시 답변 생성
5. **차종 변경**: 챗봇 페이지에서 "← 차종 변경" 버튼으로 다시 선택 페이지로
6. **대화 히스토리**: 차종 변경 시 초기화, 같은 차종 내에서는 유지

### 2. src/data_loader.py (문서 로딩)

```python
from pathlib import Path
from typing import List, Dict
from llama_index.core import Document

def load_documents(data_dir: str = "data/raw") -> List[Document]:
    """차량별 문서를 로드하고 메타데이터 추가"""
    documents = []
    data_path = Path(data_dir)

    for car_folder in data_path.iterdir():
        if not car_folder.is_dir():
            continue

        car_model = car_folder.name

        for doc_file in car_folder.glob("*.md"):
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 카테고리 추출 (파일명에서)
            category = doc_file.stem.replace("_", "/")

            # Document 객체 생성
            doc = Document(
                text=content,
                metadata={
                    "car_model": car_model,
                    "category": category,
                    "source_file": doc_file.name,
                    "file_path": str(doc_file)
                }
            )
            documents.append(doc)

    return documents
```

### 3. src/chunking.py (청킹)

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from typing import List

def create_contextual_chunks(documents: List[Document]) -> List[TextNode]:
    """Contextual headers를 추가한 청킹"""

    # Sentence splitter 초기화
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    nodes = []

    for doc in documents:
        # 메타데이터에서 컨텍스트 추출
        car_model = doc.metadata.get("car_model", "")
        category = doc.metadata.get("category", "")

        # Contextual header 생성
        context_header = f"[차종: {car_model}] [카테고리: {category}]\n\n"

        # 문서를 청크로 분할
        chunks = splitter.split_text(doc.text)

        for i, chunk_text in enumerate(chunks):
            # 각 청크에 헤더 추가
            enhanced_text = context_header + chunk_text

            # TextNode 생성
            node = TextNode(
                text=enhanced_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": i
                }
            )
            nodes.append(node)

    return nodes
```

### 4. src/indexing.py (인덱싱)

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from typing import List
from llama_index.core.schema import TextNode

def create_index(nodes: List[TextNode], persist_dir: str = "vectorstore/chroma_db"):
    """Vector index 생성 및 저장"""

    # ChromaDB 클라이언트
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("vehicle_docs")

    # Vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Embedding model
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")

    # Index 생성
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    return index
```

### 5. src/retrieval.py (Hybrid Search)

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

class HybridRetriever:
    def __init__(self, index: VectorStoreIndex):
        self.index = index

        # Vector retriever
        self.vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10
        )

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=index.docstore.docs.values(),
            similarity_top_k=10
        )

    def retrieve(self, query: str, car_model: str):
        """Hybrid search with metadata filtering"""

        # Metadata filter
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="car_model", value=car_model)
            ]
        )

        # Vector retrieval
        vector_results = self.vector_retriever.retrieve(query)

        # BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(query)

        # Reciprocal Rank Fusion (간소화)
        # 실제로는 더 정교한 fusion 로직 필요
        all_results = vector_results + bm25_results

        # Metadata 필터링 적용
        filtered_results = [
            r for r in all_results
            if r.node.metadata.get("car_model") == car_model
        ]

        # 중복 제거 및 상위 k개 선택
        unique_results = self._deduplicate(filtered_results)

        return unique_results[:5]

    def _deduplicate(self, results):
        """중복 노드 제거"""
        seen = set()
        unique = []
        for r in results:
            node_id = r.node.node_id
            if node_id not in seen:
                seen.add(node_id)
                unique.append(r)
        return unique
```

### 6. src/generation.py (LLM 생성)

```python
from anthropic import Anthropic
from typing import List, Dict

client = Anthropic(api_key="your-api-key")

def generate_response(
    query: str,
    car_model: str,
    chat_history: List[Dict],
    retrieved_chunks: List[str]
) -> Dict:
    """Claude를 사용한 답변 생성"""

    # 컨텍스트 구성
    context = "\n\n---\n\n".join([
        f"문서 {i+1}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    # 시스템 프롬프트
    system_prompt = f"""당신은 {car_model} 차량의 이용 가이드를 도와주는 친절한 어시스턴트입니다.

제공된 문서를 바탕으로 사용자의 질문에 정확하고 명확하게 답변해주세요.

답변 시 유의사항:
1. 제공된 문서에 있는 정보만 사용하세요
2. 문서에 없는 내용은 "해당 정보는 제공되지 않았습니다"라고 답하세요
3. 구체적이고 실용적인 단계별 설명을 제공하세요
4. 친근하고 이해하기 쉬운 톤을 유지하세요
5. 필요시 주의사항도 함께 안내하세요"""

    # 사용자 프롬프트
    user_prompt = f"""참고 문서:
{context}

사용자 질문: {query}

위 문서를 참고하여 질문에 답변해주세요."""

    # Claude API 호출
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = response.content[0].text

    # 출처 정보 추출
    sources = [
        f"{chunk.metadata['car_model']} > {chunk.metadata['source_file']}"
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": list(set(sources))  # 중복 제거
    }
```

### 7. config.py (설정)

```python
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Paths
DATA_DIR = "data/raw"
VECTORSTORE_DIR = "vectorstore/chroma_db"
FAQ_CONFIG_PATH = "data/faqs/faq_config.json"

# Available cars
AVAILABLE_CARS = [
    "아이오닉5",
    "아이오닉6",
    "코나일렉트릭",
    "니로EV",
    "EV6",
    "테슬라모델3",
    "테슬라모델Y",
    # ... 더 추가
]

# Model settings
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "claude-3-5-sonnet-20241022"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5

# UI Settings
PAGE_TITLE = "차량 이용 가이드 챗봇"
PAGE_ICON = "🚗"
FAQ_COUNT = 3  # 차종별 표시할 FAQ 개수
```

### 8. src/faq_manager.py (FAQ 관리 - 신규)

```python
import json
from typing import List, Dict
from pathlib import Path

class FAQManager:
    """차종별 FAQ 관리 클래스"""

    def __init__(self, config_path: str = "data/faqs/faq_config.json"):
        self.config_path = Path(config_path)
        self.faqs = self._load_faqs()

    def _load_faqs(self) -> Dict[str, List[str]]:
        """FAQ 설정 파일 로드"""
        if not self.config_path.exists():
            return self._get_default_faqs()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"FAQ 로드 실패: {e}. 기본 FAQ 사용")
            return self._get_default_faqs()

    def _get_default_faqs(self) -> Dict[str, List[str]]:
        """기본 FAQ 반환"""
        return {
            "기본": [
                "차량 시동은 어떻게 거나요?",
                "에어컨 사용법을 알려주세요",
                "내비게이션 설정 방법은?"
            ]
        }

    def get_faqs_for_car(self, car_model: str, count: int = 3) -> List[str]:
        """특정 차종의 FAQ 반환"""
        return self.faqs.get(car_model, self.faqs.get("기본", []))[:count]

    def add_faq(self, car_model: str, question: str):
        """FAQ 추가"""
        if car_model not in self.faqs:
            self.faqs[car_model] = []
        self.faqs[car_model].append(question)
        self._save_faqs()

    def _save_faqs(self):
        """FAQ 저장"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.faqs, f, ensure_ascii=False, indent=2)

    def get_all_questions(self) -> List[str]:
        """모든 FAQ 질문 반환 (중복 제거)"""
        all_questions = []
        for questions in self.faqs.values():
            all_questions.extend(questions)
        return list(set(all_questions))
```

**특징**:
- FAQ 로딩, 저장, 관리 기능
- 차종별 FAQ 조회
- 동적 FAQ 추가 기능 (Phase 3에서 사용자 피드백 기반 FAQ 자동 생성)

---

## 프로젝트 타임라인

```
Week 1-2: Phase 1 (MVP)
├─ Day 1-2: 프로젝트 설정, 샘플 데이터 작성 (3-5 차종)
├─ Day 3-4: 기본 인덱싱 및 검색 구현
├─ Day 5-7: Streamlit UI 구현
├─ Day 8-10: 통합 테스트 및 버그 수정
└─ Day 11-14: 초기 사용자 피드백 수집

Week 3-5: Phase 2 (고도화)
├─ Day 15-18: Hybrid search 구현
├─ Day 19-22: Contextual chunking 적용
├─ Day 23-26: Reranker 추가
├─ Day 27-30: 대화 히스토리 관리
└─ Day 31-35: 전체 차종 데이터 추가, 테스트

Week 6-8: Phase 3 (프로덕션)
├─ Day 36-40: Qdrant 마이그레이션
├─ Day 41-45: 캐싱, 모니터링, 로깅
├─ Day 46-50: 평가 파이프라인 구축
├─ Day 51-55: 보안, 인증, Rate limiting
└─ Day 56-60: 최종 테스트 및 배포 준비
```

---

## 예상 성과

### 정량적 성과

#### 검색 정확도
- **Baseline (Vector only)**: 70%
- **Hybrid Search 적용**: 91% (30% 상대 향상)
- **Reranking 추가**: 95% (추가 4% 향상)

#### 응답 속도
- **초기 검색**: < 2초
- **캐싱 적용 후 (자주 묻는 질문)**: < 0.5초
- **사용자 체감 만족도 목표**: 95% 이상

#### 비용 효율성
- **월 예상 비용** (10,000 쿼리 기준):
  - Embedding: ~$0.10
  - LLM (Claude): ~$5-10
  - 총: **$5-10/월** (매우 경제적)

### 정성적 성과

1. **고객 지원 효율 향상**
   - Zendesk 티켓 감소 (차량 이용 문의)
   - 상담원 업무 부담 경감

2. **사용자 만족도 향상**
   - 24/7 즉시 답변 제공
   - 일관된 정보 제공

3. **지식 관리 개선**
   - 차량 정보 체계적 정리
   - 업데이트 용이성

4. **확장 가능성**
   - 새 차종 추가 용이
   - 다국어 지원 확장 가능
   - 멀티모달 (이미지) 확장 가능

---

## 리스크 및 대응 방안

### 기술적 리스크

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|-----------|
| 검색 정확도 부족 | 높음 | Hybrid search + Reranking, 평가 파이프라인 구축 |
| LLM 환각(hallucination) | 높음 | 엄격한 프롬프트, Self-RAG 적용, 출처 명시 |
| API 비용 초과 | 중간 | 캐싱, Rate limiting, 비용 모니터링 |
| 응답 속도 저하 | 중간 | 캐싱, 인덱스 최적화, Qdrant 사용 |

### 데이터 리스크

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|-----------|
| 문서 품질 불균일 | 높음 | 문서 작성 가이드라인, 리뷰 프로세스 |
| 업데이트 지연 | 중간 | 자동화된 인덱싱 파이프라인, 주기적 갱신 |
| 차종별 정보 부족 | 중간 | 우선순위 차종부터 시작, 점진적 확장 |

### 운영 리스크

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|-----------|
| 사용자 피드백 부족 | 중간 | 피드백 UI 구현, 인센티브 제공 |
| 유지보수 부담 | 낮음 | 명확한 문서화, 모듈화된 코드 |

---

## 다음 단계

### 즉시 시작 가능한 작업

1. **환경 설정**
   ```bash
   mkdir vehicle-info-chatbot
   cd vehicle-info-chatbot
   python -m venv .venv
   source .venv/bin/activate
   pip install streamlit llama-index chromadb anthropic openai python-dotenv
   ```

2. **샘플 데이터 작성**
   - 3-5개 차종 선택 (예: 아이오닉5, 코나, 니로EV)
   - 각 차종당 5개 카테고리 문서 작성
   - 총 15-25개 마크다운 파일

3. **기본 프로토타입 구현**
   - Streamlit UI 뼈대
   - LlamaIndex 인덱싱
   - 간단한 쿼리 엔진

### 의사결정 필요 사항

1. **초기 타겟 차종**: 어떤 차종부터 시작할지?
2. **배포 환경**: Streamlit Cloud / AWS / 기타?
3. **API 키 관리**: OpenAI + Anthropic 사용 승인?
4. **데이터 소스**: 기존 차량 매뉴얼 활용 가능한지?

---

## UI/UX 개선 사항 요약

### 페이지 구분 전략

#### Page 1: 차종 선택 페이지
- **목적**: 사용자가 이용할 차량을 명확하게 선택
- **디자인**: 2열 그리드 버튼 레이아웃
- **장점**:
  - 직관적인 선택 프로세스
  - 모바일에서도 사용하기 편한 큰 버튼
  - 한눈에 보이는 차종 리스트

#### Page 2: 챗봇 페이지
- **목적**: 선택된 차량에 대한 질의응답
- **구성**:
  1. 헤더: 차종 변경 버튼 + 현재 차종 표시
  2. FAQ 섹션: 자주 묻는 질문 3개 (버튼)
  3. 대화 영역: 채팅 히스토리
  4. 입력창: 자유 질문 입력
- **장점**:
  - 맥락이 명확함 (어떤 차종에 대해 묻는지)
  - FAQ로 빠른 답변 접근
  - 차종 변경이 간편함

### FAQ 기능의 가치

#### 사용자 측면
1. **빠른 답변 접근**: 타이핑 없이 클릭만으로 질문
2. **학습 효과**: FAQ를 보며 "이런 것도 물어볼 수 있구나" 인지
3. **진입 장벽 낮춤**: 무엇을 물어야 할지 모르는 사용자 가이드

#### 운영 측면
1. **데이터 수집**: 어떤 질문이 많이 클릭되는지 분석 가능
2. **FAQ 최적화**: 클릭 빈도 기반으로 FAQ 자동 갱신 (Phase 3)
3. **캐싱 효율**: 자주 묻는 질문은 캐시하여 비용 절감

#### 구현 측면
1. **차종별 맞춤화**: 전기차는 "충전", 일반차는 "주유" 등
2. **동적 업데이트**: JSON 파일 수정으로 FAQ 즉시 변경
3. **확장 가능**: 향후 AI 기반 FAQ 자동 생성 가능

---

## 결론

이 설계는 최신 RAG 기술(Hybrid search, Semantic chunking, Reranking)과 직관적인 UI/UX(페이지 구분, FAQ 버튼)를 결합하여 차량 정보 챗봇을 구현하는 실용적이고 확장 가능한 프레임워크를 제공합니다.

**핵심 강점**:
- ✅ 차종별 필터링으로 정확한 정보 제공
- ✅ 페이지 구분으로 명확한 사용자 플로우
- ✅ FAQ 버튼으로 빠른 답변 접근
- ✅ Hybrid search로 30% 검색 정확도 향상
- ✅ 경제적인 비용 ($5-10/월)
- ✅ 빠른 프로토타입 → 프로덕션 전환 가능
- ✅ 확장 가능한 아키텍처

**권장 사항**:
- Phase 1 (MVP)부터 시작하여 사용자 피드백 기반으로 점진적 개선
- LlamaIndex + Claude 조합으로 간단하면서도 강력한 시스템 구축
- FAQ 클릭 데이터를 수집하여 지속적으로 FAQ 개선
- 평가 파이프라인을 초기부터 구축하여 지속적 개선 가능하게 설정

**차별화 포인트**:
1. **페이지 기반 UX**: Sidebar 선택 방식보다 명확한 사용자 여정
2. **FAQ 우선 노출**: 80% 질문을 FAQ로 커버하여 RAG 호출 최소화
3. **차종 맞춤 FAQ**: 일반적인 FAQ가 아닌 차종별 특화 질문 제공

이 제안서를 바탕으로 프로젝트를 시작하시면 됩니다. 추가 질문이나 조정이 필요한 부분이 있다면 말씀해주세요!

---

# 헬프센터 QA RAG 시스템 설계

> 카셰어링 헬프센터 FAQ/QA 데이터를 활용한 자연어 질의응답 시스템

**작성일**: 2026년 1월 9일
**데이터 소스**: Zendesk 헬프센터 (2,216개 문서)
**목적**: 자연어 질의에 대해 가장 관련성 높은 헬프센터 문서를 검색하여 반환

---

## 목차

1. [시스템 개요](#시스템-개요-1)
2. [데이터 구조 분석](#데이터-구조-분석)
3. [RAG 아키텍처](#rag-아키텍처-1)
4. [검색 전략](#검색-전략)
5. [메타데이터 활용](#메타데이터-활용)
6. [구현 설계](#구현-설계)
7. [응답 구조](#응답-구조)
8. [평가 및 최적화](#평가-및-최적화)

---

## 시스템 개요

### 문제 정의

카셰어링 서비스 이용자가 다양한 질문(예약, 요금, 사고 처리, 차량 이용 방법 등)을 할 때:
- 관련 헬프센터 문서를 찾기 어려움
- 키워드 검색만으로는 유사한 표현을 포착하지 못함
- 카테고리별로 흩어진 정보를 통합하여 제공하기 어려움

### 솔루션

Zendesk 헬프센터의 2,216개 QA 문서를 RAG 시스템에 인덱싱하여:
- 자연어 질의에 대해 의미적으로 가장 유사한 문서 검색
- 키워드 + 의미 기반 Hybrid Search로 검색 정확도 극대화
- 원본 문서의 id, url, title, body 등을 그대로 반환하여 활용성 제고

### 핵심 기능

1. **자연어 질의 처리**: "예약을 취소하고 싶어요" → 관련 취소 정책 문서 반환
2. **Hybrid Search**: BM25 (키워드) + Vector Search (의미) 결합
3. **메타데이터 필터링**: DOMAIN_NAME, category_name으로 검색 범위 제한 가능
4. **원본 데이터 반환**: id, url, title, body, category_name 등 모든 메타데이터 포함
5. **관련도 점수**: 각 결과의 관련성 점수 제공

---

## 데이터 구조 분석

### Zendesk 헬프센터 데이터

#### 데이터 통계
- **총 문서 수**: 2,216개 (카셰어링 관련만 필터링)
- **도메인**: 4개 (쏘카, 쏘카 검색비노출, 쏘카 고객센터, 쏘카 공지사항)
- **카테고리**: 10개 (자주 하는 질문, 약관 및 정책, 공지사항 등)

#### 필드 구조

```python
{
    # 핵심 식별자
    "id": 54042400000000.0,                    # 고유 문서 ID
    "url": "https://socarhelp.zendesk.com/...", # API URL
    "html_url": "https://socarhelp.zendesk.com/hc/...", # 웹 URL

    # 메타데이터
    "DOMAIN_NAME": "쏘카",                      # 도메인 분류
    "category_name": "자주 하는 질문",          # 카테고리
    "section_name": "이용 정보",                # 섹션

    # 콘텐츠
    "title": "예약 취소는 어떻게 하나요?",      # 질문 제목
    "body": "예약 취소는 쏘카 앱에서...",       # 답변 본문

    # 부가 정보
    "created_at": "2021-10-14T02:08:22Z",
    "updated_at": "2023-06-07T05:37:40Z",
    "vote_sum": 15,                             # 사용자 투표 점수
    "vote_count": 20,
    "outdated": False                           # 오래된 문서 여부
}
```

#### 카테고리별 분포

| DOMAIN_NAME | category_name | 문서 수 | 비고 |
|-------------|--------------|---------|------|
| 쏘카 검색비노출 | 약관 및 정책 | 614 | 정책 문서 |
| 쏘카 공지사항 | 공지사항 | 427 | 시간에 민감한 정보 |
| 쏘카 검색비노출 | 기타 고객 안내 | 437 | 일반 안내 |
| 쏘카 | 자주 하는 질문 | 337 | **핵심 FAQ** |
| 쏘카 검색비노출 | 요금표 및 휴차보상료 | 96 | 요금 정보 |
| 쏘카 | 비공개 | 61 | 특별 프로모션 |
| 쏘카 검색비노출 | 차량매뉴얼 | 61 | 차량 사용법 |
| 쏘카 | 고객센터 | 16 | 주요 가이드 |
| 쏘카 고객센터 | 가이드 | 7 | 가입/인증 |
| 쏘카 고객센터 | 공지사항 | 1 | - |

---

## RAG 아키텍처

### 시스템 흐름도

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                                │
│           "예약 취소하고 싶은데 수수료가 궁금해요"                │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Query Processing Layer                         │
│                                                                   │
│  1. Query Normalization                                          │
│     - 맞춤법 보정 (선택)                                         │
│     - 불용어 제거 (선택)                                         │
│                                                                   │
│  2. Query Expansion (선택)                                       │
│     - 동의어 확장: "취소" → ["취소", "취소하기", "예약취소"]     │
│     - 약어 확장: "크레딧" → ["크레딧", "적립금"]                │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Metadata Filtering (선택)                      │
│                                                                   │
│  사용자가 필터를 지정한 경우:                                     │
│  - DOMAIN_NAME = "쏘카"                                          │
│  - category_name = "자주 하는 질문"                              │
│  → 해당 카테고리 문서만 검색 범위로 제한                          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Hybrid Retrieval Layer                         │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. BM25 Retrieval (Keyword-based)                         │ │
│  │     - "예약", "취소", "수수료" 키워드 매칭                  │ │
│  │     - TF-IDF 기반 점수 계산                                 │ │
│  │     - Top 20 문서 검색                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         ↓                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  2. Vector Search (Semantic)                               │ │
│  │     - Query → Embedding (text-embedding-3-large)           │ │
│  │     - Cosine similarity with document embeddings           │ │
│  │     - Top 20 문서 검색                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                         ↓                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  3. Reciprocal Rank Fusion (RRF)                           │ │
│  │     - BM25 결과 + Vector 결과 융합                          │ │
│  │     - 각 문서의 랭크 기반 점수 재계산                        │ │
│  │     - 중복 제거 및 정렬                                     │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Reranking Layer (Phase 2)                      │
│                                                                   │
│  Cross-Encoder Reranker:                                         │
│  - Query와 각 문서(title + body)의 관련성 점수 재계산           │
│  - Cohere Rerank API 또는 로컬 Cross-Encoder 사용               │
│  - Top 10 → Top 5로 정제                                         │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Post-processing Layer                          │
│                                                                   │
│  1. Metadata Enrichment                                          │
│     - 원본 문서의 모든 필드 첨부                                 │
│     - 관련도 점수 정규화 (0-1)                                   │
│                                                                   │
│  2. Deduplication                                                │
│     - 동일 ID 문서 제거                                          │
│                                                                   │
│  3. Ranking Adjustment (선택)                                    │
│     - vote_sum, vote_count 고려                                  │
│     - 최신 문서 우선 (updated_at)                                │
│     - outdated=True 문서 페널티                                  │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Response Formation                             │
│                                                                   │
│  반환 형식:                                                       │
│  {                                                                │
│    "query": "예약 취소하고 싶은데 수수료가 궁금해요",            │
│    "results": [                                                   │
│      {                                                            │
│        "id": 54042400000000.0,                                   │
│        "url": "https://...",                                     │
│        "html_url": "https://...",                                │
│        "title": "예약 취소 및 취소수수료",                        │
│        "body": "예약 취소는...",                                  │
│        "category_name": "자주 하는 질문",                         │
│        "DOMAIN_NAME": "쏘카",                                     │
│        "relevance_score": 0.95,                                  │
│        "vote_sum": 150                                            │
│      },                                                           │
│      ...                                                          │
│    ],                                                             │
│    "total_results": 5                                             │
│  }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 아키텍처 특징

1. **다단계 검색**: BM25 + Vector + RRF로 정확도 극대화
2. **메타데이터 활용**: 카테고리, 도메인 필터링으로 검색 범위 제한
3. **유연한 반환**: 원본 데이터 그대로 반환하여 다양한 용도 활용 가능
4. **확장 가능**: Reranking, Query Expansion 등 고급 기능 추가 용이

---

## 검색 전략

### 1. Hybrid Search 구현

#### A. BM25 (Keyword Search)

**목적**: 정확한 키워드 매칭
**대상 필드**: `title` + `body`
**장점**:
- 전문 용어 정확 매칭 (예: "크레딧", "주행료", "휴차보상료")
- 수식어 제거 후 핵심 키워드 추출
- 빠른 검색 속도

**구현 예시**:
```python
from rank_bm25 import BM25Okapi
import jieba  # 또는 KoNLPy

# 문서 전처리: title + body 결합
corpus = [
    f"{doc['title']} {doc['body']}"
    for doc in documents
]

# 토큰화
tokenized_corpus = [jieba.lcut(doc) for doc in corpus]

# BM25 인덱스 생성
bm25 = BM25Okapi(tokenized_corpus)

# 검색
query = "예약 취소 수수료"
tokenized_query = jieba.lcut(query)
scores = bm25.get_scores(tokenized_query)

# Top-k 추출
top_k_indices = scores.argsort()[-20:][::-1]
```

#### B. Vector Search (Semantic Search)

**목적**: 의미적 유사도 기반 검색
**Embedding Model**: OpenAI `text-embedding-3-large` (권장)
**대상 필드**: `title` + `body` (또는 title만)
**장점**:
- 유사 표현 포착 (예: "취소하기" ≈ "예약을 철회")
- 맥락 이해 (예: "사고 났어요" → 사고 처리 문서)

**구현 예시**:
```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Embedding 모델
embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=3072  # 또는 1536 (비용 절감)
)

# 문서 생성
documents = [
    Document(
        text=f"제목: {row['title']}\n\n내용: {row['body']}",
        metadata={
            "id": row['id'],
            "url": row['url'],
            "html_url": row['html_url'],
            "title": row['title'],
            "category_name": row['category_name'],
            "DOMAIN_NAME": row['DOMAIN_NAME'],
            "vote_sum": row['vote_sum'],
            "updated_at": row['updated_at']
        }
    )
    for _, row in df.iterrows()
]

# ChromaDB 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("help_center")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 인덱스 생성
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    vector_store=vector_store
)

# 검색
query_engine = index.as_query_engine(similarity_top_k=20)
results = query_engine.retrieve("예약 취소 수수료")
```

#### C. Reciprocal Rank Fusion (RRF)

**목적**: BM25와 Vector 결과를 융합하여 최종 순위 결정

**수식**:
```
RRF_score(d) = Σ 1 / (k + rank_i(d))

d: 문서
k: 상수 (일반적으로 60)
rank_i(d): i번째 retriever에서의 문서 d의 순위
```

**구현 예시**:
```python
def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    BM25와 Vector Search 결과를 RRF로 융합
    """
    scores = {}

    # BM25 결과
    for rank, doc_id in enumerate(bm25_results):
        if doc_id not in scores:
            scores[doc_id] = 0
        scores[doc_id] += 1 / (k + rank + 1)

    # Vector Search 결과
    for rank, doc_id in enumerate(vector_results):
        if doc_id not in scores:
            scores[doc_id] = 0
        scores[doc_id] += 1 / (k + rank + 1)

    # 점수 기준 정렬
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, score in sorted_docs]
```

**효과**:
- BM25와 Vector가 모두 높은 순위를 준 문서가 최상위로
- 한쪽에서만 검색된 문서도 결과에 포함
- Pinecone 벤치마크: **30% 관련성 향상**

### 2. Chunking 전략

#### 현재 데이터의 특징
- **문서 단위**: 각 row가 하나의 완결된 QA
- **평균 길이**: title ~50자, body ~500-2000자
- **구조**: 대부분 단일 주제 (예: "예약 취소 방법")

#### 권장 접근법

**Option 1: 문서 단위 인덱싱 (권장)**
- Chunking 없이 전체 문서(title + body)를 하나의 단위로 인덱싱
- **장점**:
  - 맥락 손실 없음
  - 구현 간단
  - 각 문서가 이미 독립적인 QA 단위
- **단점**:
  - 매우 긴 body의 경우 임베딩 품질 저하 가능

**Option 2: Contextual Chunking (긴 문서용)**
- body가 2000자 이상인 경우에만 청킹
- 각 청크에 title을 헤더로 추가

```python
from llama_index.core.node_parser import SentenceSplitter

def create_chunks(row, max_length=2000):
    """
    긴 문서만 청킹, 짧은 문서는 그대로 유지
    """
    title = row['title']
    body = row['body']

    if len(body) <= max_length:
        # 짧은 문서: 청킹 없이 그대로
        return [f"제목: {title}\n\n{body}"]
    else:
        # 긴 문서: 청킹 후 각 청크에 title 추가
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_text(body)
        return [f"제목: {title}\n\n{chunk}" for chunk in chunks]
```

**권장**: 초기에는 Option 1 (문서 단위) 사용, 필요시 Option 2 적용

---

## 메타데이터 활용

### 1. 필터링 전략

사용자가 검색 범위를 제한하고 싶을 때 메타데이터 필터를 적용

#### A. 카테고리 필터링

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

# 예: "자주 하는 질문" 카테고리만 검색
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="category_name", value="자주 하는 질문")
    ]
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    filters=filters
)
```

#### B. 도메인 필터링

```python
# 예: "쏘카" 도메인만 검색 (검색비노출 제외)
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="DOMAIN_NAME", value="쏘카")
    ]
)
```

#### C. 복합 필터링

```python
# 예: "쏘카" 도메인의 "자주 하는 질문" 카테고리만
filters = MetadataFilters(
    filters=[
        MetadataFilter(key="DOMAIN_NAME", value="쏘카"),
        MetadataFilter(key="category_name", value="자주 하는 질문")
    ]
)
```

### 2. 결과 재순위화에 메타데이터 활용

검색 후 메타데이터를 활용하여 결과 순위 조정

```python
def adjust_ranking_with_metadata(results, alpha=0.7):
    """
    검색 결과에 메타데이터 신호를 반영하여 재순위화

    alpha: 원래 점수의 가중치 (0~1)
    """
    adjusted_results = []

    for result in results:
        base_score = result['relevance_score']

        # 메타데이터 신호
        vote_score = min(result.get('vote_sum', 0) / 100, 1.0)  # 투표 점수
        recency_score = get_recency_score(result.get('updated_at'))  # 최신성
        outdated_penalty = 0 if result.get('outdated') else 1.0

        # 최종 점수
        metadata_score = (vote_score * 0.3 + recency_score * 0.3 + outdated_penalty * 0.4)
        final_score = alpha * base_score + (1 - alpha) * metadata_score

        result['final_score'] = final_score
        adjusted_results.append(result)

    # 최종 점수로 정렬
    adjusted_results.sort(key=lambda x: x['final_score'], reverse=True)

    return adjusted_results

def get_recency_score(updated_at, decay_days=365):
    """
    최신성 점수 계산 (1년 이내 문서는 1.0, 이후 지수 감소)
    """
    from datetime import datetime

    if not updated_at:
        return 0.5

    updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
    days_old = (datetime.now(updated_date.tzinfo) - updated_date).days

    # 지수 감소
    import math
    return math.exp(-days_old / decay_days)
```

### 3. 메타데이터 기반 그룹화

동일 카테고리의 결과를 그룹화하여 제시

```python
def group_results_by_category(results):
    """
    검색 결과를 카테고리별로 그룹화
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    for result in results:
        category = result.get('category_name', '기타')
        grouped[category].append(result)

    return dict(grouped)

# 사용 예시
grouped = group_results_by_category(results)

# 출력
for category, docs in grouped.items():
    print(f"\n[{category}] ({len(docs)}개)")
    for doc in docs[:3]:  # 각 카테고리에서 상위 3개만
        print(f"  - {doc['title']}")
```

---

## 구현 설계

### 디렉토리 구조

```
vehicle-info-chatbot/
├── data/
│   ├── help_center_carsharing_only_20260109_135511.csv  # 원본 데이터
│   └── processed/
│       └── embeddings_cache.pkl  # 임베딩 캐시 (선택)
│
├── vectorstore/
│   └── help_center_chroma/  # ChromaDB 저장소
│
├── src/
│   ├── data_loader.py           # CSV 데이터 로딩
│   ├── indexing.py              # 벡터 인덱스 생성
│   ├── hybrid_retriever.py      # Hybrid Search 구현
│   ├── reranker.py              # Reranking (Phase 2)
│   └── query_engine.py          # 통합 쿼리 엔진
│
├── api/
│   └── search_api.py            # FastAPI 엔드포인트
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # 데이터 분석
│   ├── 02_embedding_test.ipynb        # 임베딩 테스트
│   └── 03_search_evaluation.ipynb     # 검색 성능 평가
│
├── config.py                    # 설정
├── requirements.txt
└── README.md
```

### 코드 구현

#### 1. src/data_loader.py

```python
import pandas as pd
from llama_index.core import Document
from typing import List

def load_help_center_data(csv_path: str) -> pd.DataFrame:
    """
    CSV 파일 로드 및 전처리
    """
    df = pd.read_csv(csv_path)

    # 결측치 처리
    df['body'] = df['body'].fillna('')
    df['title'] = df['title'].fillna('제목 없음')
    df['category_name'] = df['category_name'].fillna('기타')

    # outdated 필드 처리
    df['outdated'] = df['outdated'].fillna(False)

    return df

def create_documents(df: pd.DataFrame) -> List[Document]:
    """
    DataFrame을 LlamaIndex Document 객체로 변환
    """
    documents = []

    for idx, row in df.iterrows():
        # 텍스트: title + body 결합
        text = f"제목: {row['title']}\n\n내용: {row['body']}"

        # 메타데이터
        metadata = {
            "id": str(int(row['id'])) if pd.notna(row['id']) else f"doc_{idx}",
            "url": row['url'],
            "html_url": row['html_url'],
            "title": row['title'],
            "category_name": row['category_name'],
            "DOMAIN_NAME": row['DOMAIN_NAME'],
            "section_name": row.get('section_name', ''),
            "vote_sum": int(row['vote_sum']) if pd.notna(row['vote_sum']) else 0,
            "vote_count": int(row['vote_count']) if pd.notna(row['vote_count']) else 0,
            "created_at": row['created_at'],
            "updated_at": row['updated_at'],
            "outdated": bool(row['outdated'])
        }

        doc = Document(
            text=text,
            metadata=metadata,
            excluded_llm_metadata_keys=["id", "url", "html_url"],  # LLM에는 전달 안 함
            excluded_embed_metadata_keys=["id", "url", "html_url", "created_at", "updated_at"]
        )

        documents.append(doc)

    return documents
```

#### 2. src/indexing.py

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
from typing import List
from llama_index.core.schema import Document

def create_vector_index(
    documents: List[Document],
    persist_dir: str = "vectorstore/help_center_chroma",
    collection_name: str = "help_center",
    embedding_model: str = "text-embedding-3-large"
):
    """
    벡터 인덱스 생성 및 저장
    """
    # Embedding 모델
    embed_model = OpenAIEmbedding(
        model=embedding_model,
        dimensions=1536  # 비용 절감을 위해 1536 사용 (3072도 가능)
    )

    # ChromaDB 클라이언트
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    # Vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 인덱스 생성
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print(f"✅ 인덱스 생성 완료: {len(documents)}개 문서")

    return index

def load_vector_index(
    persist_dir: str = "vectorstore/help_center_chroma",
    collection_name: str = "help_center",
    embedding_model: str = "text-embedding-3-large"
):
    """
    저장된 벡터 인덱스 로드
    """
    embed_model = OpenAIEmbedding(model=embedding_model, dimensions=1536)

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    print(f"✅ 인덱스 로드 완료")

    return index
```

#### 3. src/hybrid_retriever.py

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Optional
import numpy as np

class HybridRetriever:
    """
    BM25 + Vector Search를 결합한 Hybrid Retriever
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        documents: List[Dict],
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        final_top_k: int = 10
    ):
        self.index = index
        self.documents = documents
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k

        # Vector retriever
        self.vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=vector_top_k
        )

        # BM25 인덱스 생성
        self._build_bm25_index()

    def _build_bm25_index(self):
        """BM25 인덱스 구축"""
        corpus = [
            f"{doc['title']} {doc['body']}"
            for doc in self.documents
        ]

        # 토큰화 (한국어)
        tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict] = None,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid search 수행

        Args:
            query: 검색 쿼리
            filters: 메타데이터 필터 (예: {"category_name": "자주 하는 질문"})
            alpha: BM25 가중치 (0~1, 1이면 BM25만, 0이면 Vector만)

        Returns:
            검색 결과 리스트
        """
        # 1. BM25 검색
        bm25_results = self._bm25_search(query, top_k=self.bm25_top_k)

        # 2. Vector 검색
        vector_results = self._vector_search(query, top_k=self.vector_top_k)

        # 3. RRF 융합
        fused_results = self._reciprocal_rank_fusion(
            bm25_results, vector_results, alpha=alpha
        )

        # 4. 메타데이터 필터 적용
        if filters:
            fused_results = self._apply_filters(fused_results, filters)

        # 5. Top-k 반환
        return fused_results[:self.final_top_k]

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 검색"""
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        # Top-k 인덱스
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            result = self.documents[idx].copy()
            result['bm25_score'] = float(scores[idx])
            result['rank'] = len(results)
            results.append(result)

        return results

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Vector 검색"""
        nodes = self.vector_retriever.retrieve(query)

        results = []
        for rank, node in enumerate(nodes[:top_k]):
            result = node.metadata.copy()
            result['body'] = node.text.split('\n\n내용: ', 1)[-1]  # body 추출
            result['vector_score'] = float(node.score) if node.score else 0.0
            result['rank'] = rank
            results.append(result)

        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        alpha: float = 0.5,
        k: int = 60
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion
        """
        scores = {}
        doc_map = {}

        # BM25 결과
        for result in bm25_results:
            doc_id = result['id']
            rank = result['rank']
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (k + rank + 1)
            doc_map[doc_id] = result

        # Vector 결과
        for result in vector_results:
            doc_id = result['id']
            rank = result['rank']
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # 정렬
        sorted_doc_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 결과 구성
        fused_results = []
        for doc_id, score in sorted_doc_ids:
            result = doc_map[doc_id].copy()
            result['relevance_score'] = float(score)
            fused_results.append(result)

        return fused_results

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """메타데이터 필터 적용"""
        filtered = []
        for result in results:
            match = True
            for key, value in filters.items():
                if result.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(result)

        return filtered
```

#### 4. src/query_engine.py

```python
from typing import List, Dict, Optional
from src.hybrid_retriever import HybridRetriever

class HelpCenterQueryEngine:
    """
    헬프센터 검색 엔진 통합 인터페이스
    """

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
        alpha: float = 0.5
    ) -> Dict:
        """
        헬프센터 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            category_filter: 카테고리 필터 (예: "자주 하는 질문")
            domain_filter: 도메인 필터 (예: "쏘카")
            alpha: BM25 가중치 (0~1)

        Returns:
            검색 결과 딕셔너리
        """
        # 필터 구성
        filters = {}
        if category_filter:
            filters['category_name'] = category_filter
        if domain_filter:
            filters['DOMAIN_NAME'] = domain_filter

        # 검색 수행
        results = self.retriever.retrieve(
            query=query,
            filters=filters if filters else None,
            alpha=alpha
        )

        # 결과 포맷팅
        formatted_results = []
        for result in results[:top_k]:
            formatted_results.append({
                "id": result['id'],
                "url": result['url'],
                "html_url": result['html_url'],
                "title": result['title'],
                "body": result['body'],
                "category_name": result['category_name'],
                "DOMAIN_NAME": result['DOMAIN_NAME'],
                "section_name": result.get('section_name', ''),
                "relevance_score": round(result['relevance_score'], 4),
                "vote_sum": result.get('vote_sum', 0),
                "vote_count": result.get('vote_count', 0),
                "updated_at": result.get('updated_at', '')
            })

        return {
            "query": query,
            "filters": filters,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
```

#### 5. api/search_api.py (FastAPI 엔드포인트)

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List
from src.query_engine import HelpCenterQueryEngine
from src.data_loader import load_help_center_data, create_documents
from src.indexing import load_vector_index
from src.hybrid_retriever import HybridRetriever

app = FastAPI(title="Help Center Search API")

# 초기화 (서버 시작 시 한 번만 실행)
@app.on_event("startup")
async def startup_event():
    global query_engine

    # 데이터 로드
    df = load_help_center_data("data/help_center_carsharing_only_20260109_135511.csv")
    documents_data = df.to_dict('records')

    # 인덱스 로드
    index = load_vector_index()

    # LlamaIndex Documents 생성
    documents_llama = create_documents(df)

    # Hybrid Retriever 생성
    hybrid_retriever = HybridRetriever(
        index=index,
        documents=documents_data,
        final_top_k=20
    )

    # Query Engine 생성
    query_engine = HelpCenterQueryEngine(hybrid_retriever)

    print("✅ API 서버 시작 완료")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None
    domain_filter: Optional[str] = None
    alpha: float = 0.5

@app.post("/search")
async def search(request: SearchRequest):
    """
    헬프센터 검색 엔드포인트

    Example:
        POST /search
        {
            "query": "예약 취소하고 싶어요",
            "top_k": 5,
            "category_filter": "자주 하는 질문",
            "alpha": 0.5
        }
    """
    results = query_engine.search(
        query=request.query,
        top_k=request.top_k,
        category_filter=request.category_filter,
        domain_filter=request.domain_filter,
        alpha=request.alpha
    )

    return results

@app.get("/search")
async def search_get(
    q: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(5, ge=1, le=20, description="반환할 결과 수"),
    category: Optional[str] = Query(None, description="카테고리 필터"),
    domain: Optional[str] = Query(None, description="도메인 필터"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="BM25 가중치")
):
    """
    헬프센터 검색 (GET 방식)

    Example:
        GET /search?q=예약취소&top_k=5&category=자주 하는 질문
    """
    results = query_engine.search(
        query=q,
        top_k=top_k,
        category_filter=category,
        domain_filter=domain,
        alpha=alpha
    )

    return results

@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

#### 6. config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
DATA_PATH = "data/help_center_carsharing_only_20260109_135511.csv"
VECTORSTORE_DIR = "vectorstore/help_center_chroma"
COLLECTION_NAME = "help_center"

# Model settings
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536  # 또는 3072

# Retrieval settings
BM25_TOP_K = 20
VECTOR_TOP_K = 20
FINAL_TOP_K = 10
RRF_K = 60  # Reciprocal Rank Fusion 상수
ALPHA = 0.5  # BM25 가중치 (0: Vector만, 1: BM25만)

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
```

---

## 응답 구조

### API 응답 예시

#### 요청
```json
POST /search
{
  "query": "예약을 취소하고 싶은데 수수료가 궁금해요",
  "top_k": 3,
  "category_filter": "자주 하는 질문"
}
```

#### 응답
```json
{
  "query": "예약을 취소하고 싶은데 수수료가 궁금해요",
  "filters": {
    "category_name": "자주 하는 질문"
  },
  "total_results": 3,
  "results": [
    {
      "id": "360001757053",
      "url": "https://socarhelp.zendesk.com/api/v2/help_center/ko/articles/360001757053.json",
      "html_url": "https://socarhelp.zendesk.com/hc/ko/articles/360001757053",
      "title": "예약 변경 및 예약 취소 / 취소수수료",
      "body": "예약 취소는 쏘카 앱 > 마이페이지 > 이용/취소내역에서 가능합니다...",
      "category_name": "자주 하는 질문",
      "DOMAIN_NAME": "쏘카",
      "section_name": "예약 관리",
      "relevance_score": 0.9537,
      "vote_sum": 150,
      "vote_count": 165,
      "updated_at": "2025-11-15T08:30:00Z"
    },
    {
      "id": "360001757054",
      "url": "https://socarhelp.zendesk.com/api/v2/help_center/ko/articles/360001757054.json",
      "html_url": "https://socarhelp.zendesk.com/hc/ko/articles/360001757054",
      "title": "취소수수료는 얼마인가요?",
      "body": "취소 수수료는 예약 시작 시간 기준으로...",
      "category_name": "자주 하는 질문",
      "DOMAIN_NAME": "쏘카",
      "section_name": "요금 안내",
      "relevance_score": 0.8821,
      "vote_sum": 98,
      "vote_count": 110,
      "updated_at": "2025-10-20T14:15:00Z"
    },
    {
      "id": "360001757055",
      "url": "https://socarhelp.zendesk.com/api/v2/help_center/ko/articles/360001757055.json",
      "html_url": "https://socarhelp.zendesk.com/hc/ko/articles/360001757055",
      "title": "예약 취소 후 환불은 언제 되나요?",
      "body": "결제 수단에 따라 환불 시기가 다릅니다...",
      "category_name": "자주 하는 질문",
      "DOMAIN_NAME": "쏘카",
      "section_name": "결제/환불",
      "relevance_score": 0.8153,
      "vote_sum": 72,
      "vote_count": 85,
      "updated_at": "2025-09-05T11:20:00Z"
    }
  ]
}
```

---

## 평가 및 최적화

### 1. 검색 품질 평가

#### A. Offline 평가

**테스트 데이터셋 구축**:
- 실제 사용자 질문 100개 수집
- 각 질문에 대한 정답 문서 ID 라벨링
- 카테고리별로 균형있게 구성

**평가 지표**:
```python
def evaluate_retrieval(test_queries, ground_truth, retriever):
    """
    검색 성능 평가
    """
    metrics = {
        'precision@5': [],
        'recall@5': [],
        'mrr': [],  # Mean Reciprocal Rank
        'ndcg@5': []  # Normalized Discounted Cumulative Gain
    }

    for query, true_doc_ids in zip(test_queries, ground_truth):
        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = [r['id'] for r in results]

        # Precision@5
        relevant_retrieved = len(set(retrieved_ids) & set(true_doc_ids))
        precision = relevant_retrieved / len(retrieved_ids)
        metrics['precision@5'].append(precision)

        # Recall@5
        recall = relevant_retrieved / len(true_doc_ids)
        metrics['recall@5'].append(recall)

        # MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in true_doc_ids:
                metrics['mrr'].append(1 / rank)
                break
        else:
            metrics['mrr'].append(0)

    # 평균 계산
    return {k: sum(v) / len(v) for k, v in metrics.items()}
```

**목표 지표**:
- Precision@5: > 0.8
- Recall@5: > 0.6
- MRR: > 0.7

#### B. A/B 테스트

**비교 시나리오**:
1. Vector Search only vs Hybrid Search
2. Alpha=0.3 vs Alpha=0.5 vs Alpha=0.7
3. Reranker 적용 전 vs 후

### 2. 최적화 전략

#### A. Hyperparameter 튜닝

```python
# Alpha (BM25 가중치) 튜닝
alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
best_alpha = None
best_score = 0

for alpha in alphas:
    score = evaluate_with_alpha(alpha, test_set)
    if score > best_score:
        best_score = score
        best_alpha = alpha

print(f"Best alpha: {best_alpha}, Score: {best_score}")
```

#### B. Embedding Model 비교

| 모델 | Dimensions | 비용 | 성능 (예상) |
|------|-----------|------|------------|
| text-embedding-3-small | 512 | 낮음 | 중간 |
| text-embedding-3-large (1536) | 1536 | 중간 | 높음 |
| text-embedding-3-large (3072) | 3072 | 높음 | 최고 |

**권장**: 초기에는 1536 차원 사용, 성능 부족 시 3072로 업그레이드

#### C. 캐싱 전략

**자주 묻는 질문 캐싱**:
```python
from functools import lru_cache
import hashlib

class CachedQueryEngine:
    def __init__(self, query_engine):
        self.query_engine = query_engine
        self.cache = {}

    def search(self, query: str, **kwargs):
        # 캐시 키 생성
        cache_key = hashlib.md5(
            f"{query}_{kwargs}".encode()
        ).hexdigest()

        # 캐시 확인
        if cache_key in self.cache:
            print(f"✅ 캐시 히트: {query}")
            return self.cache[cache_key]

        # 검색 수행
        results = self.query_engine.search(query, **kwargs)

        # 캐시 저장
        self.cache[cache_key] = results

        return results
```

**효과**:
- 응답 시간: 2초 → 0.1초
- API 비용: 70% 절감 (자주 묻는 질문 기준)

---

## 구현 로드맵

### Phase 1: MVP (1-2주)

**목표**: 기본 검색 기능 동작

**구현 항목**:
- [x] CSV 데이터 로딩 (data_loader.py)
- [ ] Vector 인덱스 생성 (indexing.py)
- [ ] Vector Search 구현
- [ ] BM25 Search 구현
- [ ] Hybrid Retrieval (RRF)
- [ ] FastAPI 엔드포인트
- [ ] 기본 테스트 (10개 쿼리)

**기대 결과**:
- API 호출로 검색 가능
- Precision@5 > 0.6

### Phase 2: 고도화 (2-3주)

**목표**: 검색 정확도 및 성능 향상

**구현 항목**:
- [ ] Reranker 추가 (Cohere 또는 Cross-Encoder)
- [ ] 메타데이터 기반 재순위화
- [ ] 캐싱 레이어
- [ ] 테스트 데이터셋 구축 (100개 쿼리)
- [ ] 평가 파이프라인
- [ ] Hyperparameter 튜닝

**기대 결과**:
- Precision@5 > 0.8
- 응답 시간 < 1초 (캐시 미적중 시)

### Phase 3: 프로덕션 (2-3주)

**목표**: 실제 서비스 배포

**구현 항목**:
- [ ] Qdrant로 마이그레이션 (스케일링 대비)
- [ ] 모니터링 (로그, 메트릭)
- [ ] Rate limiting
- [ ] 에러 핸들링
- [ ] API 문서화 (Swagger)
- [ ] Docker 컨테이너화
- [ ] CI/CD 파이프라인

**기대 결과**:
- 안정적인 프로덕션 서비스
- 99% uptime

---

## 기술 스택 요약

| 구분 | 기술 | 용도 |
|------|------|------|
| **Embedding** | OpenAI text-embedding-3-large | 문서 및 쿼리 임베딩 |
| **Vector DB** | ChromaDB (초기) → Qdrant (프로덕션) | 벡터 저장 및 검색 |
| **Framework** | LlamaIndex | 인덱싱 및 검색 |
| **BM25** | rank-bm25 | 키워드 검색 |
| **Tokenizer** | jieba | 한국어 토큰화 |
| **API** | FastAPI | REST API |
| **Reranker** | Cohere Rerank (Phase 2) | 결과 재순위화 |
| **Deployment** | Docker + Kubernetes (선택) | 배포 |

---

## 예상 비용

### 월간 비용 (10,000 쿼리 기준)

| 항목 | 비용 | 설명 |
|------|------|------|
| Embedding (인덱싱) | $0.50 | 2,216 문서 × $0.13/1M tokens |
| Embedding (검색) | $0.10 | 10,000 쿼리 × $0.13/1M tokens |
| Reranker (Cohere) | $2.00 | 10,000 쿼리 × $0.002/search (Phase 2) |
| **총계** | **$2.60/월** | 매우 경제적 |

**참고**:
- 캐싱 적용 시 70% 비용 절감 가능 → **$0.78/월**
- LLM 생성은 별도 (필요시 Claude 추가)

---

## 결론 및 권장사항

### 핵심 강점

1. **정확한 검색**: Hybrid Search (BM25 + Vector)로 30% 성능 향상
2. **유연한 필터링**: 카테고리, 도메인별 검색 범위 제한
3. **원본 데이터 반환**: 모든 메타데이터 포함하여 활용성 극대화
4. **경제적**: 월 $2.60 (10,000 쿼리 기준)
5. **확장 가능**: Reranking, Query Expansion 등 고급 기능 추가 용이

### 권장 사항

1. **Phase 1부터 시작**: MVP로 빠르게 프로토타입 구축
2. **평가 파이프라인 조기 구축**: 지속적 개선을 위한 평가 체계
3. **사용자 피드백 수집**: 실제 검색 로그 분석하여 개선
4. **캐싱 적극 활용**: 비용 절감 및 응답 속도 개선

### 차별화 포인트

1. **도메인 특화**: 카셰어링 서비스에 최적화된 검색
2. **메타데이터 활용**: 카테고리, 투표 점수, 최신성 등 다양한 신호 활용
3. **Hybrid Search**: 키워드 + 의미 검색 결합으로 높은 정확도

---

**다음 단계**: Phase 1 MVP 구현 시작

1. 환경 설정
   ```bash
   pip install llama-index chromadb openai rank-bm25 jieba fastapi uvicorn
   ```

2. 인덱스 생성
   ```python
   python -m src.indexing
   ```

3. API 서버 실행
   ```bash
   uvicorn api.search_api:app --reload
   ```

4. 테스트
   ```bash
   curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "예약 취소 방법", "top_k": 5}'
   ```
