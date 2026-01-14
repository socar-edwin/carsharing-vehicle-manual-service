# Vehicle Manual RAG Chatbot

카셰어링 차량별 조작 방법을 안내하는 AI 챗봇입니다. 사용자가 차종을 선택하면 해당 차량의 매뉴얼에서 시동, 주유/충전, 블루투스, 트렁크 등 조작 방법을 자연어로 답변합니다.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)

## Features

- **차량 매뉴얼 RAG 챗봇** - ChromaDB + OpenAI 기반 검색 증강 생성
- **car_class_id URL 파라미터** - 앱에서 호출 시 자동 차종 선택
- **전기차/일반차 자동 감지** - 차종에 따라 FAQ 동적 변경
- **Heading ID 기반 출처 링크** - 답변과 함께 매뉴얼 섹션 바로가기 제공
- **모바일 반응형 UI** - SOCAR 브랜딩 적용

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/vehicle-info-chatbot.git
cd vehicle-info-chatbot

# 가상환경 생성 (권장)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
cp .env.example .env
```

`.env` 파일 편집:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### 3. Run

```bash
streamlit run app_vehicle.py
```

브라우저에서 http://localhost:8501 접속

### 4. URL Parameter (앱 연동)

앱에서 챗봇 호출 시 car_class_id를 전달하면 자동으로 해당 차종 선택:
```
http://localhost:8501/?car_class_id=368
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  car_class_id   │────▶│  Mapping Table   │────▶│  Vehicle Name   │
│  (URL Param)    │     │  (CSV)           │     │  Filter         │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  User Question  │────▶│  Query Analysis  │◀─────────────┘
│                 │     │  (LLM)           │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ChromaDB       │◀───▶│  RAG Retrieval   │────▶│  LLM Response   │
│  (Vector Store) │     │                  │     │  + Source URL   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Project Structure

```
vehicle-info-chatbot/
├── app_vehicle.py              # 메인 앱 (Vehicle Manual Chatbot)
├── config.py                   # 환경 설정
├── requirements.txt            # Python 의존성
│
├── src/                        # Core modules
│   ├── vehicle_chatbot.py      # LLM 챗봇 로직
│   ├── vehicle_retriever.py    # 차량 매뉴얼 검색
│   ├── vehicle_indexing.py     # ChromaDB 인덱스
│   └── vehicle_data_loader.py  # CSV 로더
│
├── data/                       # 데이터
│   ├── vehicle_manual_data.csv           # 차량 매뉴얼 (105개 차종)
│   └── car_class_manual_mapping.csv      # car_class_id 매핑 (128개)
│
├── img/                        # 이미지 에셋
│   ├── Socar_Symbol_RGB.png
│   └── Socar_Signature_WhiteBG_RGB.png
│
└── vectorstore/                # ChromaDB 벡터 인덱스
    └── vehicle_manuals_chroma/ # 5,613 chunks
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API 키 |

### Model Settings (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI 임베딩 모델 |
| `EMBEDDING_DIMENSIONS` | `1536` | 벡터 차원 |

## Rebuilding Vector Index

매뉴얼 데이터 변경 시:
```bash
python -m src.vehicle_indexing
```

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini
- **Embedding**: OpenAI text-embedding-3-large
- **Vector Store**: ChromaDB
- **UI**: Streamlit
- **Framework**: LlamaIndex

## API (Optional)

Help Center Search API도 포함되어 있습니다:

```bash
# API 서버 실행
uvicorn api.search_api:app --reload

# 테스트
curl "http://localhost:8000/search?q=예약취소&top_k=3"
```

자세한 내용은 `api/search_api.py` 참조

## Related Documents

- [DESIGN_PROPOSAL.md](./DESIGN_PROPOSAL.md) - 시스템 설계 문서
- [RAG_RESEARCH.md](./RAG_RESEARCH.md) - RAG 기술 연구
- [VEHICLE_CHATBOT_DESIGN.md](./VEHICLE_CHATBOT_DESIGN.md) - 차량 챗봇 설계

## License

Private Project - SOCAR
