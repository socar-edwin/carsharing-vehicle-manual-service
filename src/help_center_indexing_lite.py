"""
Help Center Vector Store Indexing (Lite Version)

LlamaIndex 없이 직접 ChromaDB + OpenAI 임베딩 사용
- 불필요한 메타데이터 제거로 용량 50-70% 절감
"""
import chromadb
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Optional
from pathlib import Path
import logging
import re
import hashlib

logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_COLLECTION_NAME = "help_center"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100  # OpenAI 임베딩 배치 크기


def load_help_center_data(data_path: str | Path) -> pd.DataFrame:
    """
    Help Center CSV 데이터 로드 및 필터링
    """
    logger.info(f"Loading Help Center data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"  - 전체 행: {len(df)}")

    # 1. 드래프트 제외 (draft = False만)
    df = df[df['draft'] == False]
    logger.info(f"  - draft=False 후: {len(df)}")

    # 2. 시승하기 서비스 제외
    df = df[~df['section_name'].str.contains('시승', na=False)]
    logger.info(f"  - 시승 제외 후: {len(df)}")

    # 3. 차량매뉴얼 카테고리 제거
    df = df[df['category_name'] != '차량매뉴얼']
    logger.info(f"  - 차량매뉴얼 제외 후: {len(df)}")

    # 4. 비공개 카테고리 제외
    df = df[df['category_name'] != '비공개']
    logger.info(f"  - 비공개 제외 후: {len(df)}")

    # 필요한 컬럼만 선택
    required_cols = ['id', 'title', 'body', 'html_url', 'category_name', 'section_name']
    df = df[required_cols].copy()

    # 빈 body 제거
    df = df[df['body'].notna() & (df['body'].str.strip() != '')]

    logger.info(f"Loaded {len(df)} Help Center articles (filtered)")
    return df


def clean_html_body(body: str) -> str:
    """HTML 태그 제거 및 텍스트 정리"""
    if not body:
        return ""

    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', ' ', body)
    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text)
    # 앞뒤 공백 제거
    text = text.strip()

    return text


def split_text(text: str, max_size: int = 2000) -> List[str]:
    """텍스트를 최대 크기로 분할 (문장 단위)"""
    if len(text) <= max_size:
        return [text]

    # 문장 단위로 분할
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_documents(df: pd.DataFrame, max_chunk_size: int = 2000) -> List[Dict]:
    """
    DataFrame을 문서 리스트로 변환 (경량 메타데이터)
    """
    documents = []
    seen_ids = set()

    for idx, row in df.iterrows():
        title = row['title'] or ''
        body = clean_html_body(row['body'] or '')

        if not body:
            continue

        # 텍스트 구성: 제목 + 본문
        full_text = f"[{title}]\n\n{body}"

        # article_id 정수화 (부동소수점 제거)
        article_id = str(int(float(row['id'])))

        # 경량 메타데이터 (필수 정보만)
        metadata = {
            'article_id': article_id,
            'title': title,
            'url': row['html_url'] or '',
            'category': row['category_name'] or '',
            'section': row['section_name'] or '',
        }

        # 청크 분할 (긴 문서의 경우)
        if len(full_text) <= max_chunk_size:
            # URL을 포함해서 고유 ID 생성
            doc_id = hashlib.md5(f"{row['html_url']}_{idx}_0".encode()).hexdigest()
            if doc_id in seen_ids:
                doc_id = hashlib.md5(f"{row['html_url']}_{idx}_0_{title}".encode()).hexdigest()
            seen_ids.add(doc_id)
            documents.append({
                'id': doc_id,
                'text': full_text,
                'metadata': metadata
            })
        else:
            chunks = split_text(full_text, max_chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk'] = i
                doc_id = hashlib.md5(f"{row['html_url']}_{idx}_{i}".encode()).hexdigest()
                if doc_id in seen_ids:
                    doc_id = hashlib.md5(f"{row['html_url']}_{idx}_{i}_{title}".encode()).hexdigest()
                seen_ids.add(doc_id)
                documents.append({
                    'id': doc_id,
                    'text': chunk,
                    'metadata': chunk_metadata
                })

    logger.info(f"Created {len(documents)} documents from {len(df)} articles")
    return documents


def get_embeddings(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
) -> List[List[float]]:
    """
    OpenAI API로 임베딩 생성
    """
    client = OpenAI()

    response = client.embeddings.create(
        input=texts,
        model=model,
        dimensions=dimensions
    )

    return [item.embedding for item in response.data]


def create_help_center_index(
    documents: List[Dict],
    persist_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
):
    """
    Help Center 문서로 ChromaDB 인덱스 생성 (LlamaIndex 없이)
    """
    logger.info(f"Creating Help Center index with {len(documents)} documents")
    logger.info(f"  - Collection: {collection_name}")
    logger.info(f"  - Persist dir: {persist_dir}")
    logger.info(f"  - Embedding: {embedding_model} ({embedding_dimensions}D)")

    # ChromaDB 클라이언트 생성
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))

    # 기존 컬렉션 삭제 (재생성)
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # 새 컬렉션 생성
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Help Center documents for SOCAR chatbot (lite)"}
    )
    logger.info(f"Created collection: {collection_name}")

    # 배치로 임베딩 생성 및 저장
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(documents), BATCH_SIZE):
        batch = documents[batch_idx:batch_idx + BATCH_SIZE]
        current_batch = batch_idx // BATCH_SIZE + 1

        logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch)} docs)")

        # 텍스트 추출
        texts = [doc['text'] for doc in batch]
        ids = [doc['id'] for doc in batch]
        metadatas = [doc['metadata'] for doc in batch]

        # 임베딩 생성
        embeddings = get_embeddings(
            texts,
            model=embedding_model,
            dimensions=embedding_dimensions
        )

        # ChromaDB에 저장
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    logger.info(f"Index created successfully! Total: {collection.count()} documents")
    return collection


def load_help_center_collection(
    persist_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> chromadb.Collection:
    """
    저장된 Help Center 컬렉션 로드
    """
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma_client.get_collection(collection_name)
    logger.info(f"Loaded collection: {collection_name} ({collection.count()} docs)")
    return collection


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from config import PROJECT_ROOT

    # 경로 설정
    DATA_PATH = PROJECT_ROOT / "data" / "help_center_carsharing_only_20260109_135511.csv"
    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "help_center_chroma"

    print("=" * 60)
    print("Building Help Center Vector Index (Lite)")
    print("=" * 60)

    # 데이터 로드
    df = load_help_center_data(DATA_PATH)

    # 문서 생성
    documents = create_documents(df, max_chunk_size=2000)

    print(f"\nTotal documents to index: {len(documents)}")

    # 인덱스 생성
    collection = create_help_center_index(
        documents=documents,
        persist_dir=VECTORSTORE_DIR,
        collection_name=DEFAULT_COLLECTION_NAME
    )

    print("\n" + "=" * 60)
    print("Index creation complete!")
    print(f"Output: {VECTORSTORE_DIR}")
    print("=" * 60)
