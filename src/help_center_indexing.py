"""
Help Center Vector Store Indexing

쏘카 Help Center 문서를 ChromaDB에 인덱싱
(예약, 결제, 사고, 보험 등 일반 서비스 문의)
"""
import chromadb
import pandas as pd
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from typing import List, Optional
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_COLLECTION_NAME = "help_center"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


def load_help_center_data(data_path: str | Path) -> pd.DataFrame:
    """
    Help Center CSV 데이터 로드

    Args:
        data_path: CSV 파일 경로

    Returns:
        DataFrame
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


def create_help_center_documents(
    df: pd.DataFrame,
    max_chunk_size: int = 2000
) -> List[Document]:
    """
    Help Center DataFrame을 LlamaIndex Document 리스트로 변환

    Args:
        df: Help Center DataFrame
        max_chunk_size: 최대 청크 크기 (문자 수)

    Returns:
        List of Document
    """
    documents = []

    for _, row in df.iterrows():
        title = row['title'] or ''
        body = clean_html_body(row['body'] or '')

        if not body:
            continue

        # 메타데이터
        metadata = {
            'id': str(row['id']),
            'title': title,
            'html_url': row['html_url'] or '',
            'category_name': row['category_name'] or '',
            'section_name': row['section_name'] or '',
            'source': 'help_center'
        }

        # 텍스트 구성: 제목 + 본문
        full_text = f"[{title}]\n\n{body}"

        # 청크 분할 (긴 문서의 경우)
        if len(full_text) <= max_chunk_size:
            doc = Document(
                text=full_text,
                metadata=metadata
            )
            documents.append(doc)
        else:
            # 긴 문서는 분할
            chunks = split_text(full_text, max_chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i
                doc = Document(
                    text=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)

    logger.info(f"Created {len(documents)} documents from {len(df)} articles")
    return documents


def split_text(text: str, max_size: int) -> List[str]:
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


def create_help_center_index(
    documents: List[Document],
    persist_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
) -> VectorStoreIndex:
    """
    Help Center 문서로 Vector Store 인덱스 생성
    """
    logger.info(f"Creating Help Center index with {len(documents)} documents")
    logger.info(f"  - Collection: {collection_name}")
    logger.info(f"  - Persist dir: {persist_dir}")

    # 임베딩 모델 설정
    embed_model = OpenAIEmbedding(
        model=embedding_model,
        dimensions=embedding_dimensions
    )

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
    chroma_collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Help Center documents for SOCAR chatbot"}
    )
    logger.info(f"Created collection: {collection_name}")

    # Vector Store 설정
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 인덱스 생성
    logger.info("Building vector index (this may take a few minutes)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    logger.info(f"Index created successfully!")
    return index


def load_help_center_index(
    persist_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
) -> VectorStoreIndex:
    """
    저장된 Help Center Vector Store 인덱스 로드
    """
    logger.info(f"Loading Help Center index from {persist_dir}")

    # 임베딩 모델 설정
    embed_model = OpenAIEmbedding(
        model=embedding_model,
        dimensions=embedding_dimensions
    )

    # ChromaDB 클라이언트 연결
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_collection(collection_name)

    logger.info(f"Loaded collection: {collection_name}")
    logger.info(f"  - Documents in collection: {chroma_collection.count()}")

    # Vector Store 설정
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 인덱스 로드
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    logger.info("Index loaded successfully!")
    return index


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
    print("Building Help Center Vector Index")
    print("=" * 60)

    # 데이터 로드
    df = load_help_center_data(DATA_PATH)

    # 문서 생성
    documents = create_help_center_documents(df, max_chunk_size=2000)

    print(f"\nTotal documents to index: {len(documents)}")

    # 인덱스 생성
    index = create_help_center_index(
        documents=documents,
        persist_dir=VECTORSTORE_DIR,
        collection_name=DEFAULT_COLLECTION_NAME
    )

    print("\n" + "=" * 60)
    print("Index creation complete!")
    print("=" * 60)
