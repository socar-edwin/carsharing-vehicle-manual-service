"""
Vehicle Manual Vector Store Indexing

차종별 매뉴얼을 ChromaDB에 인덱싱
"""
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from typing import List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_COLLECTION_NAME = "vehicle_manuals"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


def create_vehicle_index(
    documents: List[Document],
    persist_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
) -> VectorStoreIndex:
    """
    차종 매뉴얼 문서로 Vector Store 인덱스 생성

    Args:
        documents: LlamaIndex Document 리스트
        persist_dir: ChromaDB 저장 경로
        collection_name: 컬렉션 이름
        embedding_model: OpenAI 임베딩 모델
        embedding_dimensions: 임베딩 차원

    Returns:
        VectorStoreIndex
    """
    logger.info(f"Creating vehicle manual index with {len(documents)} documents")
    logger.info(f"  - Collection: {collection_name}")
    logger.info(f"  - Embedding model: {embedding_model}")
    logger.info(f"  - Dimensions: {embedding_dimensions}")
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
        metadata={"description": "Vehicle manual documents for SOCAR chatbot"}
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
    logger.info(f"  - Total documents indexed: {len(documents)}")

    return index


def load_vehicle_index(
    persist_dir: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
) -> VectorStoreIndex:
    """
    저장된 Vector Store 인덱스 로드

    Args:
        persist_dir: ChromaDB 저장 경로
        collection_name: 컬렉션 이름
        embedding_model: OpenAI 임베딩 모델
        embedding_dimensions: 임베딩 차원

    Returns:
        VectorStoreIndex
    """
    logger.info(f"Loading vehicle manual index from {persist_dir}")

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


def get_vehicle_names_from_index(persist_dir: str | Path, collection_name: str = DEFAULT_COLLECTION_NAME) -> List[str]:
    """
    인덱스에서 차종 목록 가져오기

    Args:
        persist_dir: ChromaDB 저장 경로
        collection_name: 컬렉션 이름

    Returns:
        List of vehicle names
    """
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    collection = chroma_client.get_collection(collection_name)

    # 모든 메타데이터 가져오기
    results = collection.get(include=['metadatas'])

    vehicle_names = set()
    for metadata in results['metadatas']:
        if metadata and 'vehicle_name' in metadata:
            vehicle_names.add(metadata['vehicle_name'])

    return sorted(list(vehicle_names))


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from config import OPENAI_API_KEY, PROJECT_ROOT
    from src.vehicle_data_loader import load_vehicle_data, create_vehicle_documents

    # 경로 설정
    DATA_PATH = PROJECT_ROOT / "data" / "vehicle_manual_data.csv"
    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "vehicle_manuals_chroma"

    print("=" * 60)
    print("Building Vehicle Manual Vector Index")
    print("=" * 60)

    # 데이터 로드
    df = load_vehicle_data(DATA_PATH)

    # 문서 생성
    documents = create_vehicle_documents(df, chunk_by_section=True, max_chunk_size=1500)

    print(f"\nTotal documents to index: {len(documents)}")

    # 인덱스 생성
    index = create_vehicle_index(
        documents=documents,
        persist_dir=VECTORSTORE_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions=DEFAULT_EMBEDDING_DIMENSIONS
    )

    print("\n" + "=" * 60)
    print("Index creation complete!")
    print("=" * 60)

    # 테스트: 인덱스 로드
    print("\nTesting index load...")
    loaded_index = load_vehicle_index(
        persist_dir=VECTORSTORE_DIR,
        collection_name=DEFAULT_COLLECTION_NAME
    )

    # 테스트: 차종 목록
    vehicle_names = get_vehicle_names_from_index(VECTORSTORE_DIR)
    print(f"\nVehicles in index: {len(vehicle_names)}")
    for name in vehicle_names[:10]:
        print(f"  - {name}")
    if len(vehicle_names) > 10:
        print(f"  ... and {len(vehicle_names) - 10} more")
