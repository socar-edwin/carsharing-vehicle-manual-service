"""
Help Center Retriever

Help Center 문서 검색 및 포맷팅 (경량화 버전 - LlamaIndex 미사용)
"""
import chromadb
from openai import OpenAI
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_COLLECTION_NAME = "help_center"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


class HelpCenterRetriever:
    """
    Help Center 문서 검색 클래스 (ChromaDB 직접 사용)
    """

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        default_top_k: int = 5
    ):
        """
        Initialize HelpCenterRetriever

        Args:
            persist_dir: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
            embedding_model: OpenAI 임베딩 모델
            embedding_dimensions: 임베딩 차원
            default_top_k: 기본 검색 결과 수
        """
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.default_top_k = default_top_k
        self._openai_client = OpenAI()

        # ChromaDB 연결
        self._chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._chroma_client.get_collection(collection_name)

        logger.info(f"HelpCenterRetriever initialized")
        logger.info(f"  - Collection: {collection_name} ({self._collection.count()} docs)")
        logger.info(f"  - default_top_k: {default_top_k}")

    def _get_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터 생성"""
        response = self._openai_client.embeddings.create(
            input=text,
            model=self.embedding_model,
            dimensions=self.embedding_dimensions
        )
        return response.data[0].embedding

    def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Help Center 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 검색 결과 수

        Returns:
            검색 결과 딕셔너리
        """
        k = top_k or self.default_top_k

        # 쿼리 임베딩 생성
        query_embedding = self._get_embedding(query)

        # ChromaDB 검색
        search_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        # 결과 포맷팅
        results = []
        documents = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        distances = search_results.get('distances', [[]])[0]

        for text, metadata, distance in zip(documents, metadatas, distances):
            # ChromaDB는 L2 거리를 반환, 유사도로 변환
            similarity = 1 / (1 + distance)

            result = {
                'text': text,
                'title': metadata.get('title', ''),
                'html_url': metadata.get('url', ''),
                'category_name': metadata.get('category', ''),
                'section_name': metadata.get('section', ''),
                'relevance_score': similarity
            }
            results.append(result)

        logger.info(f"Help Center search for '{query[:30]}...' returned {len(results)} results")

        return {
            'query': query,
            'results': results,
            'total': len(results)
        }

    def get_context_for_llm(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> str:
        """
        LLM 컨텍스트용 검색 결과 포맷팅

        Args:
            query: 검색 쿼리
            top_k: 검색 결과 수

        Returns:
            포맷된 컨텍스트 문자열
        """
        search_results = self.search(query, top_k)
        results = search_results.get('results', [])

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            title = result.get('title', '')
            text = result.get('text', '')
            category = result.get('category_name', '')

            context_parts.append(f"[도움말 {i}: {title}]\n카테고리: {category}\n{text}")

        return "\n\n".join(context_parts)


def format_help_center_context(results: List[Dict]) -> str:
    """
    Help Center 검색 결과를 LLM 컨텍스트로 포맷팅

    Args:
        results: 검색 결과 리스트

    Returns:
        포맷된 컨텍스트 문자열
    """
    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results, 1):
        title = result.get('title', '')
        text = result.get('text', '')
        category = result.get('category_name', '')

        context_parts.append(f"[도움말 {i}: {title}]\n카테고리: {category}\n{text}")

    return "\n\n".join(context_parts)
