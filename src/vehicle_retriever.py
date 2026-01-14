"""
Vehicle Manual Retriever with Metadata Filtering

차종 필터링을 적용한 Vector 검색
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)
from llama_index.core.retrievers import VectorIndexRetriever
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VehicleRetriever:
    """
    차종별 메타데이터 필터링을 적용한 검색기
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        default_top_k: int = 5
    ):
        """
        Initialize VehicleRetriever

        Args:
            index: VectorStoreIndex 인스턴스
            default_top_k: 기본 검색 결과 수
        """
        self.index = index
        self.default_top_k = default_top_k
        logger.info(f"VehicleRetriever initialized (default_top_k={default_top_k})")

    def retrieve(
        self,
        query: str,
        vehicle_name: str,
        top_k: Optional[int] = None,
        vehicle_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        차종 필터링된 검색 수행

        Args:
            query: 검색 쿼리
            vehicle_name: 차종명 (필수)
            top_k: 검색 결과 수
            vehicle_type: 차량 유형 (선택, 추가 필터)

        Returns:
            검색 결과 리스트 (dict 형태)
        """
        top_k = top_k or self.default_top_k

        logger.info(f"Searching for vehicle: {vehicle_name}")
        logger.info(f"Query: {query}")

        # 메타데이터 필터 구성
        filters = [
            MetadataFilter(
                key="vehicle_name",
                value=vehicle_name,
                operator=FilterOperator.EQ
            )
        ]

        # 차량 유형 필터 추가 (선택)
        if vehicle_type:
            filters.append(
                MetadataFilter(
                    key="vehicle_type",
                    value=vehicle_type,
                    operator=FilterOperator.EQ
                )
            )

        metadata_filters = MetadataFilters(
            filters=filters,
            condition=FilterCondition.AND
        )

        # Retriever 생성
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            filters=metadata_filters
        )

        # 검색 실행
        nodes = retriever.retrieve(query)

        # 결과 포맷팅
        results = []
        for node in nodes:
            result = {
                'text': node.node.text,
                'score': node.score,
                'vehicle_name': node.node.metadata.get('vehicle_name', ''),
                'vehicle_type': node.node.metadata.get('vehicle_type', ''),
                'section': node.node.metadata.get('section', ''),
                'html_url': node.node.metadata.get('html_url', ''),
                'source_id': node.node.metadata.get('source_id', ''),
            }
            results.append(result)

        logger.info(f"Found {len(results)} results")

        return results

    def retrieve_all_vehicles(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        모든 차종에서 검색 (필터 없음)

        Args:
            query: 검색 쿼리
            top_k: 검색 결과 수

        Returns:
            검색 결과 리스트
        """
        top_k = top_k or self.default_top_k

        logger.info(f"Searching all vehicles")
        logger.info(f"Query: {query}")

        # 필터 없이 검색
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )

        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            result = {
                'text': node.node.text,
                'score': node.score,
                'vehicle_name': node.node.metadata.get('vehicle_name', ''),
                'vehicle_type': node.node.metadata.get('vehicle_type', ''),
                'section': node.node.metadata.get('section', ''),
                'html_url': node.node.metadata.get('html_url', ''),
                'source_id': node.node.metadata.get('source_id', ''),
            }
            results.append(result)

        logger.info(f"Found {len(results)} results")

        return results

    def retrieve_by_type(
        self,
        query: str,
        vehicle_type: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        차량 유형으로 필터링된 검색

        Args:
            query: 검색 쿼리
            vehicle_type: 차량 유형 (경형, SUV, 전기차 등)
            top_k: 검색 결과 수

        Returns:
            검색 결과 리스트
        """
        top_k = top_k or self.default_top_k

        logger.info(f"Searching vehicle type: {vehicle_type}")
        logger.info(f"Query: {query}")

        metadata_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="vehicle_type",
                    value=vehicle_type,
                    operator=FilterOperator.EQ
                )
            ]
        )

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            filters=metadata_filters
        )

        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            result = {
                'text': node.node.text,
                'score': node.score,
                'vehicle_name': node.node.metadata.get('vehicle_name', ''),
                'vehicle_type': node.node.metadata.get('vehicle_type', ''),
                'section': node.node.metadata.get('section', ''),
                'html_url': node.node.metadata.get('html_url', ''),
                'source_id': node.node.metadata.get('source_id', ''),
            }
            results.append(result)

        logger.info(f"Found {len(results)} results")

        return results

    def get_vehicle_document(
        self,
        vehicle_name: str,
        query: str = "",
        max_chunks: int = 20
    ) -> List[Dict[str, Any]]:
        """
        특정 차종의 매뉴얼 문서 로드 (쿼리 기반 관련 청크 우선)

        차종당 매뉴얼이 1개이므로, 해당 차종의 청크를 가져오되
        사용자 쿼리와 관련된 청크를 우선적으로 로드

        Args:
            vehicle_name: 차종명
            query: 사용자 질문 (관련 청크 우선 로드용)
            max_chunks: 최대 청크 수 (기본 20개)

        Returns:
            해당 차종의 청크 리스트 (관련도 순)
        """
        logger.info(f"Loading document for vehicle: {vehicle_name}, query: '{query}'")

        # 메타데이터 필터로 해당 차종의 문서만 가져오기
        metadata_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="vehicle_name",
                    value=vehicle_name,
                    operator=FilterOperator.EQ
                )
            ]
        )

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=max_chunks,
            filters=metadata_filters
        )

        # 사용자 쿼리로 검색하여 관련 청크 우선 로드
        search_query = query if query else "차량 매뉴얼 정보"
        nodes = retriever.retrieve(search_query)

        results = []
        seen_texts = set()  # 중복 제거용

        for node in nodes:
            text = node.node.text
            # 중복 청크 제거
            text_hash = hash(text[:100])
            if text_hash in seen_texts:
                continue
            seen_texts.add(text_hash)

            result = {
                'text': text,
                'score': 1.0,  # 전체 문서이므로 score 의미 없음
                'vehicle_name': node.node.metadata.get('vehicle_name', ''),
                'vehicle_type': node.node.metadata.get('vehicle_type', ''),
                'section': node.node.metadata.get('section', ''),
                'html_url': node.node.metadata.get('html_url', ''),
                'source_id': node.node.metadata.get('source_id', ''),
                'chunk_index': node.node.metadata.get('chunk_index', 0),
                'heading_map': node.node.metadata.get('heading_map', ''),  # heading ID 매핑
            }
            results.append(result)

        # chunk_index로 정렬하여 문서 순서 유지
        results.sort(key=lambda x: x.get('chunk_index', 0))

        logger.info(f"Loaded {len(results)} chunks for {vehicle_name}")

        return results


def format_context_for_llm(results: List[Dict[str, Any]]) -> str:
    """
    검색 결과를 LLM 컨텍스트용 텍스트로 포맷팅

    Args:
        results: 검색 결과 리스트

    Returns:
        포맷팅된 컨텍스트 문자열
    """
    if not results:
        return "관련 매뉴얼 정보를 찾을 수 없습니다."

    context_parts = []

    for i, result in enumerate(results, 1):
        section = result.get('section', '정보')
        text = result.get('text', '')
        score = result.get('score', 0)

        context_parts.append(f"[참고 {i}] ({section}, 관련도: {score:.2f})\n{text}")

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from config import PROJECT_ROOT
    from src.vehicle_indexing import load_vehicle_index

    # 경로 설정
    VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore" / "vehicle_manuals_chroma"

    print("=" * 60)
    print("Testing Vehicle Retriever")
    print("=" * 60)

    # 인덱스 로드
    try:
        index = load_vehicle_index(persist_dir=VECTORSTORE_DIR)
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Please run vehicle_indexing.py first to create the index.")
        sys.exit(1)

    # Retriever 생성
    retriever = VehicleRetriever(index=index, default_top_k=3)

    # 테스트 1: 특정 차종 검색
    print("\n" + "=" * 60)
    print("Test 1: 아이오닉 5 - '충전 방법'")
    print("=" * 60)

    results = retriever.retrieve(
        query="충전은 어떻게 하나요?",
        vehicle_name="아이오닉 5"
    )

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Section: {r['section']}")
        print(f"Text preview: {r['text'][:200]}...")

    # 테스트 2: 다른 차종
    print("\n" + "=" * 60)
    print("Test 2: 캐스퍼 - '시동 거는 방법'")
    print("=" * 60)

    results = retriever.retrieve(
        query="시동은 어떻게 거나요?",
        vehicle_name="캐스퍼"
    )

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        print(f"Section: {r['section']}")
        print(f"Text preview: {r['text'][:200]}...")

    # 테스트 3: 컨텍스트 포맷팅
    print("\n" + "=" * 60)
    print("Test 3: Context formatting")
    print("=" * 60)

    context = format_context_for_llm(results)
    print(context[:500] + "...")
