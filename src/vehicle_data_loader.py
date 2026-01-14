"""
Vehicle Manual Data Loader with Chunking

차종별 매뉴얼 데이터를 로드하고 섹션별로 청킹하여 LlamaIndex Document로 변환
"""
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
from llama_index.core import Document
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# 섹션 구분 패턴 (매뉴얼에서 자주 사용되는 헤더)
SECTION_PATTERNS = [
    r'시동\s*(걸기|방법|버튼)',
    r'기어\s*(조절|변속|사용)',
    r'파킹\s*브레이크',
    r'사이드\s*브레이크',
    r'충전\s*(방법|안내|하기|구|커넥터)?',
    r'주유\s*(방법|구|안내)?',
    r'블루투스\s*(연결|설정)?',
    r'트렁크\s*(사용|열기|닫기)?',
    r'에어컨|히터',
    r'시트\s*조절',
    r'사이드\s*미러',
    r'스티어링\s*휠|핸들',
    r'오디오|AVN',
    r'하이패스',
    r'스마트키',
    r'자주\s*하는\s*질문',
    r'먼저\s*확인',
]


def load_vehicle_data(csv_path: str | Path) -> pd.DataFrame:
    """
    차종 매뉴얼 CSV 데이터 로드

    Args:
        csv_path: CSV 파일 경로

    Returns:
        DataFrame with vehicle manual data
    """
    logger.info(f"Loading vehicle manual data from {csv_path}")

    df = pd.read_csv(csv_path, dtype={'id': str})

    # 결측치 처리
    df['body'] = df['body'].fillna('')
    df['html_url'] = df['html_url'].fillna('')
    df['vehicle_name'] = df['vehicle_name'].fillna('')
    df['vehicle_type'] = df['vehicle_type'].fillna('미분류')

    logger.info(f"Loaded {len(df)} vehicle manuals")
    logger.info(f"  - Vehicle types: {df['vehicle_type'].nunique()}")
    logger.info(f"  - Unique vehicles: {df['vehicle_name'].nunique()}")

    return df


def extract_sections(text: str) -> List[Tuple[str, str]]:
    """
    텍스트에서 섹션을 추출

    Args:
        text: 매뉴얼 본문 텍스트

    Returns:
        List of (section_name, section_content) tuples
    """
    if not text or len(text.strip()) == 0:
        return [("전체", text)]

    sections = []
    current_section = "개요"
    current_content = []

    # 줄 단위로 처리
    lines = text.split('\n')

    for line in lines:
        line_stripped = line.strip()

        # 섹션 헤더 감지
        is_section_header = False
        detected_section = None

        for pattern in SECTION_PATTERNS:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                is_section_header = True
                detected_section = line_stripped[:50]  # 섹션명 제한
                break

        if is_section_header and detected_section:
            # 이전 섹션 저장
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((current_section, content))

            # 새 섹션 시작
            current_section = detected_section
            current_content = [line]
        else:
            current_content.append(line)

    # 마지막 섹션 저장
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_section, content))

    # 섹션이 너무 적으면 전체를 하나로
    if len(sections) <= 1:
        return [("전체", text)]

    return sections


def chunk_by_size(text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    텍스트를 크기 기반으로 청킹

    Args:
        text: 입력 텍스트
        max_chunk_size: 최대 청크 크기 (문자 수)
        overlap: 청크 간 오버랩

    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chunk_size

        # 문장 경계에서 자르기 시도
        if end < len(text):
            # 마침표, 물음표, 느낌표 또는 줄바꿈에서 자르기
            for sep in ['\n\n', '\n', '. ', '? ', '! ']:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start + max_chunk_size // 2:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def create_vehicle_documents(
    df: pd.DataFrame,
    chunk_by_section: bool = True,
    max_chunk_size: int = 1500
) -> List[Document]:
    """
    차종 데이터를 LlamaIndex Document로 변환

    Args:
        df: 차종 데이터 DataFrame
        chunk_by_section: 섹션별 청킹 여부
        max_chunk_size: 최대 청크 크기

    Returns:
        List of LlamaIndex Documents
    """
    logger.info(f"Creating documents from {len(df)} vehicle manuals")
    logger.info(f"  - Chunk by section: {chunk_by_section}")
    logger.info(f"  - Max chunk size: {max_chunk_size}")

    documents = []

    for idx, row in df.iterrows():
        vehicle_name = row['vehicle_name']
        vehicle_type = row['vehicle_type']
        body = row['body']
        html_url = row.get('html_url', '')
        source_id = str(row.get('id', idx))

        if not body or len(body.strip()) == 0:
            logger.warning(f"Empty body for vehicle: {vehicle_name}")
            continue

        # 기본 메타데이터
        base_metadata = {
            'vehicle_name': vehicle_name,
            'vehicle_type': vehicle_type,
            'source_id': source_id,
            'html_url': html_url,
        }

        if chunk_by_section:
            # 섹션별 청킹
            sections = extract_sections(body)

            for section_name, section_content in sections:
                # 섹션이 너무 길면 추가 청킹
                chunks = chunk_by_size(section_content, max_chunk_size)

                for chunk_idx, chunk in enumerate(chunks):
                    # 문서 텍스트 구성 (컨텍스트 포함)
                    text = f"[차종: {vehicle_name}] [{section_name}]\n\n{chunk}"

                    metadata = base_metadata.copy()
                    metadata['section'] = section_name
                    metadata['chunk_index'] = chunk_idx

                    doc = Document(
                        text=text,
                        metadata=metadata,
                        excluded_llm_metadata_keys=['source_id', 'chunk_index'],
                        excluded_embed_metadata_keys=['source_id', 'chunk_index', 'html_url']
                    )
                    documents.append(doc)
        else:
            # 전체 청킹
            chunks = chunk_by_size(body, max_chunk_size)

            for chunk_idx, chunk in enumerate(chunks):
                text = f"[차종: {vehicle_name}]\n\n{chunk}"

                metadata = base_metadata.copy()
                metadata['section'] = '전체'
                metadata['chunk_index'] = chunk_idx

                doc = Document(
                    text=text,
                    metadata=metadata,
                    excluded_llm_metadata_keys=['source_id', 'chunk_index'],
                    excluded_embed_metadata_keys=['source_id', 'chunk_index', 'html_url']
                )
                documents.append(doc)

    logger.info(f"Created {len(documents)} document chunks")

    # 차종별 청크 수 통계
    vehicle_chunk_counts = {}
    for doc in documents:
        vname = doc.metadata['vehicle_name']
        vehicle_chunk_counts[vname] = vehicle_chunk_counts.get(vname, 0) + 1

    avg_chunks = sum(vehicle_chunk_counts.values()) / len(vehicle_chunk_counts) if vehicle_chunk_counts else 0
    logger.info(f"  - Average chunks per vehicle: {avg_chunks:.1f}")

    return documents


def get_vehicle_list(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    차량 유형별 차종 목록 반환

    Args:
        df: 차종 데이터 DataFrame

    Returns:
        Dict mapping vehicle_type to list of vehicle_names
    """
    vehicle_list = {}

    for vtype in sorted(df['vehicle_type'].unique()):
        vehicles = sorted(df[df['vehicle_type'] == vtype]['vehicle_name'].unique())
        vehicle_list[vtype] = vehicles

    return vehicle_list


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 테스트
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "vehicle_manual_data.csv"

    # 데이터 로드
    df = load_vehicle_data(data_path)

    # 차종 목록
    vehicle_list = get_vehicle_list(df)
    print("\n=== 차종 목록 ===")
    for vtype, vehicles in vehicle_list.items():
        print(f"\n[{vtype}] ({len(vehicles)}개)")
        for v in vehicles[:5]:
            print(f"  - {v}")
        if len(vehicles) > 5:
            print(f"  ... 외 {len(vehicles) - 5}개")

    # 문서 생성 테스트
    documents = create_vehicle_documents(df, chunk_by_section=True)

    print(f"\n=== 문서 생성 결과 ===")
    print(f"총 청크 수: {len(documents)}")

    # 샘플 출력
    print(f"\n=== 샘플 문서 (아이오닉 5) ===")
    ioniq5_docs = [d for d in documents if d.metadata['vehicle_name'] == '아이오닉 5']
    print(f"아이오닉 5 청크 수: {len(ioniq5_docs)}")

    for i, doc in enumerate(ioniq5_docs[:3]):
        print(f"\n--- 청크 {i+1} ---")
        print(f"Section: {doc.metadata['section']}")
        print(f"Text preview: {doc.text[:200]}...")
