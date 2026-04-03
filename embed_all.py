"""
임베딩 생성용 CLI.

이 파일은 실제 이미지 데이터가 준비된 뒤
full / keyword Chroma 컬렉션을 만드는 역할을 한다.

사용 예시:
    python embed_all.py --mode full
    python embed_all.py --mode keyword
    python embed_all.py --mode both
"""

from __future__ import annotations

import argparse
import sys

from config import CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR, IMAGE_DIR, ensure_data_dirs
from embedder import CLIPEmbedder, embed_full, embed_keyword, scan_images


def build_embeddings(mode: str = "both") -> dict[str, int]:
    """
    선택한 모드에 따라 임베딩을 생성한다.

    반환값은 각 컬렉션에 몇 개가 들어갔는지 알려주는 요약 딕셔너리다.
    """
    image_files = scan_images(IMAGE_DIR)
    print(f"이미지 스캔 중: {IMAGE_DIR}")
    print(f"발견된 이미지: {len(image_files)}장")

    if not image_files:
        raise RuntimeError(
            f"이미지가 없습니다. 먼저 데이터를 준비하세요. 현재 경로: {IMAGE_DIR}"
        )

    # 카테고리별 이미지 수를 같이 출력해서 데이터가 제대로 준비됐는지 확인한다.
    from collections import Counter

    cat_counts = Counter(file_info["category"] for file_info in image_files)
    print("\n카테고리별 이미지 수:")
    for category, count in sorted(cat_counts.items()):
        print(f"  {category}: {count}장")

    embedder = CLIPEmbedder()
    results: dict[str, int] = {}

    if mode in ("full", "both"):
        full_collection = embed_full(embedder, image_files, CHROMA_FULL_DIR)
        results["full"] = full_collection.count()

    if mode in ("keyword", "both"):
        keyword_collection = embed_keyword(embedder, image_files, CHROMA_KEYWORD_DIR)
        results["keyword"] = keyword_collection.count()

    return results


def main() -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="이미지 DB 임베딩")
    parser.add_argument(
        "--mode",
        choices=["full", "keyword", "both"],
        default="both",
        help="임베딩 모드: full, keyword, both",
    )
    args = parser.parse_args()

    ensure_data_dirs()

    try:
        results = build_embeddings(mode=args.mode)
        print("\n임베딩 완료!")
        print(f"결과: {results}")
        return 0
    except Exception as exc:
        print(f"임베딩 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
