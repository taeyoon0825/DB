"""
전체 DB 임베딩 실행 스크립트
사용법:
    python embed_all.py --mode full      # 전체 임베딩 (이미지 CLIP 벡터)
    python embed_all.py --mode keyword   # 키워드 임베딩 (메타데이터 텍스트 벡터)
    python embed_all.py --mode both      # 둘 다 실행
"""
import argparse
import sys
import os

from config import IMAGE_DIR, CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR
from embedder import CLIPEmbedder, scan_images, embed_full, embed_keyword


def main():
    parser = argparse.ArgumentParser(description="이미지 DB 임베딩")
    parser.add_argument(
        "--mode",
        choices=["full", "keyword", "both"],
        default="both",
        help="임베딩 모드: full(이미지 벡터), keyword(키워드 텍스트 벡터), both(둘 다)",
    )
    args = parser.parse_args()

    # 이미지 스캔
    print(f"이미지 스캔 중: {IMAGE_DIR}")
    image_files = scan_images(IMAGE_DIR)
    print(f"발견된 이미지: {len(image_files)}장")

    if not image_files:
        print("이미지가 없습니다. 먼저 이미지 데이터를 준비하세요. (예: scripts/download_stl10.py 실행)")
        sys.exit(1)

    # 카테고리별 통계
    from collections import Counter
    cat_counts = Counter(f["category"] for f in image_files)
    print("\n카테고리별 이미지 수:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}장")

    # CLIP 모델 로딩
    embedder = CLIPEmbedder()

    if args.mode in ("full", "both"):
        embed_full(embedder, image_files, CHROMA_FULL_DIR)

    if args.mode in ("keyword", "both"):
        embed_keyword(embedder, image_files, CHROMA_KEYWORD_DIR)

    print("\n임베딩 완료!")


if __name__ == "__main__":
    main()
