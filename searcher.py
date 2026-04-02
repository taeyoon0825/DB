"""
이미지 검색 모듈
텍스트 → 이미지 검색 / 이미지 → 유사 이미지 검색
"""
import os
import argparse
from typing import List, Dict, Optional

import chromadb

from config import (
    CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR,
    COLLECTION_FULL, COLLECTION_KEYWORD,
)
from embedder import CLIPEmbedder


class ImageSearcher:
    """이미지 검색 엔진"""

    def __init__(self, mode: str = "full"):
        """
        Args:
            mode: "full" (전체 임베딩 DB) 또는 "keyword" (키워드 임베딩 DB)
        """
        self.mode = mode
        self.embedder = CLIPEmbedder()

        if mode == "full":
            chroma_dir = CHROMA_FULL_DIR
            collection_name = COLLECTION_FULL
        else:
            chroma_dir = CHROMA_KEYWORD_DIR
            collection_name = COLLECTION_KEYWORD

        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(name=collection_name)
        print(f"검색 DB 로딩: {mode} ({self.collection.count()}개 인덱싱)")

    def search_by_text(self, query: str, n_results: int = 10) -> List[Dict]:
        """텍스트 쿼리로 이미지 검색 (UTF-8 인코딩 그대로 사용)"""
        # Python3의 문자열은 기본적으로 UTF-8을 지원합니다. (translate=True 로 설정해 자동 번역 적용)
        query_vec = self.embedder.embed_text(query, translate=True).tolist()

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["metadatas", "distances", "documents"],
        )

        return self._format_results(results)

    def search_by_image(self, image_path: str, n_results: int = 10) -> List[Dict]:
        """이미지로 유사 이미지 검색"""
        query_vec = self.embedder.embed_image(image_path).tolist()

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["metadatas", "distances", "documents"],
        )

        return self._format_results(results)

    def _format_results(self, results) -> List[Dict]:
        """검색 결과 포맷팅"""
        formatted = []
        if not results["ids"][0]:
            return formatted

        for i, id_ in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # cosine distance → similarity (1 - distance)
            similarity = 1 - distance

            formatted.append({
                "id": id_,
                "similarity": round(similarity, 4),
                "path": meta.get("path", ""),
                "filename": meta.get("filename", ""),
                "category": meta.get("category", ""),
                "category_kr": meta.get("category_kr", ""),
                "format": meta.get("format", ""),
            })

        return formatted


def print_results(results: List[Dict], title: str = "검색 결과"):
    """검색 결과 출력"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

    if not results:
        print("  결과 없음")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] {r['filename']}")
        print(f"      카테고리: {r['category_kr']} ({r['category']})")
        print(f"      유사도:   {r['similarity']:.4f}")
        print(f"      경로:     {r['path']}")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="이미지 검색")
    parser.add_argument("--query", type=str, help="텍스트 검색 쿼리")
    parser.add_argument("--image", type=str, help="유사 이미지 검색용 이미지 경로")
    parser.add_argument(
        "--mode", choices=["full", "keyword"], default="full",
        help="검색 DB 모드",
    )
    parser.add_argument("--top", type=int, default=10, help="결과 개수")
    args = parser.parse_args()

    if not args.query and not args.image:
        print("--query 또는 --image 중 하나를 지정하세요.")
        return

    searcher = ImageSearcher(mode=args.mode)

    if args.query:
        results = searcher.search_by_text(args.query, n_results=args.top)
        print_results(results, f"텍스트 검색: '{args.query}' (모드: {args.mode})")
    elif args.image:
        results = searcher.search_by_image(args.image, n_results=args.top)
        print_results(results, f"이미지 검색: '{args.image}' (모드: {args.mode})")


if __name__ == "__main__":
    main()
