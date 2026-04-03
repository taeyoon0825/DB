"""
Image search module.

Supports:
- text -> image search
- image -> similar image search
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import chromadb

from config import (
    CHROMA_FULL_DIR,
    CHROMA_KEYWORD_DIR,
    COLLECTION_FULL,
    COLLECTION_KEYWORD,
)
from embedder import CLIPEmbedder


class ImageSearcher:
    """Search engine backed by ChromaDB and OpenCLIP."""

    def __init__(self, mode: str = "full") -> None:
        if mode not in {"full", "keyword"}:
            raise ValueError(f"지원하지 않는 검색 모드입니다: {mode}")

        self.mode = mode

        if mode == "full":
            chroma_dir = CHROMA_FULL_DIR
            collection_name = COLLECTION_FULL
        else:
            chroma_dir = CHROMA_KEYWORD_DIR
            collection_name = COLLECTION_KEYWORD

        self.client = chromadb.PersistentClient(path=os.fspath(chroma_dir))
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"검색 DB '{collection_name}' 을(를) 찾을 수 없습니다. "
                "먼저 `python initialize_data.py` 를 실행하세요."
            ) from exc

        self.collection_count = self.collection.count()
        if self.collection_count == 0:
            raise RuntimeError(
                f"검색 DB '{collection_name}' 이 비어 있습니다. "
                "먼저 `python initialize_data.py` 를 실행하세요."
            )

        self.embedder = CLIPEmbedder()
        print(f"검색 DB 로딩: {mode} ({self.collection_count}개 인덱싱)")

    def search_by_text(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search images using a text query."""
        query = query.strip()
        if not query:
            raise ValueError("빈 검색어는 사용할 수 없습니다.")

        query_vec = self.embedder.embed_text(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(n_results, self.collection_count),
            include=["metadatas", "distances", "documents"],
        )
        return self._format_results(results)

    def search_by_image(self, image_path: str, n_results: int = 10) -> List[Dict]:
        """Search similar images using an input image."""
        query_vec = self.embedder.embed_image(image_path).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(n_results, self.collection_count),
            include=["metadatas", "distances", "documents"],
        )
        return self._format_results(results)

    def _format_results(self, results) -> List[Dict]:
        formatted: List[Dict] = []
        if not results["ids"][0]:
            return formatted

        for index, item_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][index]
            distance = results["distances"][0][index]
            similarity = 1 - distance

            formatted.append(
                {
                    "id": item_id,
                    "similarity": round(similarity, 4),
                    "path": metadata.get("path", ""),
                    "filename": metadata.get("filename", ""),
                    "category": metadata.get("category", ""),
                    "category_kr": metadata.get("category_kr", ""),
                    "format": metadata.get("format", ""),
                }
            )

        return formatted


def print_results(results: List[Dict], title: str = "검색 결과") -> None:
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")

    if not results:
        print("  결과 없음")
        return

    for index, result in enumerate(results, 1):
        print(f"\n  [{index}] {result['filename']}")
        print(f"      카테고리: {result['category_kr']} ({result['category']})")
        print(f"      유사도:   {result['similarity']:.4f}")
        print(f"      경로:     {result['path']}")

    print(f"\n{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser(description="이미지 검색")
    parser.add_argument("--query", type=str, help="텍스트 검색 쿼리")
    parser.add_argument("--image", type=str, help="유사 이미지 검색용 이미지 경로")
    parser.add_argument(
        "--mode",
        choices=["full", "keyword"],
        default="full",
        help="검색 DB 모드",
    )
    parser.add_argument("--top", type=int, default=10, help="결과 개수")
    args = parser.parse_args()

    if not args.query and not args.image:
        print("--query 또는 --image 중 하나를 지정하세요.")
        return 1

    try:
        searcher = ImageSearcher(mode=args.mode)
        if args.query:
            results = searcher.search_by_text(args.query, n_results=args.top)
            print_results(results, f"텍스트 검색: '{args.query}' (모드: {args.mode})")
        else:
            results = searcher.search_by_image(args.image, n_results=args.top)
            print_results(results, f"이미지 검색: '{args.image}' (모드: {args.mode})")
        return 0
    except Exception as exc:
        print(f"검색 실패: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
