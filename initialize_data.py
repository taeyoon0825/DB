"""
데이터 초기화용 엔트리포인트.

이 파일은 앱 실행 전에 한 번 돌리는 용도다.
역할은 아래와 같다.

1. 샘플 이미지 다운로드
2. Chroma 임베딩 생성
3. 평가 결과 생성

즉, app.py 와 달리 "데이터 준비 작업"만 담당한다.
"""

from __future__ import annotations

import argparse
import logging

from config import ensure_data_dirs
from embed_all import build_embeddings
from evaluate import run_evaluation
from scripts.download_stl10 import download_stl10_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """초기화 CLI 진입점."""
    parser = argparse.ArgumentParser(description="이미지 검색 앱용 데모 데이터 초기화")
    parser.add_argument("--skip-download", action="store_true", help="STL-10 샘플 다운로드를 건너뜀")
    parser.add_argument("--skip-embed", action="store_true", help="임베딩 생성을 건너뜀")
    parser.add_argument("--skip-evaluate", action="store_true", help="평가 생성을 건너뜀")
    parser.add_argument(
        "--mode",
        choices=["full", "keyword", "both"],
        default="both",
        help="임베딩 생성 시 사용할 모드",
    )
    parser.add_argument(
        "--replace-existing-images",
        action="store_true",
        help="기존 샘플 이미지가 있으면 교체",
    )
    args = parser.parse_args()

    ensure_data_dirs()

    try:
        if not args.skip_download:
            total = download_stl10_sample(replace_existing=args.replace_existing_images)
            logger.info("Downloaded sample images: %s", total)

        if not args.skip_embed:
            counts = build_embeddings(mode=args.mode)
            logger.info("Embedding build finished: %s", counts)

        if not args.skip_evaluate:
            run_evaluation()
            logger.info("Evaluation finished")

        logger.info("Initialization completed successfully")
        return 0
    except Exception:
        logger.exception("Initialization failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
