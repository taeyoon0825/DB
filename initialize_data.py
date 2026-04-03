"""Explicit data initialization entrypoint for local/dev environments."""

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
    parser = argparse.ArgumentParser(description="Initialize demo data for the image search app.")
    parser.add_argument("--skip-download", action="store_true", help="Skip STL-10 sample download.")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding build.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation.")
    parser.add_argument(
        "--mode",
        choices=["full", "keyword", "both"],
        default="both",
        help="Embedding mode when --skip-embed is not used.",
    )
    parser.add_argument(
        "--replace-existing-images",
        action="store_true",
        help="Replace existing image samples when downloading.",
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
