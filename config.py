"""Application configuration and filesystem paths."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default.resolve()
    return Path(value).expanduser().resolve()


APP_DATA_DIR = _env_path("APP_DATA_DIR", BASE_DIR / "data")
IMAGE_DIR = _env_path("IMAGE_DIR", APP_DATA_DIR / "images")
CHROMA_FULL_DIR = _env_path("CHROMA_FULL_DIR", APP_DATA_DIR / "chroma_full")
CHROMA_KEYWORD_DIR = _env_path("CHROMA_KEYWORD_DIR", APP_DATA_DIR / "chroma_keyword")
EVAL_DIR = _env_path("EVAL_DIR", APP_DATA_DIR / "evaluation")
TEMP_DIR = _env_path("TEMP_DIR", APP_DATA_DIR / "tmp")
STL10_RAW_DIR = _env_path("STL10_RAW_DIR", APP_DATA_DIR / "_stl10_raw")
LOG_DIR = _env_path("LOG_DIR", APP_DATA_DIR / "logs")

EVAL_CSV_PATH = EVAL_DIR / "comparison_results.csv"
EVAL_JSON_PATH = EVAL_DIR / "comparison_results.json"
EVAL_CHART_PATH = EVAL_DIR / "comparison_chart.png"
TEMP_QUERY_IMAGE_PATH = TEMP_DIR / "temp_query_image.jpg"

COLLECTION_FULL = "image_full_embedding"
COLLECTION_KEYWORD = "image_keyword_embedding"

CATEGORIES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

CATEGORY_KR = {
    "airplane": "비행기",
    "bird": "새",
    "car": "자동차",
    "cat": "고양이",
    "deer": "사슴",
    "dog": "강아지",
    "horse": "말",
    "monkey": "원숭이",
    "ship": "배",
    "truck": "트럭",
}

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".svg"}

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
ENABLE_QUERY_TRANSLATION = os.getenv("ENABLE_QUERY_TRANSLATION", "true").lower() == "true"


def ensure_app_dirs() -> None:
    for path in (APP_DATA_DIR, TEMP_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_data_dirs() -> None:
    ensure_app_dirs()
    for path in (IMAGE_DIR, CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR, EVAL_DIR, STL10_RAW_DIR):
        path.mkdir(parents=True, exist_ok=True)
