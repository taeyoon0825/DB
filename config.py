"""
프로젝트 전역 설정 모음.

이 파일은 다음을 담당한다.
- 앱 기본 경로 계산
- 환경변수 우선 경로 설정
- 컬렉션 이름, 카테고리명, 모델명 같은 상수 정의
- 앱 실행 전 필요한 디렉터리 생성 함수 제공
"""

from __future__ import annotations

import os
from pathlib import Path

# 현재 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    """
    환경변수가 있으면 그 값을 경로로 사용하고,
    없으면 기본 경로를 사용한다.
    """
    value = os.getenv(name)
    if not value:
        return default.resolve()
    return Path(value).expanduser().resolve()


# 데이터 저장 경로들
APP_DATA_DIR = _env_path("APP_DATA_DIR", BASE_DIR / "data")
IMAGE_DIR = _env_path("IMAGE_DIR", APP_DATA_DIR / "images")
CHROMA_FULL_DIR = _env_path("CHROMA_FULL_DIR", APP_DATA_DIR / "chroma_full")
CHROMA_KEYWORD_DIR = _env_path("CHROMA_KEYWORD_DIR", APP_DATA_DIR / "chroma_keyword")
EVAL_DIR = _env_path("EVAL_DIR", APP_DATA_DIR / "evaluation")
TEMP_DIR = _env_path("TEMP_DIR", APP_DATA_DIR / "tmp")
STL10_RAW_DIR = _env_path("STL10_RAW_DIR", APP_DATA_DIR / "_stl10_raw")
LOG_DIR = _env_path("LOG_DIR", APP_DATA_DIR / "logs")

# 평가 결과와 임시 파일 경로
EVAL_CSV_PATH = EVAL_DIR / "comparison_results.csv"
EVAL_JSON_PATH = EVAL_DIR / "comparison_results.json"
EVAL_CHART_PATH = EVAL_DIR / "comparison_chart.png"
TEMP_QUERY_IMAGE_PATH = TEMP_DIR / "temp_query_image.jpg"

# Chroma 컬렉션 이름
COLLECTION_FULL = "image_full_embedding"
COLLECTION_KEYWORD = "image_keyword_embedding"

# 데모 데이터 카테고리
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

# 화면 표시에 사용하는 한글 카테고리 이름
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

# 이미지 스캔 시 허용할 확장자
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".svg"}

# OpenCLIP 모델 설정
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

# 한글 검색어를 자동 번역할지 여부
ENABLE_QUERY_TRANSLATION = os.getenv("ENABLE_QUERY_TRANSLATION", "true").lower() == "true"


def ensure_app_dirs() -> None:
    """
    앱 런타임에 필요한 최소 디렉터리를 만든다.

    앱 실행만 해도 필요한 경로:
    - APP_DATA_DIR
    - TEMP_DIR
    - LOG_DIR
    """
    for path in (APP_DATA_DIR, TEMP_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_data_dirs() -> None:
    """
    데이터 초기화에 필요한 전체 디렉터리를 만든다.

    initialize_data.py 에서 주로 사용한다.
    """
    ensure_app_dirs()
    for path in (IMAGE_DIR, CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR, EVAL_DIR, STL10_RAW_DIR):
        path.mkdir(parents=True, exist_ok=True)
