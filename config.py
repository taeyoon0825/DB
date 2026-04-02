"""프로젝트 설정"""
import os

# 기본 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "이미지")

# ChromaDB 경로
CHROMA_FULL_DIR = os.path.join(BASE_DIR, "chroma_full")
CHROMA_KEYWORD_DIR = os.path.join(BASE_DIR, "chroma_keyword")

# 컬렉션 이름
COLLECTION_FULL = "image_full_embedding"
COLLECTION_KEYWORD = "image_keyword_embedding"

# 이미지 카테고리
CATEGORIES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# 카테고리 한글 매핑
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

# 지원 포맷
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".svg"}

# 임베딩 모델
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
