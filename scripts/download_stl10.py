"""
STL-10 실제 사진 데이터셋 다운로드 및 저장
10개 카테고리 × 10장 = 100장의 실제 사진을 DB/이미지/ 에 저장합니다.

카테고리: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
"""
import os
import sys
import shutil
import random

import torch
from torchvision.datasets import STL10
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_DIR

# STL-10 카테고리 (인덱스 순서)
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

# 한글 매핑
STL10_KR = {
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

IMAGES_PER_CATEGORY = 10


def main():
    print("=" * 50)
    print("STL-10 데이터셋 다운로드 및 저장")
    print("=" * 50)

    # 기존 이미지 폴더 백업/삭제
    if os.path.exists(IMAGE_DIR):
        backup = IMAGE_DIR + "_backup_dummy"
        if os.path.exists(backup):
            shutil.rmtree(backup)
        shutil.move(IMAGE_DIR, backup)
        print(f"기존 더미 이미지 백업: {backup}")

    os.makedirs(IMAGE_DIR, exist_ok=True)

    # STL-10 다운로드
    download_dir = os.path.join(os.path.dirname(IMAGE_DIR), "_stl10_raw")
    print(f"\nSTL-10 데이터셋 다운로드 중... (약 2.5GB)")
    dataset = STL10(root=download_dir, split="train", download=True)

    # 카테고리별 인덱스 수집
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    total = 0
    for class_idx, class_name in enumerate(STL10_CLASSES):
        kr_name = STL10_KR[class_name]
        cat_dir = os.path.join(IMAGE_DIR, class_name)
        os.makedirs(cat_dir, exist_ok=True)

        # 랜덤으로 10장 선택
        indices = class_indices[class_idx]
        random.seed(42)
        selected = random.sample(indices, min(IMAGES_PER_CATEGORY, len(indices)))

        print(f"\n[{kr_name} ({class_name})] - {len(selected)}장")
        for i, idx in enumerate(selected):
            img, _ = dataset[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            # 다양한 포맷으로 저장
            if i < 4:
                ext = "jpg"
                fmt = "JPEG"
            elif i < 7:
                ext = "png"
                fmt = "PNG"
            else:
                ext = "webp"
                fmt = "WEBP"

            filename = f"{class_name}_{i+1:02d}.{ext}"
            filepath = os.path.join(cat_dir, filename)
            img.save(filepath, fmt)
            print(f"  저장: {filename} ({img.size[0]}x{img.size[1]})")
            total += 1

    print(f"\n{'=' * 50}")
    print(f"총 {total}장 실제 사진 저장 완료!")
    print(f"저장 위치: {IMAGE_DIR}")
    print(f"{'=' * 50}")
    print(f"\n다음 단계: python embed_all.py --mode both")


if __name__ == "__main__":
    main()
