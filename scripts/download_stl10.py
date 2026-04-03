"""
STL-10 샘플 다운로드 스크립트.

전체 STL-10 데이터셋을 직접 서비스에 쓰는 것이 아니라,
카테고리별 일부 이미지만 뽑아서 데모용 샘플 세트를 만든다.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

from PIL import Image
from torchvision.datasets import STL10

# scripts 폴더에서 상위 프로젝트 모듈을 import 하기 위한 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import IMAGE_DIR, STL10_RAW_DIR  # noqa: E402

STL10_CLASSES = [
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


def download_stl10_sample(
    image_dir: Path = IMAGE_DIR,
    download_dir: Path = STL10_RAW_DIR,
    images_per_category: int = 10,
    replace_existing: bool = False,
    backup_existing: bool = True,
) -> int:
    """
    STL-10 을 다운로드하고 카테고리별 샘플 이미지를 만든다.

    images_per_category 개수만큼만 뽑기 때문에
    전체 raw 데이터보다 훨씬 작은 데모 세트를 만들 수 있다.
    """
    image_dir = Path(image_dir)
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    has_existing_images = image_dir.exists() and any(path.is_file() for path in image_dir.rglob("*"))
    if has_existing_images:
        if not replace_existing:
            raise FileExistsError(
                f"기존 이미지 데이터가 이미 존재합니다: {image_dir} "
                "덮어쓰려면 --replace-existing 옵션을 사용하세요."
            )

        if backup_existing:
            backup_dir = image_dir.parent / f"{image_dir.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(str(image_dir), str(backup_dir))
            print(f"기존 이미지 백업: {backup_dir}")
        else:
            shutil.rmtree(image_dir)
            print(f"기존 이미지 삭제: {image_dir}")

    image_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("STL-10 데이터셋 다운로드 및 저장")
    print("=" * 50)
    print(f"다운로드 위치: {download_dir}")
    print(f"샘플 저장 위치: {image_dir}")

    dataset = STL10(root=str(download_dir), split="train", download=True)

    # 각 라벨별 데이터셋 인덱스를 모아 둔다.
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    total = 0
    for class_idx, class_name in enumerate(STL10_CLASSES):
        class_dir = image_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # 고정 시드로 샘플을 뽑아서 결과가 매번 동일하게 나오도록 한다.
        random.seed(42)
        selected_indices = random.sample(
            class_indices[class_idx],
            min(images_per_category, len(class_indices[class_idx])),
        )

        print(f"\n[{STL10_KR[class_name]} ({class_name})] - {len(selected_indices)}장")
        for image_idx, dataset_idx in enumerate(selected_indices):
            image, _ = dataset[dataset_idx]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # 일부는 jpg, 일부는 png, 일부는 webp 로 저장해서
            # 포맷 다양성도 함께 테스트한다.
            if image_idx < 4:
                ext, fmt = "jpg", "JPEG"
            elif image_idx < 7:
                ext, fmt = "png", "PNG"
            else:
                ext, fmt = "webp", "WEBP"

            file_path = class_dir / f"{class_name}_{image_idx + 1:02d}.{ext}"
            image.save(file_path, fmt)
            print(f"  저장: {file_path.name} ({image.size[0]}x{image.size[1]})")
            total += 1

    print(f"\n총 {total}장 저장 완료")
    return total


def main() -> int:
    """다운로드 CLI 진입점."""
    parser = argparse.ArgumentParser(description="STL-10 샘플 데이터 다운로드")
    parser.add_argument(
        "--images-per-category",
        type=int,
        default=10,
        help="카테고리별 저장할 이미지 수",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="기존 이미지 데이터가 있으면 교체",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="기존 이미지 교체 시 백업 폴더를 만들지 않음",
    )
    args = parser.parse_args()

    try:
        total = download_stl10_sample(
            images_per_category=args.images_per_category,
            replace_existing=args.replace_existing,
            backup_existing=not args.no_backup,
        )
        print(f"완료: {total}장")
        return 0
    except Exception as exc:
        print(f"데이터 다운로드 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
