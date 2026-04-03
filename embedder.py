"""
임베딩 관련 핵심 로직.

이 파일은 다음 역할을 가진다.
- OpenCLIP 모델 로딩
- 이미지 -> 벡터 변환
- 텍스트 -> 벡터 변환
- 이미지 폴더 스캔
- full / keyword 방식으로 ChromaDB 컬렉션 생성
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List

import chromadb
import numpy as np
import open_clip
import torch
from PIL import Image

from config import (
    CATEGORY_KR,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    COLLECTION_FULL,
    COLLECTION_KEYWORD,
    ENABLE_QUERY_TRANSLATION,
    SUPPORTED_FORMATS,
)


class CLIPEmbedder:
    """OpenCLIP 기반 이미지/텍스트 임베더."""

    def __init__(self) -> None:
        # CUDA 가 있으면 GPU, 없으면 CPU를 사용한다.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"디바이스: {self.device}")
        print(f"모델 로딩: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.model.eval()
        print("모델 로딩 완료!")

    @torch.no_grad()
    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """
        이미지 파일을 CLIP 이미지 임베딩으로 변환한다.

        반환값은 정규화된 1차원 벡터다.
        """
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_text(self, text: str, translate: bool | None = None) -> np.ndarray:
        """
        텍스트를 CLIP 텍스트 임베딩으로 변환한다.

        translate=True 이면 한글을 영어로 번역한 뒤 임베딩한다.
        """
        should_translate = ENABLE_QUERY_TRANSLATION if translate is None else translate
        if should_translate and re.search(r"[가-힣]", text):
            try:
                from deep_translator import GoogleTranslator

                translated = GoogleTranslator(source="ko", target="en").translate(text)
                print(f"자동 번역: '{text}' -> '{translated}'")
                text = translated
            except Exception as exc:
                # 번역에 실패해도 검색 자체는 계속할 수 있게 원문으로 진행한다.
                print(f"번역 실패, 원문으로 계속 진행합니다: {exc}")

        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """여러 텍스트를 한 번에 벡터화한다."""
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def scan_images(base_dir: str | Path) -> List[Dict]:
    """
    이미지 디렉터리를 재귀적으로 스캔해서 메타데이터 목록을 만든다.

    여기서 만든 메타데이터는 이후 ChromaDB 저장에 그대로 사용된다.
    """
    base_path = Path(base_dir)
    image_files: List[Dict] = []

    for path in sorted(base_path.rglob("*")):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in SUPPORTED_FORMATS or ext == ".svg":
            # SVG는 현재 CLIP 이미지 임베딩 대상으로 사용하지 않는다.
            continue

        category = path.parent.name
        category_kr = CATEGORY_KR.get(category, category)
        image_files.append(
            {
                "id": f"{category}_{path.name}",
                "path": str(path.resolve()),
                "relative_path": str(Path(category) / path.name),
                "filename": path.name,
                "category": category,
                "category_kr": category_kr,
                "format": ext.lstrip("."),
                "size": path.stat().st_size,
            }
        )

    return image_files


def embed_full(embedder: CLIPEmbedder, image_files: List[Dict], chroma_dir: str | Path):
    """
    full 방식 임베딩 생성.

    각 이미지 자체를 CLIP 이미지 인코더로 벡터화해서
    ChromaDB 컬렉션에 저장한다.
    """
    client = chromadb.PersistentClient(path=os.fspath(chroma_dir))

    # 다시 생성할 때는 기존 컬렉션을 지우고 덮어쓴다.
    try:
        client.delete_collection(COLLECTION_FULL)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_FULL,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\n전체 임베딩 시작 ({len(image_files)}장)...")
    start = time.time()

    # 너무 큰 배치로 한 번에 올리면 메모리를 많이 쓰므로 나눠서 처리한다.
    batch_size = 10
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i : i + batch_size]
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for info in batch:
            try:
                vec = embedder.embed_image(info["path"])
                ids.append(info["id"])
                embeddings.append(vec.tolist())
                metadatas.append(
                    {
                        "path": info["path"],
                        "relative_path": info["relative_path"],
                        "filename": info["filename"],
                        "category": info["category"],
                        "category_kr": info["category_kr"],
                        "format": info["format"],
                        "size": info["size"],
                    }
                )
                # document 필드는 사람이 읽는 참고 텍스트 역할이다.
                documents.append(f"{info['category_kr']} {info['category']} {info['filename']}")
            except Exception as exc:
                print(f"  오류 ({info['filename']}): {exc}")

        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        print(f"  {min(i + batch_size, len(image_files))}/{len(image_files)} 완료")

    elapsed = time.time() - start
    print(f"전체 임베딩 완료! (소요시간: {elapsed:.1f}초, {collection.count()}개 인덱싱)")
    return collection


def embed_keyword(embedder: CLIPEmbedder, image_files: List[Dict], chroma_dir: str | Path):
    """
    keyword 방식 임베딩 생성.

    이미지 자체를 보는 것이 아니라,
    "카테고리명 + 파일명" 텍스트를 CLIP 텍스트 인코더로 벡터화한다.
    """
    client = chromadb.PersistentClient(path=os.fspath(chroma_dir))

    try:
        client.delete_collection(COLLECTION_KEYWORD)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_KEYWORD,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\n키워드 임베딩 시작 ({len(image_files)}장)...")
    start = time.time()

    batch_size = 10
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i : i + batch_size]
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for info in batch:
            name_no_ext = Path(info["filename"]).stem
            keyword_text = f"{info['category_kr']} {info['category']} {name_no_ext}"

            try:
                vec = embedder.embed_text(keyword_text, translate=False)
                ids.append(info["id"])
                embeddings.append(vec.tolist())
                metadatas.append(
                    {
                        "path": info["path"],
                        "relative_path": info["relative_path"],
                        "filename": info["filename"],
                        "category": info["category"],
                        "category_kr": info["category_kr"],
                        "format": info["format"],
                        "size": info["size"],
                    }
                )
                documents.append(keyword_text)
            except Exception as exc:
                print(f"  오류 ({info['filename']}): {exc}")

        if ids:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
        print(f"  {min(i + batch_size, len(image_files))}/{len(image_files)} 완료")

    elapsed = time.time() - start
    print(f"키워드 임베딩 완료! (소요시간: {elapsed:.1f}초, {collection.count()}개 인덱싱)")
    return collection
