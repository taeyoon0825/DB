"""
Image embedding utilities.

Supports:
- full embedding: image -> CLIP image embedding
- keyword embedding: metadata text -> CLIP text embedding
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
    """OpenCLIP-backed image and text embedder."""

    def __init__(self) -> None:
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
        """Convert an image file to a normalized CLIP embedding."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_text(self, text: str, translate: bool | None = None) -> np.ndarray:
        """Convert text to a normalized CLIP embedding."""
        should_translate = ENABLE_QUERY_TRANSLATION if translate is None else translate
        if should_translate and re.search(r"[가-힣]", text):
            try:
                from deep_translator import GoogleTranslator

                translated = GoogleTranslator(source="ko", target="en").translate(text)
                print(f"자동 번역: '{text}' -> '{translated}'")
                text = translated
            except Exception as exc:
                print(f"번역 실패, 원문으로 계속 진행합니다: {exc}")

        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch-embed multiple text values."""
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def scan_images(base_dir: str | Path) -> List[Dict]:
    """Scan a directory recursively and collect image metadata."""
    base_path = Path(base_dir)
    image_files: List[Dict] = []

    for path in sorted(base_path.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_FORMATS or ext == ".svg":
            continue

        category = path.parent.name
        category_kr = CATEGORY_KR.get(category, category)
        image_files.append(
            {
                "id": f"{category}_{path.name}",
                "path": str(path.resolve()),
                "filename": path.name,
                "category": category,
                "category_kr": category_kr,
                "format": ext.lstrip("."),
                "size": path.stat().st_size,
            }
        )

    return image_files


def embed_full(embedder: CLIPEmbedder, image_files: List[Dict], chroma_dir: str | Path):
    """Create full image embeddings and store them in ChromaDB."""
    client = chromadb.PersistentClient(path=os.fspath(chroma_dir))

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
                        "filename": info["filename"],
                        "category": info["category"],
                        "category_kr": info["category_kr"],
                        "format": info["format"],
                        "size": info["size"],
                    }
                )
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
    """Create metadata keyword embeddings and store them in ChromaDB."""
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
