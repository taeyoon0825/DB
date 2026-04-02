"""
이미지 임베딩 모듈
전체 임베딩 (CLIP 이미지 벡터) vs 키워드 임베딩 (파일명/경로 텍스트 벡터) 지원
"""
import os
import sys
import time
from typing import List, Dict, Optional

import chromadb
import torch
import open_clip
from PIL import Image
import numpy as np

from config import (
    IMAGE_DIR, CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR,
    COLLECTION_FULL, COLLECTION_KEYWORD,
    CLIP_MODEL_NAME, CLIP_PRETRAINED,
    SUPPORTED_FORMATS, CATEGORY_KR,
)


class CLIPEmbedder:
    """OpenCLIP 기반 이미지/텍스트 임베더"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"디바이스: {self.device}")
        print(f"모델 로딩: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.model.eval()
        print("모델 로딩 완료!")

    @torch.no_grad()
    def embed_image(self, image_path: str) -> np.ndarray:
        """이미지를 CLIP 벡터로 변환"""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_text(self, text: str, translate: bool = False) -> np.ndarray:
        """텍스트를 CLIP 벡터로 변환 (옵션: 한글 자동 번역)"""
        if translate:
            import re
            if bool(re.search('[가-힣]', text)):
                try:
                    from deep_translator import GoogleTranslator
                    trans_text = GoogleTranslator(source='ko', target='en').translate(text)
                    print(f"🔄 자동 번역: '{text}' ➔ '{trans_text}'")
                    text = trans_text
                except Exception as e:
                    print(f"번역 모듈 로드 실패: {e}")

        tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """여러 텍스트를 일괄 벡터화"""
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def scan_images(base_dir: str) -> List[Dict]:
    """이미지 디렉토리를 스캔하여 파일 정보 수집"""
    image_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_FORMATS and ext != ".svg":  # SVG는 CLIP으로 직접 처리 불가
                filepath = os.path.join(root, f)
                # 카테고리 = 부모 폴더명
                category = os.path.basename(root)
                category_kr = CATEGORY_KR.get(category, category)
                image_files.append({
                    "id": f"{category}_{f}",
                    "path": filepath,
                    "filename": f,
                    "category": category,
                    "category_kr": category_kr,
                    "format": ext.lstrip("."),
                    "size": os.path.getsize(filepath),
                })
    return image_files


def embed_full(embedder: CLIPEmbedder, image_files: List[Dict], chroma_dir: str):
    """전체 임베딩: 이미지 자체를 CLIP으로 벡터화하여 ChromaDB에 저장"""
    client = chromadb.PersistentClient(path=chroma_dir)

    # 기존 컬렉션 삭제 후 재생성
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

    # 배치 처리
    batch_size = 10
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for info in batch:
            try:
                vec = embedder.embed_image(info["path"])
                ids.append(info["id"])
                embeddings.append(vec.tolist())
                metadatas.append({
                    "path": info["path"],
                    "filename": info["filename"],
                    "category": info["category"],
                    "category_kr": info["category_kr"],
                    "format": info["format"],
                    "size": info["size"],
                })
                documents.append(f"{info['category_kr']} {info['category']} {info['filename']}")
            except Exception as e:
                print(f"  오류 ({info['filename']}): {e}")

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


def embed_keyword(embedder: CLIPEmbedder, image_files: List[Dict], chroma_dir: str):
    """키워드 임베딩: 파일명/경로에서 키워드를 추출하여 텍스트 벡터로 저장"""
    client = chromadb.PersistentClient(path=chroma_dir)

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
        batch = image_files[i:i + batch_size]
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for info in batch:
            # 키워드 = 카테고리(한글 + 영문) + 파일명(확장자 제외)
            name_no_ext = os.path.splitext(info["filename"])[0]
            keyword_text = f"{info['category_kr']} {info['category']} {name_no_ext}"

            try:
                vec = embedder.embed_text(keyword_text)
                ids.append(info["id"])
                embeddings.append(vec.tolist())
                metadatas.append({
                    "path": info["path"],
                    "filename": info["filename"],
                    "category": info["category"],
                    "category_kr": info["category_kr"],
                    "format": info["format"],
                    "size": info["size"],
                })
                documents.append(keyword_text)
            except Exception as e:
                print(f"  오류 ({info['filename']}): {e}")

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
