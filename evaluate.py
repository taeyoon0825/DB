"""
평가 로직.

이 파일은 full 임베딩과 keyword 임베딩의 검색 성능을 비교한다.
현재 평가는 "이미지 단위 정답"이 아니라 "카테고리 정답" 기준이다.

즉, 예를 들어 정답이 ship 이면
상위 결과의 category 가 ship 인지만 보고 점수를 계산한다.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import chromadb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from config import (
    CATEGORY_KR,
    CHROMA_FULL_DIR,
    CHROMA_KEYWORD_DIR,
    COLLECTION_FULL,
    COLLECTION_KEYWORD,
    EVAL_CHART_PATH,
    EVAL_CSV_PATH,
    EVAL_DIR,
    EVAL_JSON_PATH,
)
from embedder import CLIPEmbedder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 카테고리별 테스트 문장
TEST_QUERIES = [
    {"query": "구름을 뚫고 지나가는 날개 달린 거대한 기계", "expected": "airplane"},
    {"query": "a large metal vehicle with wings soaring high", "expected": "airplane"},
    {"query": "작고 깃털이 있으며 나뭇가지에 앉아 지저귀는 동물", "expected": "bird"},
    {"query": "a small creature with colorful feathers resting on a branch", "expected": "bird"},
    {"query": "도로 위를 빠르게 주행하는 바퀴가 4개 달린 교통수단", "expected": "car"},
    {"query": "a four-wheeled vehicle traveling fast on the street", "expected": "car"},
    {"query": "수염이 길고 유연하며 꼬리가 긴 애완동물", "expected": "cat"},
    {"query": "a flexible furry animal with long whiskers and sharp eyes", "expected": "cat"},
    {"query": "머리에 나뭇가지처럼 멋진 뿔이 나 있는 숲 속의 포유류", "expected": "deer"},
    {"query": "a woodland creature with elegant antlers on its head", "expected": "deer"},
    {"query": "사람을 잘 따르고 털이 북실북실하며 꼬리를 흔드는 반려동물", "expected": "dog"},
    {"query": "a loyal furry pet with an excitedly wagging tail", "expected": "dog"},
    {"query": "갈기가 휘날리는 빠르고 튼튼한 초식 동물", "expected": "horse"},
    {"query": "a strong grazing animal with a majestic mane running fast", "expected": "horse"},
    {"query": "긴 팔로 나무 사이를 유연하게 옮겨 다니는 영장류", "expected": "monkey"},
    {"query": "a witty primate with long arms swinging through branches", "expected": "monkey"},
    {"query": "하얀 물살을 가르며 바다를 항해하는 거대한 구조물", "expected": "ship"},
    {"query": "a massive floating vessel cutting through the ocean waves", "expected": "ship"},
    {"query": "무거운 화물을 가득 싣고 화물칸이 달린 대형 차량", "expected": "truck"},
    {"query": "a heavy-duty commercial vehicle designed to carry large cargo", "expected": "truck"},
]


def configure_matplotlib_font() -> None:
    """
    OS 별로 가능한 한글 폰트를 찾아 matplotlib 에 설정한다.

    특정 OS 전용 폰트만 강제하지 않고 fallback 순서대로 찾는다.
    """
    candidates = [
        "Malgun Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
        "NanumGothic",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available_fonts:
            matplotlib.rcParams["font.family"] = candidate
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def precision_at_k(retrieved_categories: List[str], expected: str, k: int) -> float:
    """
    Precision@K 계산.

    상위 K개 중 정답 카테고리가 몇 개인지 비율로 계산한다.
    """
    top_k = retrieved_categories[:k]
    if not top_k:
        return 0.0
    return sum(1 for category in top_k if category == expected) / len(top_k)


def recall_at_k(retrieved_categories: List[str], expected: str, k: int, total_relevant: int) -> float:
    """
    Recall@K 계산.

    데이터셋 안의 전체 정답 개수 중, 상위 K개 안에 몇 개가 들어왔는지 본다.
    현재 데이터셋은 카테고리당 10장이므로 R@1 은 보통 0.1 단위로 움직인다.
    """
    if total_relevant == 0:
        return 0.0
    top_k = retrieved_categories[:k]
    return sum(1 for category in top_k if category == expected) / total_relevant


def mrr(retrieved_categories: List[str], expected: str) -> float:
    """
    MRR 계산.

    첫 정답이 몇 위에 나왔는지에 따라
    1 / rank 값을 반환한다.
    """
    for index, category in enumerate(retrieved_categories):
        if category == expected:
            return 1.0 / (index + 1)
    return 0.0


def evaluate_mode(embedder: CLIPEmbedder, chroma_dir: str | Path, collection_name: str, mode_name: str) -> Dict:
    """
    특정 모드(full 또는 keyword)의 검색 성능을 계산한다.
    """
    client = chromadb.PersistentClient(path=os.fspath(chroma_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"평가 대상 컬렉션 '{collection_name}' 을(를) 찾을 수 없습니다. "
            "먼저 `python initialize_data.py --skip-evaluate` 또는 `python embed_all.py --mode both` 를 실행하세요."
        ) from exc

    total_count = collection.count()
    if total_count == 0:
        raise RuntimeError(
            f"평가 대상 컬렉션 '{collection_name}' 이 비어 있습니다. "
            "먼저 데이터를 임베딩하세요."
        )

    print(f"\n{'=' * 50}")
    print(f" {mode_name} 평가 시작 (DB 크기: {total_count})")
    print(f"{'=' * 50}")

    results = {
        "mode": mode_name,
        "queries": [],
        "avg_precision_1": 0,
        "avg_precision_3": 0,
        "avg_precision_5": 0,
        "avg_precision_10": 0,
        "avg_recall_1": 0,
        "avg_recall_3": 0,
        "avg_recall_5": 0,
        "avg_recall_10": 0,
        "avg_mrr": 0,
    }

    # 전체 정답 개수를 계산할 때 다시 쓰기 위해 메타데이터를 한 번 가져온다.
    collection_metadata = collection.get(include=["metadatas"])["metadatas"]

    for query_info in TEST_QUERIES:
        query = query_info["query"]
        expected = query_info["expected"]

        # 질의 텍스트는 텍스트 임베딩으로 바꿔서 검색한다.
        query_vec = embedder.embed_text(query, translate=True).tolist()
        search_results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(20, total_count),
            include=["metadatas", "distances"],
        )

        # 현재 평가는 "카테고리 비교"이므로 이미지 id 가 아니라 category 만 본다.
        retrieved_categories = [
            metadata.get("category", "") for metadata in search_results["metadatas"][0]
        ]

        total_relevant = sum(
            1 for metadata in collection_metadata if metadata.get("category") == expected
        )

        p1 = precision_at_k(retrieved_categories, expected, 1)
        p3 = precision_at_k(retrieved_categories, expected, 3)
        p5 = precision_at_k(retrieved_categories, expected, 5)
        p10 = precision_at_k(retrieved_categories, expected, 10)
        r1 = recall_at_k(retrieved_categories, expected, 1, total_relevant)
        r3 = recall_at_k(retrieved_categories, expected, 3, total_relevant)
        r5 = recall_at_k(retrieved_categories, expected, 5, total_relevant)
        r10 = recall_at_k(retrieved_categories, expected, 10, total_relevant)
        mrr_value = mrr(retrieved_categories, expected)

        results["queries"].append(
            {
                "query": query,
                "expected": expected,
                "top5_retrieved": retrieved_categories[:5],
                "precision@1": p1,
                "precision@3": p3,
                "precision@5": p5,
                "precision@10": p10,
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
                "recall@10": r10,
                "mrr": mrr_value,
            }
        )

        expected_kr = CATEGORY_KR.get(expected, expected)
        top3_kr = [CATEGORY_KR.get(category, category) for category in retrieved_categories[:3]]
        hit = "O" if p1 > 0 else "X"
        print(f"  {hit} [{expected_kr}] \"{query}\" -> Top3: {top3_kr}  P@1={p1:.1f} MRR={mrr_value:.2f}")

    query_count = len(TEST_QUERIES)
    for metric in [
        "precision@1",
        "precision@3",
        "precision@5",
        "precision@10",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr",
    ]:
        key = f"avg_{metric.replace('@', '_')}"
        results[key] = sum(query_result[metric] for query_result in results["queries"]) / query_count

    print(f"\n  평균 Precision@1:  {results['avg_precision_1']:.3f}")
    print(f"  평균 Precision@5:  {results['avg_precision_5']:.3f}")
    print(f"  평균 Precision@10: {results['avg_precision_10']:.3f}")
    print(f"  평균 Recall@10:    {results['avg_recall_10']:.3f}")
    print(f"  평균 MRR:          {results['avg_mrr']:.3f}")

    return results


def save_csv(full_results: Dict, keyword_results: Dict, output_path: Path) -> None:
    """행별 평가 결과와 평균 결과를 CSV 로 저장한다."""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "모드",
                "쿼리",
                "정답 카테고리",
                "Top5 결과",
                "P@1",
                "P@3",
                "P@5",
                "P@10",
                "R@1",
                "R@3",
                "R@5",
                "R@10",
                "MRR",
            ]
        )

        for results in [full_results, keyword_results]:
            for query_result in results["queries"]:
                writer.writerow(
                    [
                        results["mode"],
                        query_result["query"],
                        query_result["expected"],
                        str(query_result["top5_retrieved"]),
                        f"{query_result['precision@1']:.3f}",
                        f"{query_result['precision@3']:.3f}",
                        f"{query_result['precision@5']:.3f}",
                        f"{query_result['precision@10']:.3f}",
                        f"{query_result['recall@1']:.3f}",
                        f"{query_result['recall@3']:.3f}",
                        f"{query_result['recall@5']:.3f}",
                        f"{query_result['recall@10']:.3f}",
                        f"{query_result['mrr']:.3f}",
                    ]
                )

        for results in [full_results, keyword_results]:
            writer.writerow(
                [
                    f"{results['mode']} (평균)",
                    "",
                    "",
                    "",
                    f"{results['avg_precision_1']:.3f}",
                    f"{results['avg_precision_3']:.3f}",
                    f"{results['avg_precision_5']:.3f}",
                    f"{results['avg_precision_10']:.3f}",
                    f"{results['avg_recall_1']:.3f}",
                    f"{results['avg_recall_3']:.3f}",
                    f"{results['avg_recall_5']:.3f}",
                    f"{results['avg_recall_10']:.3f}",
                    f"{results['avg_mrr']:.3f}",
                ]
            )

    print(f"\nCSV 저장: {output_path}")


def save_chart(full_results: Dict, keyword_results: Dict, output_path: Path) -> None:
    """비교용 막대 차트 2개를 생성한다."""
    metrics = ["Precision@1", "Precision@3", "Precision@5", "Precision@10", "MRR"]
    full_values = [
        full_results["avg_precision_1"],
        full_results["avg_precision_3"],
        full_results["avg_precision_5"],
        full_results["avg_precision_10"],
        full_results["avg_mrr"],
    ]
    keyword_values = [
        keyword_results["avg_precision_1"],
        keyword_results["avg_precision_3"],
        keyword_results["avg_precision_5"],
        keyword_results["avg_precision_10"],
        keyword_results["avg_mrr"],
    ]

    x = np.arange(len(metrics))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 차트 1: 평균 지표 비교
    ax1 = axes[0]
    bars1 = ax1.bar(
        x - width / 2,
        full_values,
        width,
        label="전체 임베딩",
        color="#4ECDC4",
        edgecolor="white",
    )
    bars2 = ax1.bar(
        x + width / 2,
        keyword_values,
        width,
        label="키워드 임베딩",
        color="#FF6B6B",
        edgecolor="white",
    )
    ax1.set_xlabel("메트릭")
    ax1.set_ylabel("점수")
    ax1.set_title("전체 임베딩 vs 키워드 임베딩 비교")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 차트 2: 카테고리별 P@1 평균 비교
    ax2 = axes[1]
    full_p1_by_category = {}
    keyword_p1_by_category = {}

    for query_result in full_results["queries"]:
        category = CATEGORY_KR.get(query_result["expected"], query_result["expected"])
        full_p1_by_category.setdefault(category, []).append(query_result["precision@1"])
    for query_result in keyword_results["queries"]:
        category = CATEGORY_KR.get(query_result["expected"], query_result["expected"])
        keyword_p1_by_category.setdefault(category, []).append(query_result["precision@1"])

    categories = sorted(full_p1_by_category.keys())
    full_category_values = [np.mean(full_p1_by_category.get(category, [0])) for category in categories]
    keyword_category_values = [
        np.mean(keyword_p1_by_category.get(category, [0])) for category in categories
    ]

    x2 = np.arange(len(categories))
    ax2.bar(
        x2 - width / 2,
        full_category_values,
        width,
        label="전체 임베딩",
        color="#4ECDC4",
        edgecolor="white",
    )
    ax2.bar(
        x2 + width / 2,
        keyword_category_values,
        width,
        label="키워드 임베딩",
        color="#FF6B6B",
        edgecolor="white",
    )
    ax2.set_xlabel("카테고리")
    ax2.set_ylabel("Precision@1")
    ax2.set_title("카테고리별 Precision@1 비교")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.set_ylim(0, 1.3)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"차트 저장: {output_path}")


def run_evaluation() -> dict:
    """
    평가 전체 흐름을 실행한다.

    1. 폰트 설정
    2. full 평가
    3. keyword 평가
    4. CSV / JSON / 차트 저장
    """
    configure_matplotlib_font()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    embedder = CLIPEmbedder()
    full_results = evaluate_mode(embedder, CHROMA_FULL_DIR, COLLECTION_FULL, "전체 임베딩 (Full)")
    keyword_results = evaluate_mode(
        embedder,
        CHROMA_KEYWORD_DIR,
        COLLECTION_KEYWORD,
        "키워드 임베딩 (Keyword)",
    )

    save_csv(full_results, keyword_results, EVAL_CSV_PATH)
    save_chart(full_results, keyword_results, EVAL_CHART_PATH)

    with open(EVAL_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump({"full": full_results, "keyword": keyword_results}, file, ensure_ascii=False, indent=2)
    print(f"JSON 저장: {EVAL_JSON_PATH}")

    print(f"\n{'=' * 50}")
    print(" 최종 비교 요약")
    print(f"{'=' * 50}")
    print(f"  {'메트릭':<16} {'전체 임베딩':>10} {'키워드 임베딩':>12} {'차이':>8}")
    print(f"  {'-' * 48}")

    for metric, key in [
        ("Precision@1", "avg_precision_1"),
        ("Precision@5", "avg_precision_5"),
        ("Precision@10", "avg_precision_10"),
        ("Recall@10", "avg_recall_10"),
        ("MRR", "avg_mrr"),
    ]:
        full_value = full_results[key]
        keyword_value = keyword_results[key]
        diff = full_value - keyword_value
        sign = "+" if diff > 0 else ""
        print(f"  {metric:<16} {full_value:>10.3f} {keyword_value:>12.3f} {sign}{diff:>7.3f}")

    return {"full": full_results, "keyword": keyword_results}


def main() -> int:
    """평가 CLI 진입점."""
    try:
        run_evaluation()
        return 0
    except Exception as exc:
        print(f"평가 실패: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
