"""
정량적 비교 평가 스크립트
전체 임베딩 vs 키워드 임베딩의 검색 품질을 메트릭으로 비교합니다.
"""
import os
import sys
import json
import csv
from typing import List, Dict

# Windows UTF-8 출력 설정
sys.stdout.reconfigure(encoding='utf-8')

import chromadb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from config import (
    BASE_DIR, CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR,
    COLLECTION_FULL, COLLECTION_KEYWORD, CATEGORY_KR,
)
from embedder import CLIPEmbedder

# 테스트 쿼리 및 정답 카테고리 (서술형/추상적 묘사 위주로 변별력 강화)
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


def precision_at_k(retrieved_categories: List[str], expected: str, k: int) -> float:
    """Precision@K: 상위 K 결과 중 정답 비율"""
    top_k = retrieved_categories[:k]
    if not top_k:
        return 0.0
    return sum(1 for c in top_k if c == expected) / len(top_k)


def recall_at_k(retrieved_categories: List[str], expected: str, k: int, total_relevant: int) -> float:
    """Recall@K: 전체 정답 중 상위 K에 포함된 비율"""
    if total_relevant == 0:
        return 0.0
    top_k = retrieved_categories[:k]
    return sum(1 for c in top_k if c == expected) / total_relevant


def mrr(retrieved_categories: List[str], expected: str) -> float:
    """MRR: 첫 번째 정답의 역수 순위"""
    for i, c in enumerate(retrieved_categories):
        if c == expected:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_mode(embedder: CLIPEmbedder, chroma_dir: str, collection_name: str, mode_name: str) -> Dict:
    """특정 임베딩 모드의 검색 품질 평가"""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_collection(name=collection_name)
    total_count = collection.count()

    print(f"\n{'='*50}")
    print(f" {mode_name} 평가 시작 (DB 크기: {total_count})")
    print(f"{'='*50}")

    results = {
        "mode": mode_name,
        "queries": [],
        "avg_precision_1": 0, "avg_precision_3": 0,
        "avg_precision_5": 0, "avg_precision_10": 0,
        "avg_recall_1": 0, "avg_recall_3": 0,
        "avg_recall_5": 0, "avg_recall_10": 0,
        "avg_mrr": 0,
    }

    for q_info in TEST_QUERIES:
        query = q_info["query"]
        expected = q_info["expected"]

        # 쿼리 벡터화 → 검색 (번역 적용)
        query_vec = embedder.embed_text(query, translate=True).tolist()
        search_results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(20, total_count),
            include=["metadatas", "distances"],
        )

        retrieved_cats = [
            m.get("category", "") for m in search_results["metadatas"][0]
        ]

        # 해당 카테고리의 전체 개수 (ground truth)
        total_relevant = sum(1 for m in collection.get(include=["metadatas"])["metadatas"]
                            if m.get("category") == expected)

        p1 = precision_at_k(retrieved_cats, expected, 1)
        p3 = precision_at_k(retrieved_cats, expected, 3)
        p5 = precision_at_k(retrieved_cats, expected, 5)
        p10 = precision_at_k(retrieved_cats, expected, 10)
        r1 = recall_at_k(retrieved_cats, expected, 1, total_relevant)
        r3 = recall_at_k(retrieved_cats, expected, 3, total_relevant)
        r5 = recall_at_k(retrieved_cats, expected, 5, total_relevant)
        r10 = recall_at_k(retrieved_cats, expected, 10, total_relevant)
        mrr_val = mrr(retrieved_cats, expected)

        results["queries"].append({
            "query": query,
            "expected": expected,
            "top5_retrieved": retrieved_cats[:5],
            "precision@1": p1, "precision@3": p3,
            "precision@5": p5, "precision@10": p10,
            "recall@1": r1, "recall@3": r3,
            "recall@5": r5, "recall@10": r10,
            "mrr": mrr_val,
        })

        expected_kr = CATEGORY_KR.get(expected, expected)
        top3_kr = [CATEGORY_KR.get(c, c) for c in retrieved_cats[:3]]
        hit = "O" if p1 > 0 else "X"
        print(f"  {hit} [{expected_kr}] \"{query}\" → Top3: {top3_kr}  P@1={p1:.1f} MRR={mrr_val:.2f}")

    # 평균 계산
    n = len(TEST_QUERIES)
    for metric in ["precision@1", "precision@3", "precision@5", "precision@10",
                    "recall@1", "recall@3", "recall@5", "recall@10", "mrr"]:
        key = f"avg_{metric.replace('@', '_')}"
        results[key] = sum(q[metric] for q in results["queries"]) / n

    print(f"\n  평균 Precision@1:  {results['avg_precision_1']:.3f}")
    print(f"  평균 Precision@5:  {results['avg_precision_5']:.3f}")
    print(f"  평균 Precision@10: {results['avg_precision_10']:.3f}")
    print(f"  평균 Recall@10:    {results['avg_recall_10']:.3f}")
    print(f"  평균 MRR:          {results['avg_mrr']:.3f}")

    return results


def save_csv(full_results: Dict, keyword_results: Dict, output_path: str):
    """결과를 CSV로 저장"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "모드", "쿼리", "정답 카테고리", "Top5 결과",
            "P@1", "P@3", "P@5", "P@10",
            "R@1", "R@3", "R@5", "R@10", "MRR",
        ])

        for results in [full_results, keyword_results]:
            for q in results["queries"]:
                writer.writerow([
                    results["mode"], q["query"], q["expected"],
                    str(q["top5_retrieved"]),
                    f"{q['precision@1']:.3f}", f"{q['precision@3']:.3f}",
                    f"{q['precision@5']:.3f}", f"{q['precision@10']:.3f}",
                    f"{q['recall@1']:.3f}", f"{q['recall@3']:.3f}",
                    f"{q['recall@5']:.3f}", f"{q['recall@10']:.3f}",
                    f"{q['mrr']:.3f}",
                ])

        # 평균 행
        for results in [full_results, keyword_results]:
            writer.writerow([
                f"{results['mode']} (평균)", "", "",
                "",
                f"{results['avg_precision_1']:.3f}", f"{results['avg_precision_3']:.3f}",
                f"{results['avg_precision_5']:.3f}", f"{results['avg_precision_10']:.3f}",
                f"{results['avg_recall_1']:.3f}", f"{results['avg_recall_3']:.3f}",
                f"{results['avg_recall_5']:.3f}", f"{results['avg_recall_10']:.3f}",
                f"{results['avg_mrr']:.3f}",
            ])

    print(f"\nCSV 저장: {output_path}")


def save_chart(full_results: Dict, keyword_results: Dict, output_path: str):
    """비교 차트 생성"""
    metrics = ["Precision@1", "Precision@3", "Precision@5", "Precision@10", "MRR"]
    full_vals = [
        full_results["avg_precision_1"], full_results["avg_precision_3"],
        full_results["avg_precision_5"], full_results["avg_precision_10"],
        full_results["avg_mrr"],
    ]
    keyword_vals = [
        keyword_results["avg_precision_1"], keyword_results["avg_precision_3"],
        keyword_results["avg_precision_5"], keyword_results["avg_precision_10"],
        keyword_results["avg_mrr"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 차트 1: Precision + MRR 비교
    ax1 = axes[0]
    bars1 = ax1.bar(x - width / 2, full_vals, width, label="전체 임베딩", color="#4ECDC4", edgecolor="white")
    bars2 = ax1.bar(x + width / 2, keyword_vals, width, label="키워드 임베딩", color="#FF6B6B", edgecolor="white")
    ax1.set_xlabel("메트릭")
    ax1.set_ylabel("점수")
    ax1.set_title("전체 임베딩 vs 키워드 임베딩 비교")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 바 위에 값 표시
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    # 차트 2: 카테고리별 P@1 비교
    ax2 = axes[1]
    categories = list(CATEGORY_KR.values())
    full_p1_by_cat = {}
    kw_p1_by_cat = {}

    for q in full_results["queries"]:
        cat = CATEGORY_KR.get(q["expected"], q["expected"])
        full_p1_by_cat.setdefault(cat, []).append(q["precision@1"])
    for q in keyword_results["queries"]:
        cat = CATEGORY_KR.get(q["expected"], q["expected"])
        kw_p1_by_cat.setdefault(cat, []).append(q["precision@1"])

    cats = sorted(full_p1_by_cat.keys())
    full_cat_vals = [np.mean(full_p1_by_cat.get(c, [0])) for c in cats]
    kw_cat_vals = [np.mean(kw_p1_by_cat.get(c, [0])) for c in cats]

    x2 = np.arange(len(cats))
    ax2.bar(x2 - width / 2, full_cat_vals, width, label="전체 임베딩", color="#4ECDC4", edgecolor="white")
    ax2.bar(x2 + width / 2, kw_cat_vals, width, label="키워드 임베딩", color="#FF6B6B", edgecolor="white")
    ax2.set_xlabel("카테고리")
    ax2.set_ylabel("Precision@1")
    ax2.set_title("카테고리별 Precision@1 비교")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats, rotation=45, ha="right")
    ax2.set_ylim(0, 1.3)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"차트 저장: {output_path}")
    plt.close()


def main():
    output_dir = os.path.join(BASE_DIR, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    # CLIP 로딩
    embedder = CLIPEmbedder()

    # 전체 임베딩 평가
    full_results = evaluate_mode(
        embedder, CHROMA_FULL_DIR, COLLECTION_FULL, "전체 임베딩 (Full)"
    )

    # 키워드 임베딩 평가
    keyword_results = evaluate_mode(
        embedder, CHROMA_KEYWORD_DIR, COLLECTION_KEYWORD, "키워드 임베딩 (Keyword)"
    )

    # 결과 저장
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    save_csv(full_results, keyword_results, csv_path)

    chart_path = os.path.join(output_dir, "comparison_chart.png")
    save_chart(full_results, keyword_results, chart_path)

    # JSON 결과도 저장
    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"full": full_results, "keyword": keyword_results}, f, ensure_ascii=False, indent=2)
    print(f"JSON 저장: {json_path}")

    print(f"\n{'='*50}")
    print(" 최종 비교 요약")
    print(f"{'='*50}")
    print(f"  {'메트릭':<16} {'전체 임베딩':>10} {'키워드 임베딩':>12} {'차이':>8}")
    print(f"  {'-'*48}")
    for metric, k1, k2 in [
        ("Precision@1", "avg_precision_1", "avg_precision_1"),
        ("Precision@5", "avg_precision_5", "avg_precision_5"),
        ("Precision@10", "avg_precision_10", "avg_precision_10"),
        ("Recall@10", "avg_recall_10", "avg_recall_10"),
        ("MRR", "avg_mrr", "avg_mrr"),
    ]:
        f_val = full_results[k1]
        k_val = keyword_results[k2]
        diff = f_val - k_val
        sign = "+" if diff > 0 else ""
        print(f"  {metric:<16} {f_val:>10.3f} {k_val:>12.3f} {sign}{diff:>7.3f}")


if __name__ == "__main__":
    main()
