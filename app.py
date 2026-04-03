"""
Streamlit web UI for image search.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os

import chromadb
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    CHROMA_FULL_DIR,
    CHROMA_KEYWORD_DIR,
    COLLECTION_FULL,
    COLLECTION_KEYWORD,
    EVAL_CHART_PATH,
    EVAL_CSV_PATH,
    TEMP_QUERY_IMAGE_PATH,
    ensure_app_dirs,
)

st.set_page_config(
    page_title="이미지 벡터 검색",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
<style>
    .similarity-badge {
        background: linear-gradient(135deg, #4ECDC4, #44D7B6);
        color: #000;
        padding: 2px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .category-badge {
        background: #FF6B6B;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
    }
    h1 { color: #4ECDC4 !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_searcher(mode: str):
    from searcher import ImageSearcher

    return ImageSearcher(mode=mode)


def display_results(results, cols_per_row: int = 5) -> None:
    if not results:
        st.warning("검색 결과가 없습니다.")
        return

    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(results):
                break

            result = results[idx]
            with col:
                try:
                    img = Image.open(result["path"])
                    st.image(img, use_container_width=True)
                except Exception as exc:
                    st.error(f"이미지 로드 실패: {exc}")

                similarity_pct = result["similarity"] * 100
                st.markdown(
                    f"**유사도:** <span class='similarity-badge'>{similarity_pct:.1f}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**카테고리:** <span class='category-badge'>{result['category_kr']}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"📁 {result['filename']}")
                with st.expander("전체 경로"):
                    st.code(result["path"], language=None)


def _get_collection_status(path, collection_name: str):
    if not path.exists():
        return {"count": 0, "error": f"디렉터리가 없습니다: {path}"}

    try:
        client = chromadb.PersistentClient(path=os.fspath(path))
        collection = client.get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return {"count": 0, "error": f"컬렉션이 비어 있습니다: {collection_name}"}
        return {"count": count, "error": None}
    except Exception as exc:
        return {"count": 0, "error": f"{collection_name}: {exc}"}


def get_runtime_state() -> dict:
    full_status = _get_collection_status(CHROMA_FULL_DIR, COLLECTION_FULL)
    keyword_status = _get_collection_status(CHROMA_KEYWORD_DIR, COLLECTION_KEYWORD)

    problems = []
    for status in (full_status, keyword_status):
        if status["error"]:
            problems.append(status["error"])

    return {
        "search_ready": not problems,
        "eval_ready": EVAL_CHART_PATH.exists() and EVAL_CSV_PATH.exists(),
        "problems": problems,
        "full_count": full_status["count"],
        "keyword_count": keyword_status["count"],
    }


def render_sidebar(state: dict) -> tuple[str, int]:
    with st.sidebar:
        st.header("⚙️ 설정")
        search_mode = st.selectbox(
            "임베딩 모드",
            ["full", "keyword"],
            format_func=lambda value: "전체 임베딩 (Full)" if value == "full" else "키워드 임베딩 (Keyword)",
        )
        n_results = st.slider("결과 개수", 1, 20, 10)

        st.divider()
        st.header("📊 DB 정보")
        st.metric("전체 임베딩 DB", f"{state['full_count']}개" if state["full_count"] else "미준비")
        st.metric("키워드 임베딩 DB", f"{state['keyword_count']}개" if state["keyword_count"] else "미준비")

        if state["problems"]:
            st.error("런타임 문제 감지")
            for problem in state["problems"]:
                st.write(f"- {problem}")

    return search_mode, n_results


def render_not_ready(state: dict) -> None:
    st.error("검색 데이터가 준비되지 않았습니다.")
    st.write("아래 명령으로 초기화를 먼저 실행하세요.")
    st.code("python initialize_data.py")
    for problem in state["problems"]:
        st.write(f"- {problem}")
    st.stop()


def main() -> None:
    ensure_app_dirs()
    state = get_runtime_state()

    st.title("🔍 이미지 벡터 검색 시스템")
    st.markdown("*CLIP 임베딩 기반 이미지 검색 - 전체 임베딩 vs 키워드 임베딩 비교*")

    search_mode, n_results = render_sidebar(state)

    if not state["search_ready"]:
        render_not_ready(state)

    if not state["eval_ready"]:
        st.warning(
            "평가 결과 파일이 아직 없습니다. 검색은 사용할 수 있지만 비교 차트는 비어 있을 수 있습니다. "
            "`python initialize_data.py --skip-download --skip-embed` 로 평가만 생성할 수 있습니다."
        )

    tab1, tab2, tab3 = st.tabs(["📝 텍스트 검색", "🖼️ 유사 이미지 검색", "📊 평가 결과"])

    with tab1:
        st.subheader("텍스트로 이미지 검색")
        query = st.text_input(
            "검색어를 입력하세요",
            placeholder="예: 산이 보이는 풍경, 도시 야경, cute animal...",
        )
        if st.button("🔍 검색", key="text_search") and query:
            try:
                with st.spinner("검색 중..."):
                    searcher = load_searcher(search_mode)
                    results = searcher.search_by_text(query, n_results=n_results)
                st.success(f"'{query}' 검색 완료 - {len(results)}건 (모드: {search_mode})")
                display_results(results)
            except Exception as exc:
                st.error(f"텍스트 검색 실패: {exc}")

    with tab2:
        st.subheader("이미지로 유사 이미지 검색")
        uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            TEMP_QUERY_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            img.save(TEMP_QUERY_IMAGE_PATH)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(img, caption="쿼리 이미지", use_container_width=True)

            if st.button("🔍 유사 이미지 찾기", key="img_search"):
                try:
                    with st.spinner("검색 중..."):
                        searcher = load_searcher(search_mode)
                        results = searcher.search_by_image(os.fspath(TEMP_QUERY_IMAGE_PATH), n_results=n_results)
                    with col2:
                        st.success(f"유사 이미지 {len(results)}건 검색 완료")
                    display_results(results)
                except Exception as exc:
                    st.error(f"이미지 검색 실패: {exc}")

    with tab3:
        st.subheader("전체 임베딩 vs 키워드 임베딩 비교")
        if EVAL_CHART_PATH.exists():
            st.image(os.fspath(EVAL_CHART_PATH), caption="비교 차트", use_container_width=True)
        else:
            st.info(
                "평가 차트가 없습니다. `python initialize_data.py --skip-download --skip-embed` "
                "또는 `python evaluate.py` 를 실행하세요."
            )

        if EVAL_CSV_PATH.exists():
            dataframe = pd.read_csv(EVAL_CSV_PATH)
            st.dataframe(dataframe, use_container_width=True)


if __name__ == "__main__":
    main()
