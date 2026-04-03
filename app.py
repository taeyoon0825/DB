"""
Streamlit 기반 이미지 검색 UI.

이 파일은 "런타임 UI"만 담당한다.
즉, 앱 화면을 그리고 검색을 수행하며, 데이터가 없을 때는
자동으로 데이터를 만들지 않고 사용자에게 초기화 방법만 안내한다.

실행:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

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
    IMAGE_DIR,
    TEMP_QUERY_IMAGE_PATH,
    ensure_app_dirs,
)

# Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="이미지 벡터 검색",
    page_icon="🔍",
    layout="wide",
)

# 결과 카드에 쓰는 간단한 스타일
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
    """
    검색기 객체를 캐싱해서 재사용한다.

    OpenCLIP 모델 로딩 비용이 있으므로, 같은 모드(full/keyword)에서는
    매번 새 객체를 만들지 않도록 Streamlit 캐시를 사용한다.
    """
    from searcher import ImageSearcher

    return ImageSearcher(mode=mode)


def display_results(results, cols_per_row: int = 5) -> None:
    """
    검색 결과를 그리드 형태로 보여준다.

    각 결과는:
    - 이미지 미리보기
    - 유사도
    - 카테고리
    - 파일명
    - 실제 저장 경로
    를 표시한다.
    """
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
                    image_path = resolve_display_image_path(result)
                    img = Image.open(image_path)
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
                    st.code(resolve_display_image_path(result), language=None)


def resolve_display_image_path(result: dict) -> str:
    """
    화면에 표시할 이미지 경로를 현재 실행 환경 기준으로 복원한다.

    우선순위:
    1. result["path"] 가 실제로 존재하면 그대로 사용
    2. category + filename 조합으로 IMAGE_DIR 아래 경로 복원
    3. 그래도 못 찾으면 원래 path 반환
    """
    raw_path = result.get("path", "")
    if raw_path and Path(raw_path).exists():
        return raw_path

    category = result.get("category", "")
    filename = result.get("filename", "")
    if category and filename:
        candidate = IMAGE_DIR / category / filename
        if candidate.exists():
            return str(candidate)

    return raw_path


def _get_collection_status(path, collection_name: str):
    """
    Chroma 컬렉션 상태를 점검한다.

    반환값:
    - count: 현재 컬렉션에 저장된 벡터 수
    - error: 준비되지 않았을 때 보여줄 메시지
    """
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
    """
    앱 실행에 필요한 데이터가 준비됐는지 한 번에 확인한다.

    여기서 확인하는 항목:
    - full 검색용 Chroma DB
    - keyword 검색용 Chroma DB
    - 평가 차트 / CSV 파일 존재 여부
    """
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
    """
    좌측 사이드바를 그린다.

    사용자가 여기서 검색 모드와 결과 개수를 선택한다.
    """
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
    """
    검색 데이터가 준비되지 않았을 때 사용자에게 안내 메시지를 보여준다.

    중요한 점은 여기서 "자동 초기화"를 하지 않는다는 것이다.
    앱은 UI만 담당하고, 초기화는 별도 스크립트가 담당한다.
    """
    st.error("검색 데이터가 준비되지 않았습니다.")
    st.write("아래 명령으로 초기화를 먼저 실행하세요.")
    st.code("python initialize_data.py")
    for problem in state["problems"]:
        st.write(f"- {problem}")
    st.stop()


def main() -> None:
    """
    앱 메인 진입점.

    흐름:
    1. 필요한 앱 디렉터리 생성
    2. 런타임 상태 점검
    3. 데이터가 없으면 안내 후 중단
    4. 데이터가 있으면 텍스트 검색 / 이미지 검색 / 평가 결과 탭 표시
    """
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
            # 업로드 이미지는 TEMP_DIR 아래 임시 파일로 저장한 뒤 검색에 사용한다.
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
                        results = searcher.search_by_image(
                            os.fspath(TEMP_QUERY_IMAGE_PATH),
                            n_results=n_results,
                        )
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
