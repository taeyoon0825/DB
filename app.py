"""
Streamlit 기반 이미지 검색 웹 UI
실행: streamlit run app.py
"""
import os
import streamlit as st
from PIL import Image

from config import (
    BASE_DIR, IMAGE_DIR,
    CHROMA_FULL_DIR, CHROMA_KEYWORD_DIR,
    COLLECTION_FULL, COLLECTION_KEYWORD,
    CATEGORY_KR,
)

st.set_page_config(
    page_title="이미지 벡터 검색",
    page_icon="🔍",
    layout="wide",
)

# CSS 스타일
st.markdown("""
<style>
    .result-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #333;
    }
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
""", unsafe_allow_html=True)


@st.cache_resource
def load_searcher(mode):
    """검색 엔진 캐싱 로드"""
    from searcher import ImageSearcher
    return ImageSearcher(mode=mode)


def display_results(results, cols_per_row=5):
    """검색 결과를 그리드로 표시"""
    if not results:
        st.warning("검색 결과가 없습니다.")
        return

    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(results):
                break
            r = results[idx]
            with col:
                # 이미지 표시
                try:
                    img = Image.open(r["path"])
                    st.image(img, use_column_width=True)
                except Exception:
                    st.error("이미지 로드 실패")

                # 정보 표시
                sim_pct = r["similarity"] * 100
                st.markdown(
                    f"**유사도:** <span class='similarity-badge'>{sim_pct:.1f}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**카테고리:** <span class='category-badge'>{r['category_kr']}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"📁 {r['filename']}")
                with st.expander("전체 경로"):
                    st.code(r["path"], language=None)


def main():
    import sys
    import subprocess
    if not os.path.exists(CHROMA_FULL_DIR) or not os.path.exists(IMAGE_DIR):
        with st.spinner("서버 최초 세팅 중... 더미 이미지 및 검색용 벡터 DB를 생성합니다. (약 1~2분 소요)"):
            try:
                subprocess.run([sys.executable, "scripts/download_stl10.py"], check=False)
                subprocess.run([sys.executable, "embed_all.py", "--mode", "both"], check=False)
                st.success("초기 데모 데이터 세팅 완료!")
            except Exception as e:
                st.error(f"데이터 생성 실패: {e}")

    st.title("🔍 이미지 벡터 검색 시스템")
    st.markdown("*CLIP 임베딩 기반 이미지 검색 — 전체 임베딩 vs 키워드 임베딩 비교*")

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        search_mode = st.selectbox(
            "임베딩 모드",
            ["full", "keyword"],
            format_func=lambda x: "전체 임베딩 (Full)" if x == "full" else "키워드 임베딩 (Keyword)",
        )
        n_results = st.slider("결과 개수", 1, 20, 10)

        st.divider()
        st.header("📊 DB 정보")
        import chromadb
        try:
            client = chromadb.PersistentClient(path=CHROMA_FULL_DIR)
            col = client.get_collection(COLLECTION_FULL)
            st.metric("전체 임베딩 DB", f"{col.count()}개")
        except Exception:
            st.metric("전체 임베딩 DB", "미생성")
        try:
            client = chromadb.PersistentClient(path=CHROMA_KEYWORD_DIR)
            col = client.get_collection(COLLECTION_KEYWORD)
            st.metric("키워드 임베딩 DB", f"{col.count()}개")
        except Exception:
            st.metric("키워드 임베딩 DB", "미생성")

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📝 텍스트 검색", "🖼️ 유사 이미지 검색", "📊 평가 결과"])

    with tab1:
        st.subheader("텍스트로 이미지 검색")
        query = st.text_input(
            "검색어를 입력하세요",
            placeholder="예: 산이 보이는 풍경, 도시 야경, cute animal...",
        )
        if st.button("🔍 검색", key="text_search") and query:
            with st.spinner("검색 중..."):
                searcher = load_searcher(search_mode)
                results = searcher.search_by_text(query, n_results=n_results)

                st.success(f"'{query}' 검색 완료 — {len(results)}건 (모드: {search_mode})")
                display_results(results)

    with tab2:
        st.subheader("이미지로 유사 이미지 검색")
        uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            # 임시 저장
            temp_path = os.path.join(BASE_DIR, "temp_query_image.jpg")
            img = Image.open(uploaded).convert("RGB")
            img.save(temp_path)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(img, caption="쿼리 이미지", use_column_width=True)

            if st.button("🔍 유사 이미지 찾기", key="img_search"):
                with st.spinner("검색 중..."):
                    searcher = load_searcher(search_mode)
                    results = searcher.search_by_image(temp_path, n_results=n_results)
                    with col2:
                        st.success(f"유사 이미지 {len(results)}건 검색 완료")
                    display_results(results)

    with tab3:
        st.subheader("전체 임베딩 vs 키워드 임베딩 비교")

        chart_path = os.path.join(BASE_DIR, "evaluation_results", "comparison_chart.png")
        csv_path = os.path.join(BASE_DIR, "evaluation_results", "comparison_results.csv")

        if os.path.exists(chart_path):
            st.image(chart_path, caption="비교 차트", use_column_width=True)
        else:
            st.info("아직 평가가 실행되지 않았습니다. `python evaluate.py`를 먼저 실행하세요.")

        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            st.dataframe(df)


if __name__ == "__main__":
    main()
