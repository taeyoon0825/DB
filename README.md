# 멀티모달 벡터 검색 시스템 (Multi-Modal Vector Search)

이 프로젝트는 오픈 소스 모델(OpenCLIP)과 벡터 데이터베이스(ChromaDB)를 활용한 멀티모달 이미지 검색 시스템입니다. 사용자는 이미지를 통해 유사한 이미지를 찾거나, 텍스트 키워드를 입력하여 관련 이미지를 검색할 수 있습니다. 

## 🚀 주요 기능
- **Full Image Embedding 모드:** 원본 이미지 전체를 인코딩하여 전체적인 분위기나 구성 요소가 유사한 이미지를 검색
- **Keyword Embedding 모드:** 이미지의 주요 키워드 특성을 기반으로 인코딩하여 검색 정확도 테스트 및 비교
- **성능 평가 (Evaluate):** Precision@K, MRR 등의 지표를 사용해 두 가지 임베딩 모드의 검색 품질을 정량적으로 평가
- **웹 UI (Streamlit):** 검색 시스템을 시각적으로 테스트할 수 있는 직관적인 인터페이스

## 📂 주요 파일 구성
- `app.py`: Streamlit으로 구동되는 웹 애플리케이션 (검색 UI)
- `embedder.py` / `embed_all.py`: STL-10 이미지 데이터셋을 불러와 OpenCLIP으로 임베딩 구조화 후 ChromaDB에 저장하는 스크립트
- `searcher.py`: 저장된 벡터 DB를 기반으로 쿼리(이미지/텍스트) 검색을 수행
- `evaluate.py`: 검색 모듈의 성능을 평가하고 결과를 분석
- `config.py`: 프로젝트 전반의 환경 변수 및 설정 값 관리

## 💻 설치 및 환경 설정

**1. 터미널에서 필수 패키지를 설치합니다.**
```bash
pip install -r requirements.txt
```

**2. 이미지 데이터 벡터화 (Vector DB 생성)**
- 모델이 이미지를 학습할 수 있도록 임베딩하여 ChromaDB에 저장합니다. (수 분 가량 소요될 수 있습니다)
```bash
python embed_all.py
```

**3. 웹 UI 서버 실행**
- 로컬 웹 브라우저에서 검색기를 테스트해보려면 아래의 명령어를 입력하세요.
```bash
streamlit run app.py
```

## 🛠 기술 스택
- Python 3.10+
- OpenCLIP
- ChromaDB
- Pytorch
- Streamlit
