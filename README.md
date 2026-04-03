# 멀티모달 이미지 검색 데모

OpenCLIP + ChromaDB + Streamlit 기반 이미지 검색 데모입니다.

기능:
- 텍스트 -> 이미지 검색
- 이미지 -> 유사 이미지 검색
- 전체 임베딩 vs 키워드 임베딩 비교 평가

이 프로젝트는 앱 런타임과 데이터 초기화를 분리했습니다.
- 앱 실행: 검색 UI만 담당
- 데이터 초기화: `initialize_data.py`

## 요구 사항

- Python 3.11 권장
- 전역 Python 대신 가상환경 사용 권장
- 운영 환경에서는 쓰기 가능한 데이터 디렉터리 필요

## 1. 가상환경 생성

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. 의존성 설치

```bash
pip install -r requirements.txt
```

개발 도구까지 설치하려면:

```bash
pip install -r requirements-dev.txt
```

## 3. 데이터 초기화

앱 실행 전에 한 번만 수행하면 됩니다.

```bash
python initialize_data.py
```

옵션:

```bash
python initialize_data.py --skip-download
python initialize_data.py --skip-evaluate
python initialize_data.py --mode full
python initialize_data.py --replace-existing-images
```

## 4. 앱 실행

```bash
streamlit run app.py
```

데이터가 준비되지 않았으면 앱이 조용히 실패하지 않고 초기화 명령을 안내합니다.

## 5. 환경변수

기본적으로 프로젝트 내부 `./data` 경로를 사용합니다.

- `APP_DATA_DIR`
- `IMAGE_DIR`
- `CHROMA_FULL_DIR`
- `CHROMA_KEYWORD_DIR`
- `EVAL_DIR`
- `TEMP_DIR`
- `STL10_RAW_DIR`
- `LOG_DIR`
- `ENABLE_QUERY_TRANSLATION`
- `CLIP_MODEL_NAME`
- `CLIP_PRETRAINED`

예시:

### Windows PowerShell
```powershell
$env:APP_DATA_DIR="D:\image-search-data"
```

### macOS / Linux
```bash
export APP_DATA_DIR=/srv/image-search-data
```

## 6. Docker

초기 데이터 생성:

```bash
docker compose run --rm init-data
```

앱 실행:

```bash
docker compose up app
```

기본 볼륨:

- 호스트 `./app_data`
- 컨테이너 `/data`

## 7. 주요 파일

- `app.py`: Streamlit UI
- `config.py`: 환경변수 기반 경로/상수 설정
- `initialize_data.py`: 다운로드/임베딩/평가 초기화
- `embed_all.py`: ChromaDB 임베딩 생성
- `embedder.py`: OpenCLIP 임베딩 로직
- `searcher.py`: 검색 엔진
- `evaluate.py`: Precision/Recall/MRR 평가
- `scripts/download_stl10.py`: STL-10 샘플 다운로드

## 8. 배포 시 주의

- `initialize_data.py` 는 앱과 분리해서 실행하세요.
- 앱 컨테이너/서버는 쓰기 가능한 `TEMP_DIR` 과 읽기 가능한 데이터 경로가 필요합니다.
- 번역 기능은 외부 네트워크에 의존할 수 있으므로, 제한된 환경에서는 `ENABLE_QUERY_TRANSLATION=false` 로 비활성화할 수 있습니다.
