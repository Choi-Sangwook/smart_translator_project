"""Streamlit으로 구현한 스마트 번역기 화면 파일입니다."""

import sys
from pathlib import Path

# app 폴더에서 실행해도 src 패키지를 찾을 수 있도록 프로젝트 루트를 Python 경로에 추가합니다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 프로젝트 루트가 sys.path에 없으면 추가합니다.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from src.config import MODEL_PATH, META_PATH
from src.predict import detect_language, load_model, translate
from src.train import train_model


@st.cache_resource
def cached_load_or_train_model():
    """Streamlit 재실행 시마다 모델을 반복 로딩하지 않도록 캐시에 저장합니다."""
    # 모델 파일이 없으면 사용자가 바로 실행할 수 있도록 짧은 학습을 자동 수행합니다.
    if not MODEL_PATH.exists() or not META_PATH.exists():
        # 첫 실행 편의를 위해 학습을 수행하지만, 실무에서는 python -m src.train으로 미리 학습하는 것이 좋습니다.
        train_model(epochs=80)
    # 저장된 모델과 사전을 불러옵니다.
    return load_model()


# 브라우저 탭 제목과 화면 폭을 설정합니다.
st.set_page_config(page_title="Torch 스마트 번역기", page_icon="🌐", layout="centered")

# 앱 제목을 출력합니다.
st.title("🌐 Torch 기반 스마트 번역기")

# 앱 설명을 출력합니다.
st.write("영어 문장을 입력하면 한국어로, 한국어 문장을 입력하면 영어로 번역합니다.")

# 모델 상태를 화면에 표시합니다.
with st.expander("모델 상태 확인"):
    # 모델 경로를 사용자에게 보여줍니다.
    st.write(f"모델 파일: `{MODEL_PATH}`")
    # 메타 정보 경로를 사용자에게 보여줍니다.
    st.write(f"메타 파일: `{META_PATH}`")
    # 모델 파일 존재 여부를 표시합니다.
    st.write("모델 파일 존재 여부:", MODEL_PATH.exists())

# 번역할 문장을 입력받는 텍스트 영역입니다.
input_text = st.text_area(
    "번역할 문장을 입력하세요.",
    placeholder="예: hello 또는 안녕하세요",
    height=140,
)

# 입력 언어를 자동 판별하여 화면에 보여줍니다.
if input_text.strip():
    # 한글 포함 여부로 언어를 판별합니다.
    lang = detect_language(input_text)
    # 판별 결과를 사용자 친화적인 문장으로 출력합니다.
    st.info("입력 언어 판별: 한국어 → 영어 번역" if lang == "ko" else "입력 언어 판별: 영어 → 한국어 번역")

# 번역 버튼을 생성합니다.
if st.button("번역", type="primary"):
    # 빈 입력인 경우 경고 메시지를 출력합니다.
    if not input_text.strip():
        st.warning("문장을 입력한 뒤 번역 버튼을 누르세요.")
    else:
        # 모델 로딩 또는 자동 학습 중임을 알려주는 스피너를 표시합니다.
        with st.spinner("번역 중입니다..."):
            # 캐시된 모델과 문자 사전을 불러옵니다.
            model, char2idx, idx2char = cached_load_or_train_model()
            # 번역 함수를 호출하여 결과를 생성합니다.
            result = translate(input_text, model=model, char2idx=char2idx, idx2char=idx2char)
        # 번역 결과 제목을 출력합니다.
        st.subheader("번역 결과")
        # 번역 결과를 큰 박스 형태로 출력합니다.
        st.success(result)

# 사용 예시를 화면 하단에 제공합니다.
with st.expander("입력 예시"):
    # 영어 예시를 안내합니다.
    st.write("영어 → 한국어: `hello`, `thank you`, `i am a student`, `what are you doing`")
    # 한국어 예시를 안내합니다.
    st.write("한국어 → 영어: `안녕하세요`, `감사합니다`, `나는 학생입니다`, `무엇을 하고 있나요`")
