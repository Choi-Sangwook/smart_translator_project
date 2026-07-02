"""프로젝트 전체에서 공통으로 사용하는 설정값을 관리하는 파일입니다."""

from pathlib import Path

# 현재 config.py 파일의 상위 폴더(src)의 상위 폴더를 프로젝트 루트로 지정합니다.
# 예: smart_translator_project/src/config.py -> smart_translator_project 폴더 저장 처리
BASE_DIR = Path(__file__).resolve().parent.parent

# 학습 데이터 CSV 파일 경로를 지정합니다.
DATA_PATH = BASE_DIR / "data" / "translation_pairs.csv"

# 학습된 PyTorch 모델 파일이 저장될 경로를 지정합니다.
# MODEL_PATH = BASE_DIR / "models" / "smart_translator.pt"
#transformer
MODEL_PATH = BASE_DIR / "models" / "smart_transformer_translator.pt"


# 문자 사전, 문장 최대 길이, 하이퍼파라미터 등 메타 정보를 함께 저장할 경로입니다.
# META_PATH = BASE_DIR / "models" / "translator_meta.pt"
#transformer
META_PATH = BASE_DIR / "models" / "transformer_translator_meta.pt"

# ------------------------------------------------------
# Transformer 하이퍼파라미터
# (기존 GRU용 EMBED_SIZE / HIDDEN_SIZE 는 아래 값들로 대체되었습니다.)
# ------------------------------------------------------

# 토큰 임베딩 및 모델 전체에서 사용하는 벡터 차원입니다. (노트북 실습의 d에 해당)
D_MODEL = 64

# Multi-Head Attention에서 병렬로 사용할 attention head 개수입니다.
HEADS = 4

# 인코더와 디코더를 각각 몇 개의 블록으로 쌓을지 결정합니다.
DEPTH = 2

# Feed Forward Neural Network에서 d_model을 몇 배로 확장할지 결정합니다. (노트북의 n_mlp)
FF_MULT = 4

# 위치 임베딩이 감당할 수 있는 최대 문장 길이입니다.
# 방향 토큰(<EN2KO> 등)이 문자로 펼쳐지고 EOS가 붙는 점을 고려해 여유 있게 설정합니다.
MAX_LEN = 64

# 학습 반복 횟수입니다.
# 강의교안의 MY_EPOCH 개념에 해당하며, 데이터 전체를 몇 번 반복 학습할지 결정합니다.
EPOCHS = 120

# 한 번에 학습할 데이터 묶음 크기입니다.
# 작은 데이터셋이므로 16 정도로 설정하여 빠르게 학습되도록 합니다.
BATCH_SIZE = 16

# 학습률입니다.
# 옵티마이저가 가중치를 얼마나 크게 수정할지 결정합니다.
LEARNING_RATE = 0.003

# 번역 결과를 생성할 때 최대 몇 글자까지 만들지 결정합니다.
MAX_OUTPUT_LEN = 60

# 특수 토큰입니다.
# PAD는 길이를 맞추기 위한 빈 칸, SOS는 디코더 시작, EOS는 문장 종료, UNK는 사전에 없는 문자입니다.
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
