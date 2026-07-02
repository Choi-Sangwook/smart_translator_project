"""학습된 모델을 불러와 영어↔한국어 번역을 수행하는 파일입니다."""

import re
import torch
from src.config import MODEL_PATH, META_PATH, D_MODEL, HEADS, DEPTH, FF_MULT, MAX_LEN, MAX_OUTPUT_LEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, DATA_PATH
from src.data_utils import normalize_text, encode_text
from src.model import Seq2SeqTranslator


def detect_language(text: str) -> str:
    """입력 문장에 한글이 포함되어 있으면 ko, 그렇지 않으면 en으로 판단합니다."""
    # 정규표현식으로 한글 음절 범위가 포함되어 있는지 검사합니다.
    if re.search(r"[가-힣]", text):
        # 한글이 하나라도 있으면 한국어 문장으로 판단합니다.
        return "ko"
    # 한글이 없으면 영어 문장으로 판단합니다.
    return "en"


def build_directional_source(text: str, source_lang: str) -> str:
    """입력 문장 앞에 번역 방향 토큰을 붙입니다."""
    # 영어 입력이면 한국어로 번역하라는 방향 토큰을 붙입니다.
    if source_lang == "en":
        return "<EN2KO> " + normalize_text(text)
    # 한국어 입력이면 영어로 번역하라는 방향 토큰을 붙입니다.
    return "<KO2EN> " + normalize_text(text)


def load_model():
    """저장된 모델 가중치와 문자 사전을 불러옵니다."""
    # 모델 메타 파일이나 가중치 파일이 없으면 학습을 먼저 실행해야 합니다.
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("학습된 모델 파일이 없습니다. 먼저 python -m src.train 명령을 실행하세요.")
    # CPU 환경에서도 안전하게 불러오기 위해 map_location을 CPU로 지정합니다.
    meta = torch.load(META_PATH, map_location="cpu")
    # 저장된 문자→정수 사전을 가져옵니다.
    char2idx = meta["char2idx"]
    # 저장된 정수→문자 사전을 가져옵니다.
    idx2char = meta["idx2char"]
    # 저장된 사전 크기와 Transformer 하이퍼파라미터에 맞춰 모델 객체를 생성합니다.
    model = Seq2SeqTranslator(
        len(char2idx),
        d_model=meta.get("d_model", D_MODEL),
        heads=meta.get("heads", HEADS),
        depth=meta.get("depth", DEPTH),
        ff_mult=meta.get("ff_mult", FF_MULT),
        max_len=meta.get("max_len", MAX_LEN),
    )
    # 학습된 가중치를 모델에 주입합니다.
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    # 추론에서는 dropout이나 batchnorm이 학습 모드로 동작하지 않도록 평가 모드로 전환합니다.
    model.eval()
    # 추론에 필요한 모델과 사전을 반환합니다.
    return model, char2idx, idx2char


def load_exact_dictionary():
    """학습 데이터에 있는 문장은 정확한 번역을 우선 반환하기 위해 딕셔너리로 읽습니다."""
    # pandas 의존을 줄이기 위해 csv 모듈을 사용합니다.
    import csv
    # 정확 매칭 번역을 저장할 딕셔너리를 생성합니다.
    mapping = {}
    # CSV 파일을 UTF-8 인코딩으로 엽니다.
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        # DictReader는 첫 줄의 en, ko 컬럼명을 기준으로 행을 딕셔너리로 읽습니다.
        reader = csv.DictReader(f)
        # 각 번역 쌍을 순회합니다.
        for row in reader:
            # 영어 문장을 정리합니다.
            en = normalize_text(row["en"])
            # 한국어 문장을 정리합니다.
            ko = normalize_text(row["ko"])
            # 영어 입력에 대한 한국어 번역을 등록합니다.
            mapping[("en", en)] = ko
            # 한국어 입력에 대한 영어 번역을 등록합니다.
            mapping[("ko", ko)] = en
    # 정확 매칭 딕셔너리를 반환합니다.
    return mapping


def split_sentences(text: str) -> list:
    """여러 문장이 들어온 경우 문장 단위로 분리합니다.

    모델은 한 문장짜리 데이터로만 학습되어 문장 하나를 번역하면 <EOS>로 멈춥니다.
    따라서 두 문장 이상이 들어오면 각 문장을 따로 번역해야 뒷문장까지 처리할 수 있습니다.
    """
    # 마침표, 물음표, 느낌표, 줄바꿈을 문장 구분 기준으로 사용합니다.
    parts = re.split(r"[.!?\n]+", text)
    # 앞뒤 공백을 제거하고 빈 조각은 버립니다.
    sentences = [part.strip() for part in parts if part.strip()]
    # 구분자가 전혀 없으면 원문 전체를 한 문장으로 취급합니다.
    return sentences if sentences else [text.strip()]


def _translate_sentence(text: str, model, char2idx, idx2char, exact_dictionary) -> str:
    """한 문장을 번역합니다. (여러 문장 처리는 translate가 담당합니다.)"""
    # 입력 언어를 자동으로 판단합니다.
    source_lang = detect_language(text)
    # 정리된 입력 문장을 기준으로 정확 매칭을 시도합니다.
    exact_key = (source_lang, normalize_text(text))
    # 정확 매칭 결과가 있으면 바로 반환합니다.
    if exact_key in exact_dictionary:
        return exact_dictionary[exact_key]
    # 번역 방향 토큰을 포함한 인코더 입력 문자열을 만듭니다.
    source_text = build_directional_source(text, source_lang)
    # 입력 문장을 정수 인덱스 리스트로 변환합니다.
    source_ids = encode_text(source_text, char2idx, add_eos=True)
    # 모델 입력 형태 [배치, 시간]에 맞추기 위해 배치 차원을 추가합니다.
    source_tensor = torch.tensor(source_ids, dtype=torch.long).unsqueeze(0)
    # 기울기 계산을 끄면 추론 속도가 빨라지고 메모리 사용량이 줄어듭니다.
    with torch.no_grad():
        # 인코더가 입력 문장을 문맥 표현(memory)으로 변환합니다. 반복 중 재사용하기 위해 한 번만 계산합니다.
        memory = model.encoder(source_tensor)
        # 디코더 입력은 SOS 토큰 하나로 시작하며, 생성할수록 오른쪽으로 이어 붙입니다.
        # (GRU와 달리 Transformer는 hidden을 누적하지 않고, 매 스텝 지금까지의 전체 시퀀스를 다시 입력합니다.)
        generated_ids = [char2idx[SOS_TOKEN]]
        # 생성된 문자를 저장할 리스트입니다.
        result_chars = []
        # 위치 임베딩 한계를 넘지 않도록 최대 출력 길이를 max_len 이내로 제한합니다.
        max_steps = min(MAX_OUTPUT_LEN, model.max_len - 1)
        # 최대 출력 길이만큼 한 글자씩 생성합니다.
        for _ in range(max_steps):
            # 지금까지 생성한 전체 시퀀스를 [배치, 시간] 형태의 텐서로 만듭니다.
            decoder_input = torch.tensor([generated_ids], dtype=torch.long)
            # memory와 현재까지의 출력 시퀀스를 바탕으로 각 위치의 다음 글자 점수를 계산합니다.
            logits = model.decoder(decoder_input, memory)
            # 마지막 위치의 점수에서 가장 높은 문자 인덱스를 선택합니다.
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            # 선택된 인덱스를 문자로 변환합니다.
            next_char = idx2char.get(next_id, UNK_TOKEN)
            # EOS가 나오면 문장 생성이 끝났다는 의미이므로 반복을 중단합니다.
            if next_char == EOS_TOKEN:
                break
            # 특수 토큰은 화면에 출력하지 않습니다.
            if next_char not in {"<PAD>", SOS_TOKEN, UNK_TOKEN}:
                result_chars.append(next_char)
            # 방금 예측한 문자를 다음 스텝의 디코더 입력에 이어 붙입니다.
            generated_ids.append(next_id)
    # 생성된 문자들을 하나의 문자열로 합칩니다.
    result = "".join(result_chars).strip()
    # 아무 문자도 생성하지 못하면 빈 문자열을 반환하여 상위 translate가 처리하도록 합니다.
    return result


def translate(text: str, model=None, char2idx=None, idx2char=None) -> str:
    """입력을 문장 단위로 분리한 뒤 각 문장을 번역하여 다시 합칩니다."""
    # 빈 문장은 번역할 수 없으므로 안내 문구를 반환합니다.
    if not text or not text.strip():
        return "번역할 문장을 입력하세요."
    # 모델 객체가 전달되지 않았다면 저장된 모델을 한 번만 불러옵니다.
    if model is None or char2idx is None or idx2char is None:
        model, char2idx, idx2char = load_model()
    # 정확 매칭 사전을 한 번만 읽어 모든 문장에서 재사용합니다.
    exact_dictionary = load_exact_dictionary()
    # 입력을 문장 단위로 분리합니다.
    sentences = split_sentences(text)
    # 각 문장을 개별적으로 번역합니다.
    translated = [_translate_sentence(sentence, model, char2idx, idx2char, exact_dictionary) for sentence in sentences]
    # 비어 있지 않은 번역 결과만 남깁니다.
    translated = [part for part in translated if part]
    # 모든 문장에서 아무 결과도 얻지 못하면 안내 문구를 반환합니다.
    if not translated:
        return "번역 결과를 생성하지 못했습니다. 학습 데이터를 늘리거나 epoch를 증가시켜 주세요."
    # 번역된 문장들을 공백으로 이어 최종 결과를 만듭니다.
    return " ".join(translated)
