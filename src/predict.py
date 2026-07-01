"""학습된 모델을 불러와서 영어 <-> 한국어 번역을 수행하는 파일입니다."""

import re
import torch

from src.config import MODEL_PATH, META_PATH, EMBED_SIZE, HIDDEN_SIZE,MAX_OUTPUT_LEN,SOS_TOKEN,UNK_TOKEN,EOS_TOKEN,DATA_PATH
from src.data_utils import normalize_text, encode_text
from src.model import Seq2SeqTranslator

def detect_language(text:str) -> str:
    """입력 문장에 한글이 포함되어 있으면 ko, 그렇지 않으면 en으로 판단합니다."""
    # 정규표현식으로 한글 음절 범위가 포함되어 있는지 검사함
    if re.search(r"[가-힣]",text):
        # 한글이 하나라도 있으면 한국어 문장으로 판단함
        return "ko"

    # 한글이 없으면 영어 문장으로 판단함
    return "en"

def build_directional_source(text: str,source_lang: str)->str:
    """영어 입력이면 한국어로 번역하라는 방향 토큰을 붙입니다."""
    if source_lang == "en":
        return "<EN2KO> " + normalize_text(text)
    # 한국어이면 영어로 번역하라는 방향 토큰을 입력 문장앞에 붙임
    return "<KO2EN> " + text

def load_model():
    """저장된 모델 가중치와 문자 사전을 불러옵니다."""
    # 모델 메타 파일이나 가중치 파일이 없으면, 학습을 먼저 실행하도록 함
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("학습된 모델 파일이 없습니다. 먼저 python -m src/train.py 명령을 실행하세요.")
    # CPU환경에서도 안전하게 불러오기 위해 map_location을 CPU로 지정함
    meta = torch.load(META_PATH,map_location="cpu")
    # 저장된 문자 -> 정수 사전을 가져옴
    char2idx = meta["char2idx"]
    # 저장된 숫자 -> 문자 사전을 가져옴
    idx2char = meta["idx2char"]
    # 저장된 사전 크기에 맞춰 모델 객체를 생성함
    model = Seq2SeqTranslator(len(char2idx),meta.get("embed_size",EMBED_SIZE),meta.get("hidden_size",HIDDEN_SIZE))
    # 학습된 가중치를 모델에 주입함
    model.load_state_dict(torch.load(MODEL_PATH,map_location="cpu"))
    # 추론에서는 dropout이나 batchnorm이 학습모드로 동작하지 않도록 평가 모드로 전환한다.
    model.eval()
    # 추론에 필요한 모델과 사전을 반환함
    return model,char2idx,idx2char

def load_exact_dictionary():
    """학습 데이터에 있는 문장을 정확한 번역을 우선으로 하기 위해 딕셔너리로 읽습니다."""
    # pandas 의족을 줄이기 위해 csv 모둘을 사용합니다.
    import csv
    # 정확 매칭 번역을 저장할 딕셔너리 생성힘
    mapping = {}
    # CSV 파일을 utf-8 인코딩으로 엽니다.
    with open(DATA_PATH,"r",encoding="utf-8") as f:
        # DictReader는 첫 줄의 en, ko 컬럼명을 기준으로 행을 딕셔너리를 읽습니다.
        reader = csv.DictReader(f)
        # 각 번역 깡을 순회함
        for row in reader:
            # 영어 문징을 정리함
            en = normalize_text(row["en"])
            # 한국어 문장을 정리함
            ko = normalize_text(row["ko"])
            # 영어 입력에 대한 한국어 번역을 등록함
            mapping[("en",en)] = ko
            # 한국어 입력에 대한 영어 번역을 등록함
            mapping[("ko",ko)] = en
    # 정확 매칭 딕셔너리를 반환함
    return mapping


def translate(text:str, model=None, char2idx=None,idx2char=None)->str:
    """입력 ㅁ문장을 자동으로 방량 판별하여 번역합니다."""
    # 빈 문장이면 번역할 수 없으므로 안내 문구를 반환함
    if not text or not text.split():
        return "번역할 문장을 입력하세요"
    # 입력 언어를 자동으로 판단함
    source_lang = detect_language(text)
    # 학습데이터 문장을 정확히 번역하기 위해 사전을 사용함
    exact_dict = load_exact_dictionary()
    # 정리된 입력 문장을 기준으로 정확 매칭을 시도함
    exact_key = (source_lang,normalize_text(text))
    # 정확 매칭 결과가 있으면 바로 반환함
    if exact_key in exact_dict:
        return exact_dict[exact_key]

    # 모델 객체가 전달되지 않았다면 저장된 모델을 불러온다.
    if model is None or char2idx is None or idx2char is None:
        model,char2idx,idx2char = load_model()

    # 번역 방향 토큰을 포함한 인코더 입력 문자열을 만듭니다.
    source_text = build_directional_source(text,source_lang)
    # 입력 문장을 정수 인덱스 리스트로 변환
    source_idx = encode_text(source_text,char2idx,add_eos=True)
    # 모델 입력 형태가 [배치, 시간]에 맞추기 위헤서 배치 차원을 추가함
    source_tensor = torch.tensor(source_idx,dtype=torch.long).unsqueeze(0)
    # 기울기 계산은 끔
    with torch.no_grad():
        # 입력 문장을 인코더가 은닉 상태로 압축함
        hidden = model.encoder(source_tensor)
        # 디코더의 첫 입력은 sos토큰임
        decoder_input = torch.tensor([[char2idx[SOS_TOKEN]]],dtype=torch.long)
        # 생성된 문자를 저장할 리스트
        result_chars = []

        # 최대 출력 길이 만큼 한글자씩 생성합니다.
        for _ in range(MAX_OUTPUT_LEN):
            # 현재까지의 hidden과 직전 글자를 바탕으로 다음 글자 점수를 계산함
            logits,hidden = model.decoder(decoder_input,hidden)
            # 가장 점수가 높은 문자 인덱스를 선택함
            next_id = int(torch.argmax(logits[:,-1,:],dim=1).item())
            # 선택된 인덱스를 문자로 변환함
            next_char = idx2char.get(next_id,UNK_TOKEN)
            # EOS가 나오면 문장 생성이 끝난 것으로 보고 반복을 중단함
            if next_char == EOS_TOKEN:
                break

            # 특수토큰은 화면에 출력하지 않습니다.
            if next_char not in {"<PAD>",SOS_TOKEN,UNK_TOKEN}:
                result_chars.append(next_char)
            # 다음 시점의 디코더 입력으로 방금 예측한 문자를 사용함
            decoder_input = torch.tensor([[next_id]],dtype=torch.long)

    # 생성된 문자를 하나의 문자열로 합침
    result = "".join(result_chars).strip()

    # 모델이 아무문자도 생성하지 못하는 경우 안내 문구를 반환함
    if not result:
        return "번역 결과를 생성하지 못했습니다. 학습데이터를 늘리거나 epoch를 증가시켜주세요."
    # 최종 번역 결과를 반환함
    return result

