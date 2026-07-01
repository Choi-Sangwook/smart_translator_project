"""번역 데이터 로딩, 문자 사전 생성, 문장 인코딩 기능을 제공하는 파일입니다."""

import pandas as pd
import torch
from torch.utils.data import Dataset
from src.config import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


def normalize_text(text: str) -> str:
    """입력 문장을 모델이 처리하기 쉬운 형태로 정리합니다."""
    # None이나 결측값이 들어오는 경우를 방지하기 위해 문자열로 변환합니다.
    text = str(text)
    # 앞뒤 공백을 제거하고, 영어 대문자는 소문자로 통일합니다.
    # 한글에는 lower()가 영향을 거의 주지 않으므로 한영 공통으로 사용할 수 있습니다.
    text = text.strip().lower()
    # 여러 개의 공백이 있을 경우 하나의 공백으로 합칩니다.
    text = " ".join(text.split())
    # 정리된 문자열을 반환합니다.
    return text


def load_translation_pairs(csv_path):
    """CSV 파일에서 영어-한국어 번역 쌍을 읽고 양방향 학습 데이터로 확장합니다."""
    # CSV 파일을 pandas DataFrame으로 읽습니다.
    df = pd.read_csv(csv_path)
    # 영어와 한국어 컬럼이 모두 존재하는지 확인합니다.
    required_columns = {"en", "ko"}
    # 필요한 컬럼이 없으면 명확한 오류 메시지를 발생시킵니다.
    if not required_columns.issubset(set(df.columns)):
        raise ValueError("CSV 파일에는 en, ko 컬럼이 반드시 있어야 합니다.")
    # 결측값이 있는 행은 번역 학습에 사용할 수 없으므로 제거합니다.
    df = df.dropna(subset=["en", "ko"])
    # 각 문장을 정리합니다.
    df["en"] = df["en"].map(normalize_text)
    df["ko"] = df["ko"].map(normalize_text)

    # 하나의 모델이 영어->한국어, 한국어->영어를 모두 학습하도록 방향 토큰을 붙입니다.
    pairs = []
    # CSV의 각 행을 순회하며 양방향 데이터를 구성합니다.
    for _, row in df.iterrows():
        # 영어를 한국어로 번역하는 학습 예시입니다.
        pairs.append(("<EN2KO> " + row["en"], row["ko"]))
        # 한국어를 영어로 번역하는 학습 예시입니다.
        pairs.append(("<KO2EN> " + row["ko"], row["en"]))
    # 전체 학습 쌍을 반환합니다.
    return pairs


def build_vocab(pairs):
    """학습 데이터에 등장하는 모든 문자를 기반으로 문자 사전을 생성합니다."""
    # 특수 토큰을 가장 앞에 배치하여 고정된 인덱스를 갖도록 합니다.
    tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    # 모든 입력 문장과 출력 문장에서 문자 단위 집합을 수집합니다.
    charset = set()
    # 학습 쌍을 반복하면서 입력과 출력의 문자를 모두 모읍니다.
    for source, target in pairs:
        # 입력 문장의 각 문자를 집합에 추가합니다.
        charset.update(list(source))
        # 출력 문장의 각 문자를 집합에 추가합니다.
        charset.update(list(target))
    # 재현 가능한 사전을 위해 문자 집합을 정렬합니다.
    sorted_chars = sorted(charset)
    # 특수 토큰 뒤에 실제 문자를 붙여 전체 토큰 목록을 만듭니다.
    vocab = tokens + sorted_chars
    # 문자 또는 특수 토큰을 정수 인덱스로 바꾸는 딕셔너리입니다.
    char2idx = {token: idx for idx, token in enumerate(vocab)}
    # 정수 인덱스를 문자 또는 특수 토큰으로 되돌리는 딕셔너리입니다.
    idx2char = {idx: token for token, idx in char2idx.items()}
    # 두 사전을 반환합니다.
    return char2idx, idx2char


def encode_text(text, char2idx, add_eos=True):
    """문자열을 정수 인덱스 리스트로 변환합니다."""
    # 사전에 없는 문자는 UNK 인덱스로 변환합니다.
    ids = [char2idx.get(ch, char2idx[UNK_TOKEN]) for ch in text]
    # 출력 문장이나 인코더 입력의 끝을 알리기 위해 EOS 토큰을 추가합니다.
    if add_eos:
        ids.append(char2idx[EOS_TOKEN])
    # 정수 리스트를 반환합니다.
    return ids


class TranslationDataset(Dataset):
    """PyTorch DataLoader가 사용할 수 있는 번역 데이터셋 클래스입니다."""

    def __init__(self, pairs, char2idx):
        # 원본 문장 쌍을 저장합니다.
        self.pairs = pairs
        # 문자 사전을 저장합니다.
        self.char2idx = char2idx

    def __len__(self):
        # 전체 데이터 개수를 반환합니다.
        return len(self.pairs)

    def __getitem__(self, index):
        # index 위치의 입력 문장과 정답 문장을 가져옵니다.
        source, target = self.pairs[index]
        # 인코더 입력은 입력 문장 뒤에 EOS를 붙입니다.
        source_ids = encode_text(source, self.char2idx, add_eos=True)
        # 디코더 입력은 SOS로 시작하고 정답 문장을 이어 붙입니다.
        decoder_input_ids = [self.char2idx[SOS_TOKEN]] + encode_text(target, self.char2idx, add_eos=False)
        # 디코더 정답은 정답 문장 뒤에 EOS를 붙입니다.
        decoder_target_ids = encode_text(target, self.char2idx, add_eos=True)
        # 학습에 필요한 세 가지 텐서를 반환합니다.
        return torch.tensor(source_ids), torch.tensor(decoder_input_ids), torch.tensor(decoder_target_ids)


def collate_batch(batch):
    """길이가 서로 다른 문장들을 한 배치에서 사용할 수 있도록 PAD로 길이를 맞춥니다."""
    # 배치에서 인코더 입력, 디코더 입력, 디코더 정답을 각각 분리합니다.
    sources, decoder_inputs, decoder_targets = zip(*batch)
    # PAD 토큰의 인덱스는 config에서 0번이 되도록 설계했습니다.
    padding_value = 0
    # 인코더 입력 문장들의 길이를 가장 긴 문장에 맞춥니다.
    sources = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=padding_value)
    # 디코더 입력 문장들의 길이를 가장 긴 문장에 맞춥니다.
    decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=padding_value)
    # 디코더 정답 문장들의 길이를 가장 긴 문장에 맞춥니다.
    decoder_targets = torch.nn.utils.rnn.pad_sequence(decoder_targets, batch_first=True, padding_value=padding_value)
    # 패딩이 완료된 배치 텐서를 반환합니다.
    return sources, decoder_inputs, decoder_targets
