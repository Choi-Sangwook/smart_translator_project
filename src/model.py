"""PyTorch 기반 문자ㅏ 단위 Seq2Seq 번역 모델을 정의 하는 파일입니다."""

import torch
import torch.nn as nn
from src.config import PAD_TOKEN

class Encoder(nn.Module):
    """입력 문장을 읽어서 문장 전체 의미를 은닉 상태로 압축하는 인코더입니다."""

    def __init__(self,vocab_size, embed_size, hidden_size):
        # 부모클래스(nn.Module)읭 초기화 기능을 실행합니다
        super().__init__()
        # 문자 인덱스를 밀집 벡터로 변환하는 임베딩 계층입니다.
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=0)

        # 순차데이터를 처리하는 GRU계층입니다.
        # batch_first=True : 입력 텐서 형태를 (배치, 시간, 특성)으로 사용하겠다는 의미임
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, source_idx):
        # 정수 인덱스 문장을 임베딩 벡터 시퀀스로 변환합니다.
        embedded = self.embedding(source_idx)
        # GRU가 입력 문장을순서대로 읽고, 마지막 은닉상태를 생성합니다.
        outputs, hidden = self.gru(embedded)
        # 디코더는 주로 마지막 hidden을 사용함으로 hidden을 반환합니다.
        return hidden


class Decoder(nn.Module):
    """인코더 은닉 상태를 기반으로 번역 문장을 한글자씩 생성하는 디코더입니다."""
    def __init__(self,vocab_size, embed_size, hidden_size):
        # 반드시 첫줄에 부모 생성자 호출함
        super().__init__()
        # 출력 언어의 문자 인덱스를 벡터로 바꾸기 위한 임베딩 계층
        self.embedding = nn.Embedding(vocab_size, embed_size,padding_idx=0)
        # 이전 글자와 이전 은닉 상태를 이용해 다음 은닉 상태를 계산하는 GRU 계증
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        # GRU출력 벡터를 전체 문자 사전 크기의 점수(logis)로 변환하는 선형계층
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input_idx, hidden):
        # 디코더 입력 글자 덱스를 임베딩 벡터로 변환하는 작업
        embedded = self.embedding(decoder_input_idx)
        # 인코더에서 전달받은 hidden을 초기 상태로 사용해서 출력 시퀀스를 계산
        outputs,hidden = self.gru(embedded,hidden)
        # 각 시점의 GRU 출력 벡터를 문자열 점수(logits)로 변환함
        logits = self.fc(outputs)
        # 학습에는 logits가 필요하고, 다음 단계 추론에는 hidden이 필요함
        return logits, hidden


class Seq2SeqTranslator(nn.Module):
    """인코더와 디코더를 하나로 묶는 전체 번역 모델"""
    def __init__(self,vocab_size, embed_size, hidden_size):
        super().__init__()
        # 입력 문장을 의미 벡터로 압축하는 인코더를 생성함
        self.encoder = Encoder(vocab_size, embed_size, hidden_size)
        # 의미 벡터로 부터 번역 문장을 생성하는 디코더를 생성함
        self.decoder = Decoder(vocab_size, embed_size, hidden_size)

    def forward(self,source_idx,decoder_input_idx):
        # 인코더가 입력 문장을 읽고 마지막 은닉 상태를 반환함
        hidden = self.encoder(source_idx)
        # 디코더가 정답 문장의 이전 글자들을 입력받아 다음글자 점수를 예측함
        logits, _ = self.decoder(decoder_input_idx,hidden)
        # 문자별 점수 텐서를 반환함]
        return logits


