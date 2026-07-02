"""PyTorch 기반 문자 단위 Transformer 인코더-디코더 번역 모델을 정의하는 파일입니다."""

import torch
import torch.nn as nn
# softmax와 같은 함수형 신경망 연산을 사용하기 위해 torch.nn.functional을 F라는 별칭으로 불러옵니다.
import torch.nn.functional as F
# 수치 계산과 배열 처리를 위해 NumPy를 불러옵니다.
import numpy as np


# MultiHeadAttention 클래스는 노트북 실습의 SelfAttention을 확장한 것으로,
# (1) 서로 다른 입력에서 Query와 Key/Value를 만드는 cross-attention과
# (2) 미래 토큰을 가리는 causal mask를 함께 지원합니다.
class MultiHeadAttention(nn.Module):
    # __init__ 메서드는 Query/Key/Value 선형 변환 계층을 초기화합니다.
    def __init__(self, d: int, heads: int = 8):
        # 부모 클래스인 nn.Module의 초기화 기능을 실행합니다.
        super().__init__()

        # d는 한 토큰을 표현하는 임베딩 벡터의 차원 수입니다.
        self.d = d

        # heads는 병렬로 사용할 attention head의 개수입니다.
        self.h = heads

        # 입력을 Query로 변환하는 선형 계층입니다. (노트북과 동일하게 head별 차원을 d로 둡니다.)
        self.WQ = nn.Linear(d, d * heads, bias=False)

        # 입력을 Key로 변환하는 선형 계층입니다.
        self.WK = nn.Linear(d, d * heads, bias=False)

        # 입력을 Value로 변환하는 선형 계층입니다.
        self.WV = nn.Linear(d, d * heads, bias=False)

        # 여러 head 결과를 다시 하나의 d차원 벡터로 합치는 선형 계층입니다.
        self.unifyheads = nn.Linear(heads * d, d)

    # forward 메서드는 Query 입력과 (선택적으로 다른) Key/Value 입력을 받아 attention 결과를 반환합니다.
    def forward(self, x_query: torch.Tensor, x_context: torch.Tensor = None, causal_mask: bool = False) -> torch.Tensor:
        # x_context가 주어지지 않으면 자기 자신을 참고하는 self-attention으로 동작합니다.
        if x_context is None:
            x_context = x_query

        # Query 입력의 크기는 (배치 b, Query 길이 lq, 임베딩 차원 d)입니다.
        b, lq, d = x_query.size()

        # Key/Value 입력의 길이 lk입니다. cross-attention에서는 lq와 다를 수 있습니다.
        lk = x_context.size(1)

        # head 개수를 지역 변수로 저장합니다.
        h = self.h

        # Query를 (b*h, lq, d) 형태로 변환하여 각 head를 독립 배치처럼 계산합니다.
        queries = self.WQ(x_query).view(b, lq, h, d).transpose(1, 2).contiguous().view(b * h, lq, d)

        # Key를 (b*h, lk, d) 형태로 변환합니다.
        keys = self.WK(x_context).view(b, lk, h, d).transpose(1, 2).contiguous().view(b * h, lk, d)

        # Value를 (b*h, lk, d) 형태로 변환합니다.
        values = self.WV(x_context).view(b, lk, h, d).transpose(1, 2).contiguous().view(b * h, lk, d)

        # Query와 Key의 전치 행렬을 배치 행렬곱하여 토큰 간 관련도 점수 (b*h, lq, lk)를 구합니다.
        # sqrt(d)로 나누어 점수 크기를 안정화하고 softmax 쏠림을 줄입니다.
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)

        # 디코더의 self-attention처럼 미래를 보면 안 되는 경우 causal mask를 적용합니다.
        if causal_mask:
            # 대각선 위쪽(미래 위치)을 True로 만드는 마스크를 생성합니다.
            mask = torch.triu(torch.ones(lq, lk, device=attention_scores.device), diagonal=1).bool()

            # 미래 위치의 점수를 -무한대로 바꾸어 softmax 이후 0이 되도록 합니다.
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # 마지막 차원 기준 softmax로 각 토큰이 다른 토큰을 얼마나 참고할지 확률로 만듭니다.
        attention_weights = F.softmax(attention_scores, dim=-1)

        # attention 가중치를 Value에 곱하여 문맥이 반영된 표현을 계산합니다.
        out = torch.bmm(attention_weights, values).view(b, h, lq, d)

        # head 축과 길이 축을 되돌리고 모든 head 결과를 이어 붙여 (b, lq, h*d)로 만듭니다.
        out = out.transpose(1, 2).contiguous().view(b, lq, h * d)

        # 이어 붙인 multi-head 결과를 선형 계층으로 다시 d차원으로 통합합니다.
        return self.unifyheads(out)


# EncoderBlock 클래스는 하나의 Transformer 인코더 블록입니다. (노트북의 TransformerBlock과 동일한 구조)
class EncoderBlock(nn.Module):
    # __init__ 메서드는 Self-Attention, LayerNorm, FFNN 계층을 초기화합니다.
    def __init__(self, d: int, heads: int = 8, n_mlp: int = 4):
        # 부모 클래스의 초기화 기능을 실행합니다.
        super().__init__()

        # 입력 토큰 간 문맥 관계를 계산하는 self-attention입니다.
        self.attention = MultiHeadAttention(d, heads=heads)

        # attention 결과와 입력을 더한 뒤 분포를 안정화하는 LayerNorm입니다.
        self.norm1 = nn.LayerNorm(d)

        # FFNN 결과와 입력을 더한 뒤 분포를 안정화하는 LayerNorm입니다.
        self.norm2 = nn.LayerNorm(d)

        # 각 토큰 위치마다 독립적으로 적용되는 Feed Forward Neural Network입니다.
        self.ff = nn.Sequential(
            # d차원을 n_mlp*d차원으로 확장합니다.
            nn.Linear(d, n_mlp * d),
            # 비선형성을 추가합니다.
            nn.ReLU(),
            # 다시 d차원으로 축소하여 입력과 크기를 맞춥니다.
            nn.Linear(n_mlp * d, d),
        )

    # forward 메서드는 입력 x를 인코더 블록에 통과시킵니다.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention 결과를 입력과 더하고(Residual) LayerNorm을 적용합니다.
        x = self.norm1(self.attention(x) + x)

        # FFNN 결과를 입력과 더하고 LayerNorm을 적용하여 블록 출력을 반환합니다.
        x = self.norm2(self.ff(x) + x)

        # 최종 블록 출력을 반환합니다.
        return x


# DecoderBlock 클래스는 하나의 Transformer 디코더 블록입니다.
# 인코더 블록과 달리 (1) 미래를 가리는 masked self-attention과 (2) 인코더 출력을 참고하는 cross-attention을 가집니다.
class DecoderBlock(nn.Module):
    # __init__ 메서드는 masked self-attention, cross-attention, FFNN 계층을 초기화합니다.
    def __init__(self, d: int, heads: int = 8, n_mlp: int = 4):
        # 부모 클래스의 초기화 기능을 실행합니다.
        super().__init__()

        # 지금까지 생성된 출력 토큰끼리만 참고하는 masked self-attention입니다.
        self.self_attention = MultiHeadAttention(d, heads=heads)

        # 인코더가 만든 입력 문장 표현(memory)을 참고하는 cross-attention입니다.
        self.cross_attention = MultiHeadAttention(d, heads=heads)

        # masked self-attention 결과를 안정화하는 LayerNorm입니다.
        self.norm1 = nn.LayerNorm(d)

        # cross-attention 결과를 안정화하는 LayerNorm입니다.
        self.norm2 = nn.LayerNorm(d)

        # FFNN 결과를 안정화하는 LayerNorm입니다.
        self.norm3 = nn.LayerNorm(d)

        # 각 토큰 위치마다 독립적으로 적용되는 Feed Forward Neural Network입니다.
        self.ff = nn.Sequential(
            nn.Linear(d, n_mlp * d),
            nn.ReLU(),
            nn.Linear(n_mlp * d, d),
        )

    # forward 메서드는 디코더 입력 x와 인코더 출력 memory를 받아 블록 출력을 반환합니다.
    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # 미래 토큰을 보지 못하도록 causal mask를 적용한 self-attention을 수행합니다.
        x = self.norm1(self.self_attention(x, causal_mask=True) + x)

        # 인코더 출력(memory)을 Key/Value로 사용하는 cross-attention을 수행합니다.
        x = self.norm2(self.cross_attention(x, memory) + x)

        # FFNN을 적용하고 Residual + LayerNorm으로 마무리합니다.
        x = self.norm3(self.ff(x) + x)

        # 최종 블록 출력을 반환합니다.
        return x


# Seq2SeqTranslator 클래스는 Transformer 인코더-디코더 전체 번역 모델입니다.
# 기존 GRU 모델과 동일하게 .encoder(...) / .decoder(...) 인터페이스를 제공하여
# train.py, predict.py에서 최소한의 수정으로 사용할 수 있도록 설계했습니다.
class Seq2SeqTranslator(nn.Module):
    # __init__ 메서드는 임베딩, 인코더/디코더 블록, 출력 계층을 초기화합니다.
    def __init__(self, vocab_size: int, d_model: int = 64, heads: int = 4, depth: int = 2, ff_mult: int = 4, max_len: int = 64):
        # 부모 클래스의 초기화 기능을 실행합니다.
        super().__init__()

        # 모델 차원 d를 저장합니다.
        self.d_model = d_model

        # 위치 임베딩이 감당할 수 있는 최대 문장 길이를 저장합니다.
        self.max_len = max_len

        # 입력/출력이 같은 문자 사전을 공유하므로 하나의 토큰 임베딩을 사용합니다.
        # PAD(0번)는 학습되지 않도록 padding_idx=0으로 지정합니다.
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 토큰 순서 정보를 주기 위한 위치 임베딩입니다. (Transformer는 RNN처럼 순서를 알지 못하므로 필요)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # depth 개수만큼 인코더 블록을 쌓습니다.
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, heads=heads, n_mlp=ff_mult) for _ in range(depth)])

        # depth 개수만큼 디코더 블록을 쌓습니다.
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, heads=heads, n_mlp=ff_mult) for _ in range(depth)])

        # 디코더 출력을 문자 사전 크기의 점수(logits)로 변환하는 출력 계층입니다.
        self.fc = nn.Linear(d_model, vocab_size)

    # _embed 메서드는 정수 토큰 시퀀스를 (토큰 임베딩 + 위치 임베딩) 표현으로 변환합니다.
    def _embed(self, ids: torch.LongTensor) -> torch.Tensor:
        # ids의 크기는 (배치 b, 문장 길이 l)입니다.
        b, l = ids.size()

        # 0부터 l-1까지의 위치 인덱스를 생성합니다.
        position_ids = torch.arange(l, device=ids.device)

        # 토큰 임베딩과 위치 임베딩을 더하여 의미 정보와 순서 정보를 모두 담습니다.
        return self.token_emb(ids) + self.pos_emb(position_ids).unsqueeze(0)

    # encoder 메서드는 입력 문장을 읽어 문맥이 반영된 표현(memory)을 반환합니다.
    def encoder(self, source_ids: torch.LongTensor) -> torch.Tensor:
        # 입력 토큰을 임베딩합니다.
        x = self._embed(source_ids)

        # 모든 인코더 블록을 순서대로 통과시킵니다.
        for block in self.encoder_blocks:
            x = block(x)

        # 인코더 출력(memory)을 반환합니다. 크기는 (b, 입력 길이, d)입니다.
        return x

    # decoder 메서드는 디코더 입력과 인코더 출력을 받아 문자별 점수(logits)를 반환합니다.
    def decoder(self, decoder_input_ids: torch.LongTensor, memory: torch.Tensor) -> torch.Tensor:
        # 디코더 입력 토큰을 임베딩합니다.
        x = self._embed(decoder_input_ids)

        # 모든 디코더 블록을 순서대로 통과시킵니다. 각 블록은 memory를 참고(cross-attention)합니다.
        for block in self.decoder_blocks:
            x = block(x, memory)

        # 각 위치의 표현을 문자별 점수(logits)로 변환합니다. 크기는 (b, 출력 길이, vocab)입니다.
        return self.fc(x)

    # forward 메서드는 학습에 사용하며 인코더와 디코더를 한 번에 통과시킵니다.
    def forward(self, source_ids: torch.LongTensor, decoder_input_ids: torch.LongTensor) -> torch.Tensor:
        # 인코더로 입력 문장의 표현을 만듭니다.
        memory = self.encoder(source_ids)

        # 디코더로 정답의 이전 토큰들을 입력받아 다음 문자 점수를 예측합니다.
        return self.decoder(decoder_input_ids, memory)
