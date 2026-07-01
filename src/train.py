"""번역 모델을 학습하고 모델 파일을 저장하는 실행 파일입니다."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import DATA_PATH,MODEL_PATH,META_PATH,EMBED_SIZE,HIDDEN_SIZE,EPOCHS,BATCH_SIZE,LEARNING_RATE
from src.data_utils import load_translation_pairs,build_vocab,TranslationDataset,collate_batch
from src.model import Seq2SeqTranslator


def train_model(epochs=EPOCHS):
    """CSV 번역 데이터를 사용해서 Seq2SeqTranslator번역 모델을 학습합니다."""
    # CUDA GPU를 사용할 수 있으면 GPU 사용하고, 없으면 CPU를 사용함
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CSV파일에서 양방향 번역 학습 쌍을 읽어옴
    pairs = load_translation_pairs(DATA_PATH)
    # 학습 데이터에 등장하는 문자 기반 사전을 생성함
    char2idx, idx2char = build_vocab(pairs)
    # 문자 사전 크기를 계산함
    vocab_size = len(char2idx)
    # PyTorch Dataset 객체를 생성함
    dataset = TranslationDataset(pairs,char2idx)
    # DataLoader는 데이터를 배치단위로 섞어서 모델에 공급함
    loader = DataLoader(dataset,batch_size=BATCH_SIZE,collate_fn=collate_batch)
    # Seq2Seq 번역 모델을 생성하고, 연산 장치로 이동함
    model = Seq2SeqTranslator(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)
    # PAD 토큰은 실제 정답 문자가 아니므로 손실 계산에서 제외함
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])
    # Adam Optimizer는 학습률을 자동 보정하여 안정적으로 학습되는 알고리즘임
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

    # 지정한 epoch 수만큼 반복 학습 진행
    for epoch in range(1,epochs+1):
        # 모델을 학습모드로 전환함
        model.train()
        # epoch 별 손실 합계를 저장할 변수 초가화
        total_loss = 0.0

        # DataLoader에서 미니 배치를 하나씩 꺼내서 가져옴
        for source_idx, decoder_input_idx, decoder_target_idx in loader:
            # 입력 텐서를 연산장치로 이동함
            source_idx = source_idx.to(device)
            decoder_input_idx = decoder_input_idx.to(device)
            decoder_target_idx = decoder_target_idx.to(device)

            # 이전 배치에서 계산된 기울기를 초기화함
            optimizer.zero_grad()

            # 모델 실행 순전파 처리
            logits = model(source_idx,decoder_input_idx)
            # CrossEntropyLoss는 [배치 * 시간, 클래스수] 형태의 입력을 기대함으로 형태를 변경함
            loss = criterion(logits.reshape(-1,logits.size(-1)),decoder_target_idx.reshape(-1))
            # 손실값을 기준으로 역전파를 수렴해서 기울기를 계산함
            loss.backward()
            # 기울기 폭주를 방지하기 위해 기울기 크기를 제한함
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            # 옵티마이저기 모델 가중치를 업데이트함
            optimizer.step()
            # 현재 배치 손실을 누적함
            total_loss += loss.item()

        # 20 epoch마다 학습 손실을 출력해서 학습 상황을 확인하도록 함
        if epoch==1 or epoch % 20 == 0:
            print(f"Epoch {epoch:03d} / {epochs}, Loss: {total_loss / len(loader):.4f}")

    # 모델 저장 폴더가 없으면 생성하도록 함
    MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
    # 학습된 모델 가중치를 저장함
    torch.save(model.state_dict(),MODEL_PATH)
    # 추론에 필요한 문자 사전과 하이퍼파라미터를 저장함
    torch.save({
        "char2idx":char2idx,
        "idx2char":idx2char,
        "embed_size":EMBED_SIZE,
        "hidden_size":HIDDEN_SIZE,
    },META_PATH)

    print(f"모델 저장 완료 : {MODEL_PATH}")
    # 학습된 모델 객체와 메타 정보를 반환함
    return model, char2idx, idx2char

# 모델 학습 실행
if __name__ == "__main__":
    # 이 파일을 직접 실행할 때 모델 학습을 시작함
    train_model(epochs=EPOCHS)