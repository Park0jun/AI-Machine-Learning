import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset_lstm import load_dataset_sequences, augment_sequences
from src.build_model_lstm import build_bilstm, build_bilstm_with_skip

# 모델 저장 경로
MODEL_PATH = os.path.join(ROOT, "experiments", "height_lstm_model")

# 하이퍼파라미터
SEQ_LEN = 64        # 시퀀스 길이
STRIDE = 32         # 슬라이딩 윈도우 간격
BATCH_SIZE = 32     # 배치 크기
EPOCHS = 100        # 최대 에포크 (Early Stopping으로 조기 종료)
LEARNING_RATE = 1e-3
USE_SKIP_CONNECTION = True   # Skip Connection 사용 여부
USE_AUGMENTATION = True      # 데이터 증강 사용 여부


def train():
    print("="*60)
    print("Bi-LSTM Height Refinement Model Training")
    print("="*60)
    
    # 1) 데이터 로딩
    print("\n[1/4] Loading dataset...")
    X, Y = load_dataset_sequences(seq_len=SEQ_LEN, stride=STRIDE)
    
    if USE_AUGMENTATION:
        print("\n[1.5/4] Augmenting data...")
        X, Y = augment_sequences(X, Y)
    
    # 2) 모델 생성
    print("\n[2/4] Building model...")
    if USE_SKIP_CONNECTION:
        model = build_bilstm_with_skip(seq_len=SEQ_LEN, input_dim=99, output_dim=99)
        print("Using: Bi-LSTM with Skip Connection")
    else:
        model = build_bilstm(seq_len=SEQ_LEN, input_dim=99, output_dim=99)
        print("Using: Basic Bi-LSTM")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )
    
    model.summary()
    
    # 3) 콜백 설정
    callbacks = [
        # 검증 손실이 10 에포크 동안 개선되지 않으면 조기 종료
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # 검증 손실이 5 에포크 동안 개선되지 않으면 학습률 감소
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # 최고 성능 모델 저장
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 4) 학습
    print("\n[3/4] Training...")
    history = model.fit(
        X, Y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        shuffle=True,
        callbacks=callbacks
    )
    
    # 5) 결과 출력
    print("\n[4/4] Training Complete!")
    print(f"Best val_loss: {min(history.history['val_loss']):.6f}")
    print(f"Best val_mae: {min(history.history['val_mae']):.6f}")
    print(f"Model saved → {MODEL_PATH}")
    
    return history


if __name__ == "__main__":
    train()
