import numpy as np
import tensorflow as tf
import os
import sys
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.viser_test import PoseViser

# 모델 및 데이터 경로
MODEL_PATH = os.path.join(ROOT, "experiments", "height_lstm_model")
TEST_DATA_PATH = os.path.join(ROOT, "data", "test_keypoints")
TEST_OUTPUT_PATH = os.path.join(ROOT, "data", "output")

# LSTM 시퀀스 설정 (train_lstm.py와 동일해야 함)
SEQ_LEN = 64


def load_model():
    """LSTM 모델 로드"""
    return tf.keras.models.load_model(MODEL_PATH)


def refine_sequence(raw_path, out_path=None):
    """
    전체 시퀀스를 LSTM으로 정제
    
    LSTM은 고정 길이 시퀀스를 처리하므로,
    긴 시퀀스는 청크 단위로 처리 후 결합
    """
    model = load_model()
    
    raw = np.load(raw_path)    # (T, 33, 3)
    T = raw.shape[0]
    
    # (T, 33, 3) → (T, 99)
    x = raw.reshape(T, -1)
    
    # 패딩: T가 SEQ_LEN보다 작거나 나누어 떨어지지 않을 경우
    pad_len = (SEQ_LEN - (T % SEQ_LEN)) % SEQ_LEN
    if pad_len > 0:
        x_padded = np.pad(x, ((0, pad_len), (0, 0)), mode='edge')
    else:
        x_padded = x
    
    T_padded = x_padded.shape[0]
    
    # 청크 단위로 처리
    refined_chunks = []
    for start in range(0, T_padded, SEQ_LEN):
        end = start + SEQ_LEN
        chunk = x_padded[start:end]  # (SEQ_LEN, 99)
        chunk = chunk[np.newaxis, ...]  # (1, SEQ_LEN, 99)
        
        pred = model.predict(chunk, verbose=0)  # (1, SEQ_LEN, 99)
        refined_chunks.append(pred[0])
    
    # 결합 및 패딩 제거
    refined = np.concatenate(refined_chunks, axis=0)  # (T_padded, 99)
    refined = refined[:T]  # 원래 길이로 복원
    
    # (T, 99) → (T, 33, 3)
    refined = refined.reshape(T, 33, 3)
    
    if out_path is not None:
        np.save(out_path, refined)
        print(f"Saved refined → {out_path}")
    
    return refined


def main():
    print("="*60)
    print("Bi-LSTM Height Refinement - Test")
    print("="*60)
    
    os.makedirs(TEST_OUTPUT_PATH, exist_ok=True)
    
    raw_files = glob.glob(f"{TEST_DATA_PATH}/*_raw.npy")
    
    if not raw_files:
        print(f"No test files found in {TEST_DATA_PATH}")
        return
    
    for raw_path in raw_files:
        print(f"\nProcessing: {os.path.basename(raw_path)}")
        
        # 정제
        out_path = raw_path.replace("_raw.npy", "_lstm_refined.npy")
        refine = refine_sequence(raw_path, out_path)
        
        # 원본 로드
        raw = np.load(raw_path)
        
        # CSV 저장 (Unity 호환)
        T, nBones, _ = refine.shape
        rows = []
        for t in range(T):
            for b in range(nBones):
                x, y, z = refine[t, b]
                rows.append([t, b, x, y, z, 1.0])
        
        df = pd.DataFrame(rows, columns=["frame", "landmark", "x", "y", "z", "visibility"])
        
        filename = os.path.basename(raw_path)
        name_only = os.path.splitext(filename)[0]
        csv_path = os.path.join(TEST_OUTPUT_PATH, f"lstm_refined_{name_only}.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved → {csv_path}")
    
    # 마지막 파일 시각화
    print("\n" + "="*60)
    print("Visualization (last file)")
    print("="*60)
    
    vis = PoseViser(fps=30)
    vis.play_two_sequences(raw, refine, offset=0.0)


if __name__ == "__main__":
    main()
