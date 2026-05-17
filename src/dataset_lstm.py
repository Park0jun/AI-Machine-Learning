import os
import numpy as np
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")


def load_dataset_sequences(seq_len=64, stride=32):
    """
    시퀀스 단위로 데이터 로딩 (LSTM용)
    
    MLP용 load_dataset()과 달리:
    - 프레임 단위가 아닌 시퀀스 단위로 반환
    - 슬라이딩 윈도우로 데이터 증강 효과
    
    Args:
        seq_len: 시퀀스 길이 (프레임 수)
        stride: 슬라이딩 윈도우 이동 간격
    
    Returns:
        X: (N, seq_len, 99) - 입력 시퀀스들
        Y: (N, seq_len, 99) - 타겟 시퀀스들
    """
    raw_list = sorted(glob(f"{PROCESSED_DIR}/*_raw.npy"))
    X_all, Y_all = [], []

    for rp in raw_list:
        tp = rp.replace("_raw.npy", "_target.npy")
        if not os.path.exists(tp):
            print(f"Skip (no target): {rp}")
            continue
        
        raw = np.load(rp)      # (T, 33, 3)
        tgt = np.load(tp)      # (T, 33, 3)
        
        T = raw.shape[0]
        
        # (T, 33, 3) → (T, 99)
        raw = raw.reshape(T, -1)
        tgt = tgt.reshape(T, -1)
        
        # 슬라이딩 윈도우로 시퀀스 생성
        for start in range(0, T - seq_len + 1, stride):
            end = start + seq_len
            X_all.append(raw[start:end])  # (seq_len, 99)
            Y_all.append(tgt[start:end])  # (seq_len, 99)
    
    X = np.array(X_all)  # (N, seq_len, 99)
    Y = np.array(Y_all)  # (N, seq_len, 99)
    
    print(f"Loaded LSTM dataset: X={X.shape}, Y={Y.shape}")
    print(f"  - seq_len={seq_len}, stride={stride}")
    print(f"  - Total sequences: {len(X)}")
    
    return X, Y


def augment_sequences(X, Y, noise_std=0.01, scale_range=(0.95, 1.05)):
    """
    데이터 증강: 노이즈 추가 및 스케일 변환
    
    Args:
        X, Y: 원본 시퀀스 데이터
        noise_std: 가우시안 노이즈 표준편차
        scale_range: 스케일 범위 (min, max)
    
    Returns:
        증강된 X, Y
    """
    X_aug, Y_aug = [X], [Y]
    
    # 1. 노이즈 추가
    noise = np.random.normal(0, noise_std, X.shape)
    X_aug.append(X + noise)
    Y_aug.append(Y)  # 타겟은 그대로
    
    # 2. 스케일 변환
    scale = np.random.uniform(scale_range[0], scale_range[1], (X.shape[0], 1, 1))
    X_aug.append(X * scale)
    Y_aug.append(Y * scale)
    
    X_out = np.concatenate(X_aug, axis=0)
    Y_out = np.concatenate(Y_aug, axis=0)
    
    # 셔플
    idx = np.random.permutation(len(X_out))
    
    print(f"Augmented: {len(X)} → {len(X_out)} sequences")
    
    return X_out[idx], Y_out[idx]


if __name__ == "__main__":
    # 테스트
    X, Y = load_dataset_sequences(seq_len=64, stride=32)
    print(f"\nOriginal: X={X.shape}, Y={Y.shape}")
    
    X_aug, Y_aug = augment_sequences(X, Y)
    print(f"Augmented: X={X_aug.shape}, Y={Y_aug.shape}")
