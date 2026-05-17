"""
MLP vs Bi-LSTM 모델 성능 비교 스크립트

테스트 데이터에서 두 모델의 성능을 정량적으로 비교합니다.
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)  
- Jitter (프레임 간 변화량)

수정: Raw 데이터에도 동일한 전처리를 적용하여 공정한 비교
"""

import numpy as np
import tensorflow as tf
import os
import sys
import glob
from scipy.ndimage import gaussian_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 모델 경로
MLP_MODEL_PATH = os.path.join(ROOT, "experiments", "height_mlp_model")
LSTM_MODEL_PATH = os.path.join(ROOT, "experiments", "height_lstm_model")
TEST_DATA_PATH = os.path.join(ROOT, "data", "test_keypoints")

# LSTM 시퀀스 설정
SEQ_LEN = 64


# ==================== 전처리 함수 (02_process_height_dataset.py와 동일) ====================

def normalize_to_hip_center(seq):
    """엉덩이 중심을 원점으로 정규화"""
    LEFT_HIP, RIGHT_HIP = 23, 24
    hip_center = (seq[:, LEFT_HIP] + seq[:, RIGHT_HIP]) / 2
    centered = seq - hip_center[:, np.newaxis, :]
    return centered


def apply_temporal_smoothing(seq, sigma=1.0):
    """시간 축으로 가우시안 스무딩 적용"""
    smoothed = seq.copy()
    for landmark in range(33):
        for axis in range(3):
            smoothed[:, landmark, axis] = gaussian_filter1d(
                seq[:, landmark, axis], sigma=sigma
            )
    return smoothed


def create_height_corrected_target(seq):
    """높이 보정 (발 = 0)"""
    seq = seq.copy()
    LEFT_FOOT, RIGHT_FOOT = 31, 32
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    seq[:, :, 1] -= foot_y[:, None]
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


def preprocess_raw(raw):
    """Raw 데이터에 전처리 파이프라인 적용 (Ground Truth 생성)"""
    centered = normalize_to_hip_center(raw)
    smoothed = apply_temporal_smoothing(centered, sigma=1.0)
    target = create_height_corrected_target(smoothed)
    return target


# ==================== 모델 관련 함수 ====================

def load_models():
    """두 모델 로드"""
    mlp_model = tf.keras.models.load_model(MLP_MODEL_PATH)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    return mlp_model, lstm_model


def predict_mlp(model, raw):
    """MLP 모델로 예측 (프레임 단위)"""
    T = raw.shape[0]
    x = raw.reshape(T, -1)
    y = model.predict(x, verbose=0)
    return y.reshape(T, 33, 3)


def predict_lstm(model, raw):
    """LSTM 모델로 예측 (시퀀스 단위)"""
    T = raw.shape[0]
    x = raw.reshape(T, -1)
    
    pad_len = (SEQ_LEN - (T % SEQ_LEN)) % SEQ_LEN
    if pad_len > 0:
        x_padded = np.pad(x, ((0, pad_len), (0, 0)), mode='edge')
    else:
        x_padded = x
    
    T_padded = x_padded.shape[0]
    
    refined_chunks = []
    for start in range(0, T_padded, SEQ_LEN):
        end = start + SEQ_LEN
        chunk = x_padded[start:end][np.newaxis, ...]
        pred = model.predict(chunk, verbose=0)
        refined_chunks.append(pred[0])
    
    refined = np.concatenate(refined_chunks, axis=0)[:T]
    return refined.reshape(T, 33, 3)


# ==================== 지표 계산 함수 ====================

def calculate_mse(pred, target):
    """MSE 계산"""
    return np.mean((pred - target) ** 2)


def calculate_mae(pred, target):
    """MAE 계산"""
    return np.mean(np.abs(pred - target))


def calculate_jitter(seq):
    """Jitter 계산 (프레임 간 변화량의 평균)"""
    diff = seq[1:] - seq[:-1]
    jitter = np.mean(np.abs(diff))
    return jitter


# ==================== 메인 함수 ====================

def main():
    print("="*70)
    print("MLP vs Bi-LSTM Model Performance Comparison")
    print("="*70)
    
    # 1) 모델 로드
    print("\n[1/3] Loading models...")
    try:
        mlp_model, lstm_model = load_models()
        print("  [OK] MLP model loaded")
        print("  [OK] LSTM model loaded")
    except Exception as e:
        print(f"  [ERROR] Error loading models: {e}")
        return
    
    # 2) 테스트 데이터 로드
    print("\n[2/3] Loading test data...")
    raw_files = glob.glob(f"{TEST_DATA_PATH}/*_raw.npy")
    
    if not raw_files:
        print(f"  [ERROR] No test files found in {TEST_DATA_PATH}")
        return
    
    print(f"  Found {len(raw_files)} test files")
    
    # 결과 저장
    results = {
        'target_jitter': [],  # 전처리된 Ground Truth
        'mlp_jitter': [], 'mlp_mse': [], 'mlp_mae': [],
        'lstm_jitter': [], 'lstm_mse': [], 'lstm_mae': [],
    }
    
    # 3) 각 테스트 파일에 대해 비교
    print("\n[3/3] Evaluating models...")
    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        print(f"\n  Processing: {filename}")
        
        raw = np.load(raw_path)  # (T, 33, 3)
        
        # Ground Truth 생성 (Raw에 전처리 적용)
        target = preprocess_raw(raw)
        
        # 모델 예측
        mlp_pred = predict_mlp(mlp_model, raw)
        lstm_pred = predict_lstm(lstm_model, raw)
        
        # Ground Truth Jitter (전처리된 데이터)
        target_jitter = calculate_jitter(target)
        results['target_jitter'].append(target_jitter)
        
        # MLP 지표
        mlp_jitter = calculate_jitter(mlp_pred)
        mlp_mse = calculate_mse(mlp_pred, target)
        mlp_mae = calculate_mae(mlp_pred, target)
        results['mlp_jitter'].append(mlp_jitter)
        results['mlp_mse'].append(mlp_mse)
        results['mlp_mae'].append(mlp_mae)
        
        # LSTM 지표
        lstm_jitter = calculate_jitter(lstm_pred)
        lstm_mse = calculate_mse(lstm_pred, target)
        lstm_mae = calculate_mae(lstm_pred, target)
        results['lstm_jitter'].append(lstm_jitter)
        results['lstm_mse'].append(lstm_mse)
        results['lstm_mae'].append(lstm_mae)
        
        print(f"    Target Jitter: {target_jitter:.6f}")
        print(f"    MLP    Jitter: {mlp_jitter:.6f} | MSE: {mlp_mse:.6f} | MAE: {mlp_mae:.6f}")
        print(f"    LSTM   Jitter: {lstm_jitter:.6f} | MSE: {lstm_mse:.6f} | MAE: {lstm_mae:.6f}")
    
    # 평균 결과
    print("\n" + "="*70)
    print("[RESULT] Average Performance")
    print("="*70)
    
    avg_target_jitter = np.mean(results['target_jitter'])
    avg_mlp_jitter = np.mean(results['mlp_jitter'])
    avg_lstm_jitter = np.mean(results['lstm_jitter'])
    avg_mlp_mse = np.mean(results['mlp_mse'])
    avg_lstm_mse = np.mean(results['lstm_mse'])
    avg_mlp_mae = np.mean(results['mlp_mae'])
    avg_lstm_mae = np.mean(results['lstm_mae'])
    
    # 개선율 계산
    jitter_improvement = (avg_mlp_jitter - avg_lstm_jitter) / avg_mlp_jitter * 100
    mse_improvement = (avg_mlp_mse - avg_lstm_mse) / avg_mlp_mse * 100
    mae_improvement = (avg_mlp_mae - avg_lstm_mae) / avg_mlp_mae * 100
    
    print(f"""
+------------------------------------------------------------------+
|                Model Performance Comparison Table                 |
+-----------------+---------------+---------------+-----------------+
| Metric          | MLP           | Bi-LSTM       | Improvement     |
+-----------------+---------------+---------------+-----------------+
| Jitter          | {avg_mlp_jitter:.6f}      | {avg_lstm_jitter:.6f}      | {jitter_improvement:+.1f}%          |
| MSE             | {avg_mlp_mse:.6f}      | {avg_lstm_mse:.6f}      | {mse_improvement:+.1f}%          |
| MAE             | {avg_mlp_mae:.6f}      | {avg_lstm_mae:.6f}      | {mae_improvement:+.1f}%          |
+-----------------+---------------+---------------+-----------------+

* Target (Ground Truth) Jitter: {avg_target_jitter:.6f}
* Positive improvement = Bi-LSTM is better
* Negative improvement = MLP is better
""")
    
    # ==================== 그래프 생성 ====================
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Graph 1: Jitter Comparison
    ax1 = axes[0]
    models = ['Target\n(Ground Truth)', 'MLP\n(Baseline)', 'Bi-LSTM\n(Student)']
    jitters = [avg_target_jitter, avg_mlp_jitter, avg_lstm_jitter]
    colors = ['#74c0fc', '#ffa94d', '#51cf66']
    
    bars = ax1.bar(models, jitters, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Jitter (lower is better)', fontsize=12)
    ax1.set_title('Jitter Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(jitters) * 1.2)
    
    for bar, jitter in zip(bars, jitters):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                f'{jitter:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Graph 2: MSE Comparison
    ax2 = axes[1]
    models2 = ['MLP\n(Baseline)', 'Bi-LSTM\n(Student)']
    mses = [avg_mlp_mse, avg_lstm_mse]
    colors2 = ['#ffa94d', '#51cf66']
    
    bars2 = ax2.bar(models2, mses, color=colors2, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('MSE (lower is better)', fontsize=12)
    ax2.set_title('MSE Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(mses) * 1.2)
    
    for bar, mse in zip(bars2, mses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                f'{mse:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Graph 3: MAE Comparison
    ax3 = axes[2]
    maes = [avg_mlp_mae, avg_lstm_mae]
    
    bars3 = ax3.bar(models2, maes, color=colors2, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('MAE (lower is better)', fontsize=12)
    ax3.set_title('MAE Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, max(maes) * 1.2)
    
    for bar, mae in zip(bars3, maes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{mae:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 그래프 저장
    graph_path = os.path.join(ROOT, "data", "output", "model_comparison.png")
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    print(f"\n[GRAPH] Saved: {graph_path}")
    
    plt.show()
    
    print("="*70)
    print("Comparison Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
