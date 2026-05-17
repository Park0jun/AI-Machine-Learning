"""
Student A vs Student B 모델 성능 비교 스크립트

Student A: 기본 Bi-LSTM
Student B: Bi-LSTM + Skip Connection
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
STUDENT_A_PATH = os.path.join(ROOT, "experiments", "height_lstm_model")         # 기본 Bi-LSTM
STUDENT_B_PATH = os.path.join(ROOT, "experiments", "height_lstm_skip_model")   # Skip Connection
TEST_DATA_PATH = os.path.join(ROOT, "data", "test_keypoints")

SEQ_LEN = 64


# ==================== 전처리 함수 ====================

def normalize_to_hip_center(seq):
    LEFT_HIP, RIGHT_HIP = 23, 24
    hip_center = (seq[:, LEFT_HIP] + seq[:, RIGHT_HIP]) / 2
    centered = seq - hip_center[:, np.newaxis, :]
    return centered


def apply_temporal_smoothing(seq, sigma=1.0):
    smoothed = seq.copy()
    for landmark in range(33):
        for axis in range(3):
            smoothed[:, landmark, axis] = gaussian_filter1d(
                seq[:, landmark, axis], sigma=sigma
            )
    return smoothed


def create_height_corrected_target(seq):
    seq = seq.copy()
    LEFT_FOOT, RIGHT_FOOT = 31, 32
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    seq[:, :, 1] -= foot_y[:, None]
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


def preprocess_raw(raw):
    centered = normalize_to_hip_center(raw)
    smoothed = apply_temporal_smoothing(centered, sigma=1.0)
    target = create_height_corrected_target(smoothed)
    return target


# ==================== 모델 함수 ====================

def predict_lstm(model, raw):
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


# ==================== 지표 계산 ====================

def calculate_mse(pred, target):
    return np.mean((pred - target) ** 2)


def calculate_mae(pred, target):
    return np.mean(np.abs(pred - target))


def calculate_jitter(seq):
    diff = seq[1:] - seq[:-1]
    return np.mean(np.abs(diff))


# ==================== 메인 ====================

def main():
    print("="*70)
    print("Student A vs Student B Model Comparison")
    print("="*70)
    print("Student A: Basic Bi-LSTM")
    print("Student B: Bi-LSTM + Skip Connection")
    print("="*70)
    
    # 모델 로드
    print("\n[1/3] Loading models...")
    try:
        student_a = tf.keras.models.load_model(STUDENT_A_PATH)
        print("  [OK] Student A (Basic Bi-LSTM) loaded")
        student_b = tf.keras.models.load_model(STUDENT_B_PATH)
        print("  [OK] Student B (Skip Connection) loaded")
    except Exception as e:
        print(f"  [ERROR] Error loading models: {e}")
        return
    
    # 테스트 데이터 로드
    print("\n[2/3] Loading test data...")
    raw_files = glob.glob(f"{TEST_DATA_PATH}/*_raw.npy")
    
    if not raw_files:
        print(f"  [ERROR] No test files found")
        return
    
    print(f"  Found {len(raw_files)} test files")
    
    # 결과 저장
    results_a = {'jitter': [], 'mse': [], 'mae': []}
    results_b = {'jitter': [], 'mse': [], 'mae': []}
    target_jitter = []
    
    # 평가
    print("\n[3/3] Evaluating models...")
    for raw_path in raw_files:
        filename = os.path.basename(raw_path)
        print(f"\n  Processing: {filename}")
        
        raw = np.load(raw_path)
        target = preprocess_raw(raw)
        
        # Student A 예측
        pred_a = predict_lstm(student_a, raw)
        results_a['jitter'].append(calculate_jitter(pred_a))
        results_a['mse'].append(calculate_mse(pred_a, target))
        results_a['mae'].append(calculate_mae(pred_a, target))
        
        # Student B 예측
        pred_b = predict_lstm(student_b, raw)
        results_b['jitter'].append(calculate_jitter(pred_b))
        results_b['mse'].append(calculate_mse(pred_b, target))
        results_b['mae'].append(calculate_mae(pred_b, target))
        
        # Target jitter
        target_jitter.append(calculate_jitter(target))
        
        print(f"    Student A - Jitter: {results_a['jitter'][-1]:.6f} | MSE: {results_a['mse'][-1]:.6f} | MAE: {results_a['mae'][-1]:.6f}")
        print(f"    Student B - Jitter: {results_b['jitter'][-1]:.6f} | MSE: {results_b['mse'][-1]:.6f} | MAE: {results_b['mae'][-1]:.6f}")
    
    # 평균 결과
    print("\n" + "="*70)
    print("[RESULT] Student A vs Student B Comparison")
    print("="*70)
    
    avg_a = {k: np.mean(v) for k, v in results_a.items()}
    avg_b = {k: np.mean(v) for k, v in results_b.items()}
    avg_target_jitter = np.mean(target_jitter)
    
    # 개선율 계산 (A 대비 B)
    jitter_imp = (avg_a['jitter'] - avg_b['jitter']) / avg_a['jitter'] * 100
    mse_imp = (avg_a['mse'] - avg_b['mse']) / avg_a['mse'] * 100
    mae_imp = (avg_a['mae'] - avg_b['mae']) / avg_a['mae'] * 100
    
    print(f"""
+------------------------------------------------------------------+
|            Student Model Comparison Table                         |
+-----------------+---------------+---------------+-----------------+
| Metric          | Student A     | Student B     | Improvement     |
|                 | (Basic)       | (Skip Conn)   | (B vs A)        |
+-----------------+---------------+---------------+-----------------+
| Jitter          | {avg_a['jitter']:.6f}      | {avg_b['jitter']:.6f}      | {jitter_imp:+.1f}%          |
| MSE             | {avg_a['mse']:.6f}      | {avg_b['mse']:.6f}      | {mse_imp:+.1f}%          |
| MAE             | {avg_a['mae']:.6f}      | {avg_b['mae']:.6f}      | {mae_imp:+.1f}%          |
+-----------------+---------------+---------------+-----------------+

* Target (Ground Truth) Jitter: {avg_target_jitter:.6f}
* Positive improvement = Student B is better
* Negative improvement = Student A is better
""")
    
    # 그래프 생성
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['Student A\n(Basic)', 'Student B\n(Skip Conn)']
    colors = ['#ffa94d', '#51cf66']
    
    # Jitter
    ax1 = axes[0]
    jitters = [avg_a['jitter'], avg_b['jitter']]
    bars1 = ax1.bar(models, jitters, color=colors, edgecolor='black')
    ax1.axhline(y=avg_target_jitter, color='blue', linestyle='--', label='Target')
    ax1.set_ylabel('Jitter (lower is better)')
    ax1.set_title('Jitter Comparison', fontweight='bold')
    ax1.legend()
    for bar, val in zip(bars1, jitters):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                f'{val:.4f}', ha='center', fontweight='bold')
    
    # MSE
    ax2 = axes[1]
    mses = [avg_a['mse'], avg_b['mse']]
    bars2 = ax2.bar(models, mses, color=colors, edgecolor='black')
    ax2.set_ylabel('MSE (lower is better)')
    ax2.set_title('MSE Comparison', fontweight='bold')
    for bar, val in zip(bars2, mses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005, 
                f'{val:.6f}', ha='center', fontweight='bold')
    
    # MAE
    ax3 = axes[2]
    maes = [avg_a['mae'], avg_b['mae']]
    bars3 = ax3.bar(models, maes, color=colors, edgecolor='black')
    ax3.set_ylabel('MAE (lower is better)')
    ax3.set_title('MAE Comparison', fontweight='bold')
    for bar, val in zip(bars3, maes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    graph_path = os.path.join(ROOT, "data", "output", "student_comparison.png")
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    print(f"\n[GRAPH] Saved: {graph_path}")
    
    plt.show()
    
    print("="*70)
    print("Comparison Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
