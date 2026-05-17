import tensorflow as tf
from tensorflow.keras import layers, models

def build_bilstm(seq_len=64, input_dim=99, output_dim=99):
    """
    Bi-LSTM 기반 Temporal Refinement Network
    
    MLP의 한계 해결:
    - 시간적 의존성 학습 (이전/이후 프레임 참조)
    - 양방향 context로 jitter 감소
    - 연속적 모션 패턴 학습
    
    Args:
        seq_len: 시퀀스 길이 (프레임 수)
        input_dim: 입력 차원 (33 landmarks × 3 coords = 99)
        output_dim: 출력 차원 (동일)
    
    Returns:
        Keras Model
    """
    model = models.Sequential([
        # Input: (batch, seq_len, 99)
        layers.Input(shape=(seq_len, input_dim)),
        
        # Bi-LSTM Layer 1: 양방향으로 시간적 패턴 학습
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        ),
        layers.Dropout(0.3),
        
        # Bi-LSTM Layer 2: 더 깊은 시간적 특징 추출
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True)
        ),
        layers.Dropout(0.3),
        
        # Dense Layer: 각 프레임별 출력
        layers.TimeDistributed(layers.Dense(output_dim))
    ])
    
    return model


def build_bilstm_with_skip(seq_len=64, input_dim=99, output_dim=99):
    """
    Skip Connection이 추가된 Bi-LSTM 모델
    
    입력을 출력에 직접 더해서 잔차(residual)만 학습
    → 더 안정적인 학습, 더 빠른 수렴
    """
    inputs = layers.Input(shape=(seq_len, input_dim))
    
    # Bi-LSTM 블록
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense로 차원 맞추기
    x = layers.TimeDistributed(layers.Dense(output_dim))(x)
    
    # Skip Connection: 입력 + 학습된 잔차
    outputs = layers.Add()([inputs, x])
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    # 모델 테스트
    model = build_bilstm(seq_len=64, input_dim=99, output_dim=99)
    model.summary()
    
    print("\n" + "="*50)
    print("Skip Connection 버전:")
    print("="*50 + "\n")
    
    model_skip = build_bilstm_with_skip(seq_len=64, input_dim=99, output_dim=99)
    model_skip.summary()
