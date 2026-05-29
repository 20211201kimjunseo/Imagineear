import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import librosa

# 모델 로드
print("모델 로딩 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
model = tf.keras.models.load_model('baby_cry_binary.keras')

print("모델 로딩 완료!")
print("실시간 감지 시작... (Ctrl+C로 종료)")
print("-" * 40)

sample_rate = 16000
duration = 3

while True:
    print("\n녹음 중... (3초)")
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    audio = recording.flatten()

    # YAMNet embeddings 추출
    _, embeddings, _ = yamnet(audio)
    embedding = embeddings.numpy().mean(axis=0).reshape(1, -1)

    # 이진 분류
    prediction = model.predict(embedding, verbose=0)[0][0]

    if prediction > 0.5:
        print(f"아기 울음 감지! (신뢰도: {prediction*100:.1f}%)")
    else:
        print(f"일반 소리 (신뢰도: {(1-prediction)*100:.1f}%)")