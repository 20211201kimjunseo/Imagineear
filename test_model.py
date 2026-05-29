import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sounddevice as sd
import librosa

# 모델 로드
print("모델 로딩 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')
model = tf.keras.models.load_model('baby_cry_model.h5')

categories = ['belly_pain (배 아픔)', 'burping (트림)', 'discomfort (불편함)', 'hungry (배고픔)', 'tired (피곤함)']

print("모델 로딩 완료!")
print("실시간 감지 시작... (Ctrl+C로 종료)")
print("-" * 40)

sample_rate = 16000
duration = 3

while True:
    input("\n준비되면 Enter! (3초 녹음)")
    print("녹음 중...")
    
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    audio = recording.flatten()
    
    # YAMNet으로 embeddings 추출
    _, embeddings, _ = yamnet(audio)
    embedding = embeddings.numpy().mean(axis=0).reshape(1, -1)
    
    # 파인튜닝 모델로 분류
    predictions = model.predict(embedding, verbose=0)
    predicted = np.argmax(predictions)
    confidence = predictions[0][predicted]
    
    print(f"\n감지 결과: {categories[predicted]}")
    print(f"신뢰도: {confidence*100:.1f}%")
    print("\n전체 확률:")
    for i, (cat, prob) in enumerate(zip(categories, predictions[0])):
        bar = "█" * int(prob * 20)
        print(f"{cat}: {prob*100:.1f}% {bar}")