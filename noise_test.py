import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import sounddevice as sd
import noisereduce as nr
import librosa

# YAMNet 로드
print("YAMNet 모델 로딩 중...")
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy()
class_names = list(tf.io.gfile.GFile(class_map_path).readlines())
class_names = [c.strip().split(',')[2] if ',' in c else c.strip()
               for c in class_names]

cry_keywords = ['baby cry', 'crying', 'chuckle', 'sobbing']
sample_rate = 16000

def detect_cry(audio, label):
    scores, _, _ = model(audio)
    mean_scores = scores.numpy().mean(axis=0)
    top5 = np.argsort(mean_scores)[::-1][:5]
    top5_names = [class_names[i].lower() for i in top5]
    top5_scores = [mean_scores[i] for i in top5]
    
    detected = any(any(kw in name for kw in cry_keywords) for name in top5_names)
    
    print(f"\n[{label}] 감지 결과:")
    for name, score in zip(top5_names, top5_scores):
        marker = "⚠️" if any(kw in name for kw in cry_keywords) else "  "
        if score > 0.05:
            print(f"{marker} {name} ({score:.2f})")
    print(f"→ 울음 감지: {'성공' if detected else '❌ 실패'}")
    return detected

# 1단계 - 배경 소음 학습
input("\n[1단계] 키보드 치면서 평소처럼 있어줘요. 준비되면 Enter!")
print("배경 소음 녹음 중... (30초)")
bg_noise = sd.rec(int(30 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
bg_noise = bg_noise.flatten()
print("배경 소음 학습 완료!")

# 2단계 - 소음 제거 전 테스트
input("\n[2단계] 키보드 치면서 + 스마트폰으로 아기 울음소리 틀어줘요. 준비되면 Enter!")
print("녹음 중... (5초)")
noisy_audio = sd.rec(int(5 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
noisy_audio = noisy_audio.flatten()
result_before = detect_cry(noisy_audio, "소음 제거 전")

# 3단계 - 소음 제거 후 테스트
print("\n소음 제거 중...")
cleaned_audio = nr.reduce_noise(y=noisy_audio, y_noise=bg_noise, sr=sample_rate)
result_after = detect_cry(cleaned_audio, "소음 제거 후")

# 4단계 - 결과 비교
print("\n" + "="*40)
print("📊 소음 제거 효과")
print("="*40)
print(f"소음 제거 전: {'감지' if result_before else '❌ 미감지'}")
print(f"소음 제거 후: {'감지' if result_after else '❌ 미감지'}")
if not result_before and result_after:
    print("→ 소음 제거로 감지 성공! ")
elif result_before and result_after:
    print("→ 소음 제거 전후 모두 감지!")
elif result_before and not result_after:
    print("→ 소음 제거 후 오히려 악화")
else:
    print("→ 소음 제거 전후 모두 미감지")

from sklearn.model_selection import train_test_split

# 데이터 로드
print("데이터 로딩 중...")
dataset_path = 'donateacry-corpus/donateacry_corpus_cleaned_and_updated_data'
X, y = load_data(dataset_path)
print(f"데이터 로드 완료! 총 {len(X)}개")

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
print("모델 학습 중...")
model = create_model()
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 최종 정확도
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n최종 정확도: {accuracy*100:.1f}%")

# 모델 저장
model.save('baby_cry_model.h5')
print("모델 저장 완료!")