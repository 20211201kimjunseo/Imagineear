import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("YAMNet 로딩 중...")
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# 아기 울음 데이터 로드 (레이블 1)
def load_cry_data():
    dataset_path = 'donateacry-corpus/donateacry_corpus_cleaned_and_updated_data'
    categories = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    X, y = [], []
    
    print("아기 울음 데이터 로딩 중...")
    for category in categories:
        folder = os.path.join(dataset_path, category)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')][:30]
        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                audio, sr = librosa.load(fpath, sr=16000, mono=True)
                _, embeddings, _ = yamnet(audio)
                X.append(embeddings.numpy().mean(axis=0))
                y.append(1)  # 울음 = 1
            except:
                continue
    print(f"아기 울음 데이터: {len(X)}개")
    return X, y

# ESC-50 데이터 로드 (레이블 0)
def load_noise_data():
    csv_path = 'ESC-50/meta/esc50.csv'
    audio_path = 'ESC-50/audio'
    df = pd.read_csv(csv_path)
    
    # 아기 울음 제외한 소리 (baby_cry 카테고리 제외)
    df = df[df['category'] != 'crying_baby']
    
    X, y = [], []
    print("생활 소음 데이터 로딩 중...")
    
    for _, row in df.iterrows():
        fpath = os.path.join(audio_path, row['filename'])
        try:
            audio, sr = librosa.load(fpath, sr=16000, mono=True)
            _, embeddings, _ = yamnet(audio)
            X.append(embeddings.numpy().mean(axis=0))
            y.append(0)  # 울음 아님 = 0
        except:
            continue
    print(f"생활 소음 데이터: {len(X)}개")
    return X, y

# 데이터 합치기
X_cry, y_cry = load_cry_data()
X_noise, y_noise = load_noise_data()

X = np.array(X_cry + X_noise)
y = np.array(y_cry + y_noise)
print(f"\n전체 데이터: {len(X)}개")

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 이진 분류 모델
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(1024,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n모델 학습 중...")
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 성능 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n최종 정확도: {accuracy*100:.1f}%")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred, target_names=['울음 아님', '울음']))

# 모델 저장
model.save('baby_cry_binary.keras')
print("모델 저장 완료!")