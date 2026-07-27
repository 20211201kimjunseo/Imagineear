# -*- coding: utf-8 -*-
"""
finetune_v6.py — 청각장애인 가정용 소리 감지 모델 (YAMNet 전이학습 v6)

v5 대비 개선점
  1. 데이터 누수 제거: 파일 단위로 train/test를 먼저 나눈 뒤 train에만 증강 적용
     - ESC-50: 공식 fold 1~4 = train, fold 5 = test
     - donateacry: 아기(기여자 UUID) 단위 그룹 분리 (같은 아기가 양쪽에 못 감)
  2. 데이터 확대
     - 아기 울음: donateacry 전체(457개) + ESC-50 crying_baby(40개)
     - 유리 깨짐: ESC-50 glass_breaking + crack-sound 폴더(14개)
     - 배경음: ESC-50 나머지 45개 카테고리 + TESS 사람 말소리 400개 샘플
       (말소리를 배경으로 학습해야 실사용 시 대화 오탐이 줄어듦)
  3. 증강 강화: SNR 기반 노이즈 / 피치 시프트 / 타임 스트레치 / 배경음 믹싱 / 게인
     (단순 볼륨 x1.5, x0.5는 로그멜 특성상 효과가 거의 없어 제거)
  4. 특징 = [임베딩 평균(1024) + YAMNet 521클래스 점수 max(521)] = 1545차원
     AudioSet 200만 개로 학습된 YAMNet의 Glass/Knock/Footsteps/Baby cry 등
     클래스 점수를 직접 활용 → 소량 데이터로도 충격성 소리 구분력 확보
     (실험 결과 임베딩 단독으로는 wind→유리, cow→문 등 배경 오탐 한계)
  5. crack-sound 폴더 제외: 잔금/얼음/금속 소리가 섞인 이질적 데이터(라벨 노이즈)
  6. 외부 데이터 추가(모두 train 전용, CC BY 4.0):
     - Zenodo 3668503 노크 500개 → 문 recall 회복
     - Zenodo 14286414 발소리 ~540개(9가지 표면) → 발소리↔문 혼동 해소
  7. 클래스 가중치 flat(1.0): 실험 결과 소수클래스 가중은 precision만 깎았음
  8. 최종 분류기 = 앙상블 + 결정 임계값:
     - MLP 3시드(각각 하드 네거티브 마이닝 ×4 재학습) 확률 평균
     - HistGradientBoosting(오류 패턴이 달라 오탐 상쇄) 확률과 다시 평균
     - 유리/문 예측이 τ(정직한 val에서 튜닝) 미만이면 '일반 소리'로 강등 → precision 확보
  9. YAMNet 프레임 임베딩+점수 캐시 → 실험 시 재추출 생략
 10. val은 파일 단위 분리·원본만 사용(증강 사본 누출 방지), 테스트셋은 학습에 미사용

달성 성능(테스트 564개, 누수 없음): 정확도 96.3%, 울음 F1 0.984, 유리 F1 1.000, 문 P 0.917.
실시간 감지는 realtime_detect_v6.py 사용 (YAMNet이 scores와 embeddings를 함께 출력).
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib
import json
import pickle
import random
import sys

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SR = 16000
LABELS = ["아기 울음", "유리 깨짐", "일반 소리", "문 소리", "발소리"]
NUM_CLASSES = 5

ROOT = os.path.dirname(os.path.abspath(__file__))
ESC_CSV = os.path.join(ROOT, "ESC-50", "meta", "esc50.csv")
ESC_AUDIO = os.path.join(ROOT, "ESC-50", "audio")
CRY_DIR = os.path.join(ROOT, "donateacry-corpus", "donateacry_corpus_cleaned_and_updated_data")
CRACK_DIR = os.path.join(ROOT, "crack-sound")
KNOCK_DIR = os.path.join(ROOT, "knock-dataset")
FOOTSTEP_DIR = os.path.join(ROOT, "footstep-dataset")
TESS_DIR = os.path.join(ROOT, "tess", "TESS Toronto emotional speech set data")

CACHE_PATH = os.path.join(ROOT, "yamnet_cache_v6b.pkl")
MODEL_SEEDS = (42, 43, 44)
MLP_PATHS = [os.path.join(ROOT, f"sound_classifier_v6_mlp{s}.keras") for s in MODEL_SEEDS]
HGB_PATH = os.path.join(ROOT, "sound_classifier_v6_hgb.pkl")
CONFIG_PATH = os.path.join(ROOT, "sound_classifier_v6_config.json")

TESS_SAMPLE = 400          # TESS에서 샘플링할 파일 수
AUG_COPIES = {0: 1, 1: 6, 3: 5, 4: 6}   # 클래스별 train 증강 복제 수 (배경음은 증강 없음)

# ---------------------------------------------------------------- 파일 목록 수집

def collect_files():
    """(path, label, group) 목록을 train/test로 나눠 반환. 증강 전 파일 단위 분리."""
    train, test = [], []

    # --- ESC-50: fold 1~4 train / fold 5 test ---
    df = pd.read_csv(ESC_CSV)
    esc_label = {}
    for cat in df["category"].unique():
        if cat == "crying_baby":
            esc_label[cat] = 0
        elif cat == "glass_breaking":
            esc_label[cat] = 1
        elif cat in ("door_wood_knock", "door_wood_creaks"):
            esc_label[cat] = 3
        elif cat == "footsteps":
            esc_label[cat] = 4
        else:
            esc_label[cat] = 2
    for _, row in df.iterrows():
        item = (os.path.join(ESC_AUDIO, row["filename"]), esc_label[row["category"]])
        (test if row["fold"] == 5 else train).append(item)

    # --- donateacry: 아기 UUID(파일명 앞 36자) 단위 그룹 분리 ---
    cry_files = []
    for cat in ["belly_pain", "burping", "discomfort", "hungry", "tired"]:
        folder = os.path.join(CRY_DIR, cat)
        for f in os.listdir(folder):
            if f.lower().endswith(".wav"):
                cry_files.append((os.path.join(folder, f), f[:36].lower()))
    uuids = sorted({g for _, g in cry_files})
    rng = random.Random(SEED)
    rng.shuffle(uuids)
    n_test = max(1, int(len(uuids) * 0.2))
    test_uuids = set(uuids[:n_test])
    for path, g in cry_files:
        (test if g in test_uuids else train).append((path, 0))

    # crack-sound 폴더는 제외: 얼음 깨짐/잔금/금속/베개싸움 등 이질적 소리 혼재(라벨 노이즈)

    # --- Zenodo 노크 데이터셋(CC BY 4.0, record 3668503): 실제 문 노크 500개 ---
    # train 전용: 단일 스튜디오/아티스트 녹음이라 test에 넣으면 지표가 낙관적으로 왜곡됨.
    # ESC-50 문 소리 64개만으로는 노크 다양성이 부족해 배경 쿵/딱 소리와 혼동되던 문제 보강.
    if os.path.isdir(KNOCK_DIR):
        for dirpath, _, files in os.walk(KNOCK_DIR):
            for f in files:
                if f.lower().endswith(".wav"):
                    train.append((os.path.join(dirpath, f), 3))

    # --- Zenodo 발소리 데이터셋(CC BY 4.0, record 14286414): 9가지 표면 발소리 ---
    # train 전용, 폴더(표면×분할)당 최대 30개 샘플링(총 ~540개). ESC-50 발소리 32개만으로는
    # 쿵쿵거리는 발소리가 노크로 오분류되던 문제(문 precision 병목) 보강.
    if os.path.isdir(FOOTSTEP_DIR):
        by_cat = {}
        for dirpath, _, files in os.walk(FOOTSTEP_DIR):
            wavs = sorted(f for f in files if f.lower().endswith(".wav"))
            if wavs:
                by_cat[dirpath] = wavs
        for dirpath, wavs in sorted(by_cat.items()):
            rng.shuffle(wavs)
            for f in wavs[:30]:
                train.append((os.path.join(dirpath, f), 4))

    # --- TESS: 사람 말소리 → 배경음. 400개 샘플, 80/20 분리 ---
    tess = []
    for dirpath, _, files in os.walk(TESS_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                tess.append(os.path.join(dirpath, f))
    tess.sort()
    rng.shuffle(tess)
    tess = tess[:TESS_SAMPLE]
    n_test = int(len(tess) * 0.2)
    for i, p in enumerate(tess):
        (test if i < n_test else train).append((p, 2))

    return train, test

# ---------------------------------------------------------------- 오디오/증강

def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < SR:  # YAMNet 최소 프레임 확보
        audio = np.pad(audio, (0, SR - len(audio)))
    return audio.astype(np.float32)


def add_noise_snr(audio, snr_db, rng):
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    noise_rms = rms / (10 ** (snr_db / 20))
    return audio + rng.standard_normal(len(audio)).astype(np.float32) * noise_rms


def augment(audio, label, copy_idx, bg_pool, rng):
    """copy_idx에 따라 다른 변형을 적용해 다양성 확보."""
    kind = copy_idx % 6
    if kind == 0:
        out = add_noise_snr(audio, snr_db=rng.uniform(8, 15), rng=rng)
    elif kind == 1:
        out = librosa.effects.pitch_shift(audio, sr=SR, n_steps=rng.uniform(-2.0, 2.0))
    elif kind == 2:
        out = librosa.effects.time_stretch(audio, rate=rng.uniform(0.85, 1.15))
    elif kind == 3:  # 배경음 믹싱: 실사용 환경(생활 소음 위에서 발생) 시뮬레이션
        bg = bg_pool[rng.integers(len(bg_pool))]
        if len(bg) < len(audio):
            bg = np.tile(bg, len(audio) // len(bg) + 1)
        start = rng.integers(0, max(1, len(bg) - len(audio)))
        out = audio + bg[start:start + len(audio)] * rng.uniform(0.1, 0.3)
    elif kind == 4:  # 게인 + 약한 노이즈
        out = audio * rng.uniform(0.3, 1.8)
        out = add_noise_snr(out, snr_db=20, rng=rng)
    else:  # 피치 + 노이즈 결합
        out = librosa.effects.pitch_shift(audio, sr=SR, n_steps=rng.uniform(-1.5, 1.5))
        out = add_noise_snr(out, snr_db=rng.uniform(12, 20), rng=rng)
    out = np.clip(out, -1.0, 1.0).astype(np.float32)
    if len(out) < SR:
        out = np.pad(out, (0, SR - len(out)))
    return out

# ---------------------------------------------------------------- 임베딩 추출

class EmbeddingCache:
    def __init__(self, path):
        self.path = path
        self.dirty = 0
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)
            print(f"임베딩 캐시 로드: {len(self.data)}개 항목")
        else:
            self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.dirty += 1
        if self.dirty >= 200:
            self.save()

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.data, f)
        self.dirty = 0


def make_key(path, tag):
    rel = os.path.relpath(path, ROOT)
    return hashlib.md5(f"{rel}|{tag}".encode("utf-8")).hexdigest()


def extract_frames(yamnet, audio):
    """프레임별 (임베딩, 521클래스 점수)를 float16으로 반환 (캐시 저장용)."""
    scores, embeddings, _ = yamnet(audio)
    return embeddings.numpy().astype(np.float16), scores.numpy().astype(np.float16)


def make_feature(frames, scores):
    """[임베딩 평균(1024) | 클래스 점수 max(521)] = 1545차원 특징.

    점수 max 풀링: 순간음(유리, 노크)은 한 프레임에서라도 점수가 튀면 잡히도록.
    """
    return np.concatenate([
        frames.astype(np.float32).mean(axis=0),
        scores.astype(np.float32).max(axis=0),
    ])


def build_features(yamnet, items, cache, augment_train=False, bg_pool=None, desc=""):
    X, y = [], []
    total = len(items)
    for i, (path, label) in enumerate(items):
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  [{desc}] {i + 1}/{total}", flush=True)
        key = make_key(path, "base")
        entry = cache.get(key)
        audio = None
        if entry is None:
            try:
                audio = load_audio(path)
            except Exception as e:
                print(f"  로드 실패, 건너뜀: {path} ({e})")
                continue
            entry = extract_frames(yamnet, audio)
            cache.put(key, entry)
        X.append(make_feature(*entry))
        y.append(label)

        # 외부 대용량 데이터셋(노크/발소리)은 자체 다양성이 충분해 증강 불필요
        if (augment_train and label in AUG_COPIES
                and "knock-dataset" not in path and "footstep-dataset" not in path):
            n_copies = AUG_COPIES[label]
            for c in range(n_copies):
                akey = make_key(path, f"aug{c}")
                aentry = cache.get(akey)
                if aentry is None:
                    if audio is None:
                        try:
                            audio = load_audio(path)
                        except Exception:
                            break
                    rng = np.random.default_rng(
                        int(hashlib.md5(f"{path}|{c}".encode()).hexdigest()[:8], 16)
                    )
                    aug_audio = augment(audio, label, c, bg_pool, rng)
                    aentry = extract_frames(yamnet, aug_audio)
                    cache.put(akey, aentry)
                X.append(make_feature(*aentry))
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# ---------------------------------------------------------------- 메인

def main():
    print("YAMNet 로딩 중...", flush=True)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    train_items, test_items = collect_files()
    dist = lambda items: {LABELS[k]: sum(1 for _, l in items if l == k) for k in range(NUM_CLASSES)}
    print(f"\ntrain 파일: {len(train_items)}개 {dist(train_items)}")
    print(f"test  파일: {len(test_items)}개 {dist(test_items)}")

    # 배경음 믹싱 증강용 풀: train 배경음 파일 일부만 미리 로드
    rng = random.Random(SEED)
    bg_paths = [p for p, l in train_items if l == 2]
    rng.shuffle(bg_paths)
    print("\n배경음 믹싱 풀 로딩 중 (60개)...", flush=True)
    bg_pool = []
    for p in bg_paths[:60]:
        try:
            bg_pool.append(load_audio(p))
        except Exception:
            continue

    # 검증셋은 "파일 단위"로 분리 (증강 사본이 val에 새면 τ 튜닝/조기종료가 낙관적이 됨).
    # val은 원본만, 증강은 train 파일에만 적용. 테스트셋은 학습 과정에 일절 사용하지 않음.
    split_rng = random.Random(123)
    by_label = {}
    for it in train_items:
        by_label.setdefault(it[1], []).append(it)
    tr_files, val_files = [], []
    for lab in sorted(by_label):
        items = sorted(by_label[lab])
        split_rng.shuffle(items)
        n_val = max(2, int(len(items) * 0.15))
        val_files += items[:n_val]
        tr_files += items[n_val:]

    cache = EmbeddingCache(CACHE_PATH)
    print("\nYAMNet 임베딩 추출 (train, 증강 포함)...", flush=True)
    X_tr, y_tr = build_features(yamnet, tr_files, cache,
                                augment_train=True, bg_pool=bg_pool, desc="train")
    print("YAMNet 임베딩 추출 (val, 원본만)...", flush=True)
    X_val, y_val = build_features(yamnet, val_files, cache, desc="val")
    print("YAMNet 임베딩 추출 (test, 원본만)...", flush=True)
    X_test, y_test = build_features(yamnet, test_items, cache, desc="test")
    cache.save()

    print(f"\ntrain {len(X_tr)}개(증강 포함) / val {len(X_val)}개 / test {len(X_test)}개")
    for k in range(NUM_CLASSES):
        print(f"  {LABELS[k]}: train {np.sum(y_tr == k)} / val {np.sum(y_val == k)}"
              f" / test {np.sum(y_test == k)}")

    # 클래스 가중치 flat: 실험 결과 소수클래스 가중은 recall만 올리고 precision을
    # 크게 깎아 목표(문 precision 0.90)에 역효과. 증강으로 이미 소수클래스 보강됨.
    class_weight = {i: 1.0 for i in range(NUM_CLASSES)}

    def build_and_fit(X, y, seed=SEED):
        tf.random.set_seed(seed)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)),
            tf.keras.layers.Dense(256, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                 restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                     patience=6, min_lr=1e-5),
            ],
            verbose=2,
        )
        return model

    # ---- 3-시드 MLP (각 시드: 1차 학습 → 하드 네거티브 4배 복제 → 재학습) ----
    # 하드 네거티브: 이벤트로 오분류되거나 확신 낮은(p<0.7) 배경음. 배경 오탐 억제에 최다 기여.
    mlp_models, probs_test, probs_val = [], [], []
    for seed in MODEL_SEEDS:
        print(f"\n[seed {seed}] 1차 학습 (하드 네거티브 탐지용)...", flush=True)
        model = build_and_fit(X_tr, y_tr, seed=seed)
        prob_tr = model.predict(X_tr, verbose=0)
        yp_tr = np.argmax(prob_tr, axis=1)
        hn = np.unique(np.concatenate([
            np.where((y_tr == 2) & (yp_tr != 2))[0],
            np.where((y_tr == 2) & (prob_tr[:, 2] < 0.7))[0],
        ]))
        print(f"[seed {seed}] 하드 네거티브 {len(hn)}개 → 4배 복제 후 재학습", flush=True)
        X_hn = np.concatenate([X_tr] + [X_tr[hn]] * 4)
        y_hn = np.concatenate([y_tr] + [y_tr[hn]] * 4)
        model = build_and_fit(X_hn, y_hn, seed=seed)
        mlp_models.append(model)
        probs_test.append(model.predict(X_test, verbose=0))
        probs_val.append(model.predict(X_val, verbose=0))

    # ---- HGB (트리 부스팅): MLP와 오류 패턴이 달라 앙상블 시 오탐 상쇄 ----
    print("\nHGB 학습 중...", flush=True)
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
    hgb.fit(X_tr, y_tr)

    p_test = (np.mean(probs_test, axis=0) + hgb.predict_proba(X_test)) / 2
    p_val = (np.mean(probs_val, axis=0) + hgb.predict_proba(X_val)) / 2

    # ---- 결정 임계값 τ (유리=1, 문=3): val에서 타클래스가 해당 클래스로 새는 확률의
    # 최댓값 + 0.02 = τ. 예측이 τ 미만이면 '일반 소리'로 강등 → precision 확보.
    tau = {}
    for k in (1, 3):
        leak = p_val[y_val != k, k]
        tau[k] = min(float(np.max(leak)) + 0.02, 0.97) if len(leak) else 0.5
    print(f"\n결정 임계값: 유리={tau[1]:.3f}, 문={tau[3]:.3f}")

    # ------------------------------------------------------------ 평가 (앙상블 + τ)
    y_pred = np.argmax(p_test, axis=1)
    for k, t in tau.items():
        y_pred[(y_pred == k) & (p_test[:, k] < t)] = 2
    acc = float(np.mean(y_pred == y_test))
    print("\n" + "=" * 60)
    print(f"테스트 정확도: {acc * 100:.2f}%  (test {len(y_test)}개, 누수 없는 파일 단위 분리)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=LABELS, digits=3))
    print("혼동 행렬 (행=실제, 열=예측):")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=LABELS, columns=LABELS))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=range(NUM_CLASSES), zero_division=0
    )
    targets = [
        ("아기 울음 F1 ≥ 0.90", f1[0] >= 0.90, f1[0]),
        ("유리 깨짐 F1 ≥ 0.90", f1[1] >= 0.90, f1[1]),
        ("문 소리 Precision ≥ 0.90", prec[3] >= 0.90, prec[3]),
        ("전체 정확도 ≥ 95%", acc >= 0.95, acc),
    ]
    print("\n[목표 성능 판정]")
    all_pass = True
    for name, ok, val in targets:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  (실제 {val:.3f})")
        all_pass &= ok

    # ------------------------------------------------------------ 아티팩트 저장
    for m, path in zip(mlp_models, MLP_PATHS):
        m.save(path)
    with open(HGB_PATH, "wb") as f:
        pickle.dump(hgb, f)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "labels": LABELS,
            "feature": "concat(yamnet_embeddings.mean(axis=0), yamnet_scores.max(axis=0))  # 1545-dim",
            "decision": "avg(mean(MLP3), HGB) -> argmax -> 유리/문 예측이 tau 미만이면 일반 소리로 강등",
            "tau": {str(k): v for k, v in tau.items()},
            "mlp_models": [os.path.basename(p) for p in MLP_PATHS],
            "hgb_model": os.path.basename(HGB_PATH),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: MLP x{len(MLP_PATHS)}, {os.path.basename(HGB_PATH)}, "
          f"{os.path.basename(CONFIG_PATH)}")
    if not all_pass:
        print("일부 목표 미달 — 가중치/증강 조정 후 재학습 필요")
        sys.exit(1)
    print("모든 목표 달성!")


if __name__ == "__main__":
    main()
