# -*- coding: utf-8 -*-
"""realtime_detect_v7.py — 청각장애인 가정용 위험 상황 감지 (v7: 낙상 + 위험 조합 규칙).

프로젝트 정의: 기존 청각장애인 보조기기(화재경보 스트로브, 초인종 점멸등, 아기울음 감지기)가
커버하지 못하는 위험 — 낙상, 침입(유리 깨짐), 비명 — 을 감지한다.

구성
  1. 학습 분류기 6클래스: 아기 울음/유리 깨짐/일반/문 소리/발소리/낙상
     - MLP 3시드 + HGB 앙상블, 낙상 확률 부스트(놓침 방지), 유리/문 τ(오탐 방지)
  2. YAMNet 점수 보조 감지: 비명(핵심), 화재경보·사이렌(백업), 초인종(편의)
  3. 위험 조합 규칙 엔진 (이벤트 이력 기반):
     - 침입 의심(긴급): 유리 깨짐 → 60초 내 발소리 / 야간(23~06시) 유리는 단독도 긴급
     - 낙상 사고(긴급): 낙상 ↔ 30초 내 비명·울음 / 낙상 단독은 '높음'
     - 장시간 울음(긴급 승격): 아기 울음 3분 이상 지속
  4. 슬라이딩 윈도우(2초 창/1초 간격), 등급별 쿨다운, 시각 팝업(등급별 색)

실행: 프로젝트 폴더에서  venv\\Scripts\\python.exe realtime_detect_v7.py
빠른 점검(마이크 없이): --test
"""
import argparse
import json
import pickle
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub

SAMPLE_RATE = 16000
WINDOW_SEC = 2.0
HOP_SEC = 1.0
COOLDOWN_SEC = {"긴급": 10, "높음": 15, "보통": 30, "편의": 60}
CRY_CONSECUTIVE = 2      # 울음: 연속 2회 감지 시 알림
CRY_LONG_SEC = 180       # 울음 지속 3분 → 긴급 승격
COMBO_WINDOW_SEC = 60    # 유리→발소리 침입 조합 창
FALL_COMBO_SEC = 30      # 낙상↔비명/울음 조합 창
NIGHT_HOURS = range(23, 24), range(0, 6)   # 야간: 23~06시
FOOT_MIN_CONF = 0.75
CRY_MIN_CONF = 0.60
AUX_THRESHOLD = 0.30

AUX_ALERTS = [
    ("scream",   "비명",            ["screaming", "shout", "yell", "shriek"]),
    ("firealarm","화재경보/사이렌",  ["smoke detector", "fire alarm", "siren",
                                     "civil defense siren", "buzzer"]),
    ("doorbell", "초인종",           ["doorbell", "ding-dong"]),
]

SEVERITY_COLOR = {"긴급": "#cc0000", "높음": "#e07000", "보통": "#3366cc", "편의": "#555555"}
SEVERITY_ICON = {"긴급": "🚨", "높음": "🔴", "보통": "🟡", "편의": "🔔"}


def is_night():
    h = datetime.now().hour
    return h >= 23 or h < 6


def load_models():
    print("모델 로딩 중... (YAMNet + v7 앙상블)")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    with open("sound_classifier_v7_config.json", encoding="utf-8") as f:
        config = json.load(f)
    mlps = [tf.keras.models.load_model(p) for p in config["mlp_models"]]
    with open(config["hgb_model"], "rb") as f:
        hgb = pickle.load(f)
    tau = {int(k): v for k, v in config["tau"].items()}
    class_map = yamnet.class_map_path().numpy()
    rows = tf.io.gfile.GFile(class_map).read().splitlines()[1:]
    yamnet_names = [r.split(",")[2].strip('"').lower() for r in rows]
    return yamnet, mlps, hgb, tau, config["fall_boost"], config["labels"], yamnet_names


def detect(audio, yamnet, mlps, hgb, tau, fall_boost, yamnet_names):
    """2초 오디오 → (분류 idx, 확률, 보조감지 [(kind, 이름, 점수)])."""
    scores, embeddings, _ = yamnet(audio)
    scores_np = scores.numpy()
    feature = np.concatenate([embeddings.numpy().mean(axis=0),
                              scores_np.max(axis=0)]).reshape(1, -1)
    p_mlp = np.mean([m.predict(feature, verbose=0)[0] for m in mlps], axis=0)
    probs = (p_mlp + hgb.predict_proba(feature)[0]) / 2

    boosted = probs.copy()
    boosted[5] *= fall_boost               # 낙상: 놓침 방지 우선 (오탐은 조합/쿨다운으로 완화)
    idx = int(np.argmax(boosted))
    conf = float(probs[idx])
    if idx in tau and conf < tau[idx]:     # 확신 낮은 유리/문 → 일반
        idx, conf = 2, float(probs[2])
    if idx == 4 and conf < FOOT_MIN_CONF:
        idx, conf = 2, float(probs[2])
    if idx == 0 and conf < CRY_MIN_CONF:
        idx, conf = 2, float(probs[2])

    aux = []
    mean_scores = scores_np.mean(axis=0)
    for i in np.argsort(mean_scores)[::-1][:7]:
        if mean_scores[i] < AUX_THRESHOLD:
            break
        name = yamnet_names[i]
        for kind, kname, keywords in AUX_ALERTS:
            if any(k in name for k in keywords):
                aux.append((kind, kname, float(mean_scores[i])))
    return idx, conf, aux


def show_popup(message, color):
    def _popup():
        try:
            import tkinter as tk
            root = tk.Tk()
            root.title("위험 알림")
            root.attributes("-topmost", True)
            root.geometry("620x170+380+240")
            root.configure(bg=color)
            tk.Label(root, text=message, font=("Malgun Gothic", 24, "bold"),
                     fg="white", bg=color, wraplength=580).pack(expand=True, fill="both")
            root.after(3500, root.destroy)
            root.mainloop()
        except Exception:
            pass
    threading.Thread(target=_popup, daemon=True).start()


class RuleEngine:
    """이벤트 이력을 보고 위험 조합을 판정한다."""

    def __init__(self, fire, hop_sec):
        self.fire = fire          # fire(severity, message, conf)
        self.hop = hop_sec
        self.events = deque(maxlen=300)   # (time, kind)
        self.cry_streak = 0

    def _recent(self, kind, within):
        now = time.time()
        return any(k == kind and now - t <= within for t, k in self.events)

    def _mark(self, kind):
        self.events.append((time.time(), kind))

    def on_window(self, idx, conf, aux):
        night = is_night()

        # ---- 보조 감지 (비명은 낙상과 조합 확인) ----
        for kind, kname, score in aux:
            if kind == "scream":
                self._mark("scream")
                if self._recent("fall", FALL_COMBO_SEC):
                    self.fire("긴급", "낙상 사고 의심! (충격음+비명)", score)
                else:
                    self.fire("긴급", "비명 감지!", score)
            elif kind == "firealarm":
                self.fire("긴급", "화재경보/사이렌 감지! (백업 알림)", score)
            elif kind == "doorbell":
                self.fire("편의", "초인종 감지", score)

        # ---- 학습 분류기 ----
        if idx == 0:  # 아기 울음
            self.cry_streak += 1
            self._mark("cry")
            dur = self.cry_streak * self.hop
            if dur >= CRY_LONG_SEC:
                self.fire("긴급", f"아기 울음 {int(dur // 60)}분 이상 지속! (방치 위험)", conf)
            elif self.cry_streak >= CRY_CONSECUTIVE:
                if self._recent("fall", FALL_COMBO_SEC):
                    self.fire("긴급", "낙상 사고 의심! (충격음+울음)", conf)
                else:
                    self.fire("높음", "아기 울음 감지", conf)
        else:
            self.cry_streak = 0

        if idx == 1:  # 유리 깨짐
            self._mark("glass")
            if night:
                self.fire("긴급", "야간 유리 깨짐! (침입 의심)", conf)
            else:
                self.fire("높음", "유리 깨짐 감지", conf)

        elif idx == 3:  # 문 소리(노크) — 위험 아님, 편의
            self.fire("편의", "문 소리(노크) 감지", conf)

        elif idx == 4:  # 발소리 — 단독 알림 없음, 조합 재료
            self._mark("footsteps")
            if self._recent("glass", COMBO_WINDOW_SEC):
                self.fire("긴급", "침입 의심! (유리 깨짐 후 발소리)", conf)
            elif night:
                self.fire("보통", "야간 발소리 감지", conf)

        elif idx == 5:  # 낙상
            self._mark("fall")
            if self._recent("scream", FALL_COMBO_SEC) or self._recent("cry", FALL_COMBO_SEC):
                self.fire("긴급", "낙상 사고 의심! (비명/울음 동반)", conf)
            else:
                self.fire("높음", "낙상(충격음) 감지 — 확인 필요", conf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-popup", action="store_true")
    args = parser.parse_args()

    yamnet, mlps, hgb, tau, fall_boost, labels, yamnet_names = load_models()

    last_alert = {}

    def fire(severity, message, conf):
        now = time.time()
        if now - last_alert.get(message, 0) < COOLDOWN_SEC[severity]:
            return
        last_alert[message] = now
        stamp = time.strftime("%H:%M:%S")
        icon = SEVERITY_ICON[severity]
        print(f"\n{'=' * 50}\n  [{stamp}] {icon} [{severity}] {message} ({conf*100:.0f}%)\n{'=' * 50}")
        if not args.no_popup:
            show_popup(f"{icon} {message}", SEVERITY_COLOR[severity])

    engine = RuleEngine(fire, HOP_SEC)

    if args.test:
        audio = np.random.randn(int(WINDOW_SEC * SAMPLE_RATE)).astype(np.float32) * 0.01
        idx, conf, aux = detect(audio, yamnet, mlps, hgb, tau, fall_boost, yamnet_names)
        print(f"[TEST] 분류: {labels[idx]} ({conf*100:.1f}%), 보조: {aux}")
        engine.on_window(idx, conf, aux)
        print("[TEST] 파이프라인 정상")
        return

    buf = deque(maxlen=int(WINDOW_SEC * SAMPLE_RATE))
    buf.extend(np.zeros(int(WINDOW_SEC * SAMPLE_RATE), dtype=np.float32))
    lock = threading.Lock()

    def callback(indata, frames, t, status):
        with lock:
            buf.extend(indata[:, 0])

    print("위험 감지 시작... (Ctrl+C로 종료)")
    print("주력: 낙상 / 침입(유리+발소리) / 비명 / 아기 울음 방치  |  백업: 화재경보  |  편의: 노크, 초인종")
    print("-" * 60)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=callback):
        while True:
            time.sleep(HOP_SEC)
            with lock:
                audio = np.array(buf, dtype=np.float32)
            idx, conf, aux = detect(audio, yamnet, mlps, hgb, tau, fall_boost, yamnet_names)
            engine.on_window(idx, conf, aux)
            if idx == 2 and not aux:
                print(f"  {labels[idx]} ({conf*100:.0f}%)      ", end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료합니다.")
        sys.exit(0)
