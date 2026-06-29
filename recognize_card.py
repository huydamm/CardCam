#!/usr/bin/env python3
"""
Card Recognizer — inference module
Loads the trained model and classifies warped card images.

Automatically picks the right backend:
  - TFLite  (.tflite)  → used on Pi 5  (fast, lightweight)
  - Keras   (.keras)   → used on PC    (convenient for testing)

Usage as a module:
  from recognize_card import CardRecognizer
  rec = CardRecognizer()
  label, conf = rec.predict_best(warped_bgr_image)

Run directly for a live webcam test:
  python recognize_card.py

Requirements:
  PC  : pip install tensorflow
  Pi 5: pip install tflite-runtime  (much lighter than full TensorFlow)
"""

import os
import sys
import json

import cv2
import numpy as np

MODEL_DIR    = 'model'
TFLITE_PATH  = os.path.join(MODEL_DIR, 'card_model.tflite')
KERAS_PATH   = os.path.join(MODEL_DIR, 'card_model.keras')
LABELS_PATH  = os.path.join(MODEL_DIR, 'labels.json')
IMG_SIZE     = (96, 96)
CONF_HIGH    = 0.85    # green label colour above this
CONF_LOW     = 0.50    # orange label colour above this (below = red)


class CardRecognizer:
    """
    Classifies a single perspective-corrected card image.
    Prefers TFLite for deployment, falls back to Keras on PC.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        tflite = os.path.join(model_dir, 'card_model.tflite')
        keras  = os.path.join(model_dir, 'card_model.keras')
        labels = os.path.join(model_dir, 'labels.json')

        if not os.path.exists(labels):
            raise FileNotFoundError(
                f'labels.json not found at {labels}\n'
                'Run train_model.py first.'
            )
        with open(labels) as f:
            self.labels = json.load(f)
        self.n_classes = len(self.labels)

        if os.path.exists(tflite):
            self._load_tflite(tflite)
        elif os.path.exists(keras):
            self._load_keras(keras)
        else:
            raise FileNotFoundError(
                f'No model found in {model_dir}/\n'
                'Run train_model.py first.'
            )

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_tflite(self, path: str):
        # Pi 5: use tflite-runtime (pip install tflite-runtime)
        # PC  : fall back to tensorflow.lite
        try:
            import tflite_runtime.interpreter as tflite
            Interpreter = tflite.Interpreter
            backend = 'tflite-runtime'
        except ImportError:
            try:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
                backend = 'tensorflow.lite'
            except ImportError:
                raise ImportError(
                    'Install tflite-runtime (Pi) or tensorflow (PC)'
                )
        self._interp = Interpreter(model_path=path)
        self._interp.allocate_tensors()
        self._inp  = self._interp.get_input_details()[0]
        self._out  = self._interp.get_output_details()[0]
        self._mode = 'tflite'
        print(f'[CardRecognizer] TFLite ({backend})  {self.n_classes} classes')

    def _load_keras(self, path: str):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError('Install tensorflow:  pip install tensorflow')
        self._model = tf.keras.models.load_model(path)
        self._mode  = 'keras'
        print(f'[CardRecognizer] Keras  {self.n_classes} classes')

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        BGR card image (any size) → float32 batch tensor (1, 96, 96, 3).
        Pixel values scaled to [0, 255] — MobileNetV2 preprocess happens
        inside the model graph.
        """
        rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE).astype('float32')
        return np.expand_dims(resized, axis=0)   # (1, H, W, 3)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run(self, tensor: np.ndarray) -> np.ndarray:
        """Run the model and return the raw probability vector."""
        if self._mode == 'tflite':
            self._interp.set_tensor(self._inp['index'], tensor)
            self._interp.invoke()
            return self._interp.get_tensor(self._out['index'])[0]
        else:
            return self._model.predict(tensor, verbose=0)[0]

    def predict(self, card_img: np.ndarray, top_k: int = 3) -> list:
        """
        Predict card label from a warped BGR card image.
        Returns list of (label_str, confidence_float) sorted best-first.

        Example:
          [('Ah', 0.97), ('Ac', 0.02), ('Ad', 0.01)]
        """
        probs   = self._run(self._preprocess(card_img))
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [(self.labels[i], float(probs[i])) for i in top_idx]

    def predict_best(self, card_img: np.ndarray) -> tuple:
        """Return (label, confidence) for the single best prediction."""
        return self.predict(card_img, top_k=1)[0]


# ── Standalone live-test mode ─────────────────────────────────────────────────

def _conf_color(conf: float) -> tuple:
    if conf >= CONF_HIGH:
        return (0, 220, 0)      # green
    if conf >= CONF_LOW:
        return (0, 165, 255)    # orange
    return (0, 0, 220)          # red — low confidence


def main():
    from card_detector import detect_cards

    try:
        rec = CardRecognizer()
    except (FileNotFoundError, ImportError) as e:
        print(f'Error: {e}')
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam.')
        sys.exit(1)

    print('Card Recognizer — live test')
    print('  Green label  = high confidence (>=85%)')
    print('  Orange label = medium confidence (>=50%)')
    print('  Red label    = low confidence (<50%)')
    print('  Press Q to quit\n')

    BAR_H     = 56          # height of the bottom confidence bar panel
    BAR_COLOR = (30, 30, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cards   = detect_cards(frame)
        fh, fw  = frame.shape[:2]

        # ── Per-card overlays ─────────────────────────────────────────────────
        best_overall = None    # (label, conf) of highest-confidence card this frame

        for corners, warped, corner_crop in cards:
            # Predict from the corner crop — rank digit is much larger
            # relative to the 96x96 model input than in the full card
            results = rec.predict(corner_crop, top_k=3)
            best_label, best_conf = results[0]
            color = _conf_color(best_conf)

            if best_overall is None or best_conf > best_overall[1]:
                best_overall = (best_label, best_conf, results)

            # Card outline
            cv2.drawContours(frame, [corners], -1, color, 3)
            cx, cy = corners.reshape(4, 2).mean(axis=0).astype(int)

            # Label above the card centre
            label_text = f'{best_label}  {best_conf * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label_text,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            tx = max(0, min(cx - tw // 2, fw - tw))
            ty = max(th + 4, cy - 14)
            # Dark background behind text for readability
            cv2.rectangle(frame, (tx - 4, ty - th - 4),
                          (tx + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Small warped card preview next to the outline
            ph, pw = 120, 80
            small  = cv2.resize(warped, (pw, ph))
            px     = min(cx + 20, fw - pw)
            py     = max(0, cy - ph // 2)
            py     = min(py, fh - ph - BAR_H)
            frame[py:py + ph, px:px + pw] = small
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 1)

        # ── Bottom confidence bar ─────────────────────────────────────────────
        bar = np.full((BAR_H, fw, 3), BAR_COLOR, dtype=np.uint8)

        if best_overall:
            label, conf, top3 = best_overall
            color  = _conf_color(conf)
            pct    = conf * 100

            # Card name on the left
            cv2.putText(bar, label, (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Confidence bar
            bar_x, bar_y, bar_w, bar_bh = 110, 14, fw - 260, 28
            cv2.rectangle(bar, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_bh), (70, 70, 70), -1)
            filled_w = int(bar_w * conf)
            cv2.rectangle(bar, (bar_x, bar_y),
                          (bar_x + filled_w, bar_y + bar_bh), color, -1)
            cv2.rectangle(bar, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_bh), (120, 120, 120), 1)

            # Percentage text on the right of the bar
            pct_text = f'{pct:.1f}%'
            cv2.putText(bar, pct_text, (bar_x + bar_w + 8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Top-3 tiny labels in the far right
            for i, (lb, pr) in enumerate(top3):
                c = _conf_color(pr)
                cv2.putText(bar, f'{lb} {pr*100:.0f}%',
                            (fw - 130, 16 + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1)
        else:
            cv2.putText(bar, 'No card detected', (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        # Stack bar below the frame
        composite = np.vstack([frame, bar])
        cv2.imshow('Card Recognizer', composite)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
