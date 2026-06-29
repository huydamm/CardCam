#!/usr/bin/env python3
"""
Card Sample Collector — automatic dataset builder

Workflow:
  1. Type the card label  (rank then suit, e.g.  A  then  h  →  "Ah")
  2. Press ENTER to start auto-collection
  3. Hold the card in front of the camera — samples save themselves
  4. Stops automatically at TARGET_SAMPLES (304) per label
  5. ESC stops early and returns to label entry
  6. Label persists after collection — press ENTER again to top up

Two modes (TAB to toggle):
  FULL  — card lying flat on glass table (saves full card + corner crop, 2 imgs per capture)
  PEEK  — card corner only / held up     (saves corner pip, 1 img per capture)

Samples are saved to:
  samples/<label>/full_<timestamp>.jpg
  samples/<label>/crop_<timestamp>.jpg
  samples/<label>/peek_<timestamp>.jpg

Requirements: pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import sys
import time

from card_detector import (
    detect_cards, detect_corners,
    CARD_W, CARD_H, CORNER_W, CORNER_H,
)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLES_DIR      = 'samples'
RANKS            = '23456789TJQKA'
SUITS            = 'cdhs'
ALL_LABELS       = [r + s for r in RANKS for s in SUITS]
TARGET_SAMPLES   = 304    # stop collecting once this many images exist for a label
CAPTURE_INTERVAL = 6      # frames between auto-captures (~0.2 s at 30 fps)

# States
IDLE       = 'idle'        # typing label
COLLECTING = 'collecting'  # auto-capture running


# ── File I/O ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    for label in ALL_LABELS:
        os.makedirs(os.path.join(SAMPLES_DIR, label), exist_ok=True)


def count_samples() -> dict:
    counts = {}
    for label in ALL_LABELS:
        folder = os.path.join(SAMPLES_DIR, label)
        if os.path.isdir(folder):
            counts[label] = len(
                [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
            )
        else:
            counts[label] = 0
    return counts


def save_image(img: np.ndarray, label: str, prefix: str) -> str:
    folder = os.path.join(SAMPLES_DIR, label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'{prefix}_{int(time.time() * 1000)}.jpg')
    cv2.imwrite(path, img)
    return path


def normalize_label(raw: str):
    if len(raw) != 2:
        return None
    r, s = raw[0].upper(), raw[1].lower()
    return (r + s) if (r in RANKS and s in SUITS) else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

SIDEBAR_W = 260

def draw_sidebar(counts: dict, typed: str, last_msg: str,
                 height: int, mode: str, state: str,
                 saved_count: int) -> np.ndarray:
    panel = np.full((height, SIDEBAR_W, 3), 28, dtype=np.uint8)

    def text(msg, x, y, scale=0.45, color=(200, 200, 200), thick=1):
        cv2.putText(panel, msg, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    def hline(y):
        cv2.line(panel, (0, y), (SIDEBAR_W, y), (70, 70, 70), 1)

    # Mode indicator
    mode_col = (0, 200, 255) if mode == 'PEEK' else (0, 255, 100)
    text(f'MODE: {mode}', 8, 22, 0.55, mode_col, 2)
    text('TAB to switch', 130, 22, 0.35, (100, 100, 100))
    hline(30)

    # Label display
    text('Type label  (Rank + Suit)', 8, 50, 0.40, (140, 140, 140))
    display = typed if typed else '_'
    color   = (80, 255, 80) if len(typed) == 2 else (200, 200, 100)
    text(display, 8, 78, 1.0, color, 2)

    label = normalize_label(typed)
    if len(typed) == 1:
        text('now type suit: c d h s', 8, 95, 0.38, (120, 180, 120))
    elif label:
        n = counts.get(label, 0)
        remaining = max(0, TARGET_SAMPLES - n)
        if state == COLLECTING:
            text(f'collecting...  {n}/{TARGET_SAMPLES}', 8, 95, 0.38, (80, 220, 80))
        elif n >= TARGET_SAMPLES:
            text(f'{n} imgs — DONE!', 8, 95, 0.38, (0, 255, 0))
        else:
            text(f'{n} imgs — need {remaining} more', 8, 95, 0.38, (80, 200, 80))

    hline(103)

    # Progress bar for current label (when collecting)
    if state == COLLECTING and label:
        n      = counts.get(label, 0)
        pct    = min(n / TARGET_SAMPLES, 1.0)
        bar_x  = 8
        bar_w  = SIDEBAR_W - 16
        bar_h  = 16
        bar_y  = 107
        cv2.rectangle(panel, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        filled = int(bar_w * pct)
        col    = (0, 200, 0) if pct < 1.0 else (0, 255, 80)
        cv2.rectangle(panel, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), col, -1)
        text(f'{n} / {TARGET_SAMPLES}', bar_x + 4, bar_y + 12, 0.35, (220, 220, 220))
        grid_y0 = 132
    else:
        grid_y0 = 118

    # Progress grid (13 ranks × 4 suits)
    text('Progress  (green = goal reached)', 8, grid_y0 - 2, 0.36, (140, 140, 140))
    gx0, gy0 = 8, grid_y0 + 8
    cell = 14
    for ri, rank in enumerate(RANKS):
        for si, suit in enumerate(SUITS):
            lbl = rank + suit
            n   = counts.get(lbl, 0)
            x   = gx0 + si * cell
            y   = gy0 + ri * cell
            if n >= TARGET_SAMPLES:
                fill = (0, 180, 0)
            elif n > 0:
                fill = (0, int(60 + 140 * n / TARGET_SAMPLES), 160)
            else:
                fill = (55, 55, 55)
            cv2.rectangle(panel, (x, y),
                          (x + cell - 2, y + cell - 2), fill, -1)
            if normalize_label(typed) == lbl:
                cv2.rectangle(panel, (x, y),
                              (x + cell - 2, y + cell - 2), (255, 255, 255), 1)

    for si, s in enumerate(SUITS.upper()):
        text(s, gx0 + si * cell + 3, gy0 - 2, 0.30, (150, 150, 150))

    hline(gy0 + 13 * cell + 2)
    y_stat = gy0 + 13 * cell + 14

    total      = sum(counts.values())
    done_count = sum(1 for v in counts.values() if v >= TARGET_SAMPLES)
    text(f'Total imgs: {total}', 8, y_stat, 0.40, (180, 180, 180))
    text(f'Classes done: {done_count} / 52', 8, y_stat + 16, 0.42, (180, 180, 180))

    hline(y_stat + 26)
    y_keys = y_stat + 40
    if state == IDLE:
        for line in [
            'ENTER  start collecting',
            'TAB    FULL / PEEK mode',
            'SPACE  freeze frame',
            'ESC    clear  (or quit)',
        ]:
            text(line, 8, y_keys, 0.38, (110, 110, 110))
            y_keys += 15
    else:
        text('ESC    stop collecting', 8, y_keys, 0.38, (110, 110, 110))

    if last_msg:
        text(last_msg[:34], 8, height - 10, 0.38, (80, 220, 80))

    return panel


# ── Overlay helpers ───────────────────────────────────────────────────────────

def draw_collecting_overlay(display: np.ndarray, label: str,
                             count: int, detected: bool,
                             flash: bool) -> np.ndarray:
    """Big status bar at the top of the camera feed during auto-collect."""
    h, w = display.shape[:2]
    bar  = np.full((44, w, 3), (20, 20, 20), dtype=np.uint8)

    if flash:
        bar[:] = (0, 100, 0)  # green flash on capture

    status = f'AUTO-COLLECTING  {label}  —  {count} / {TARGET_SAMPLES}'
    col    = (0, 255, 100) if detected else (60, 60, 60)
    if not detected:
        status += '  (no card detected)'

    cv2.putText(bar, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2)

    pct    = min(count / TARGET_SAMPLES, 1.0)
    bx, by, bw, bh = 8, 36, w - 16, 6
    cv2.rectangle(bar, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
    cv2.rectangle(bar, (bx, by),
                  (bx + int(bw * pct), by + bh), (0, 200, 0), -1)

    return np.vstack([bar, display])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: cannot open webcam.')
        sys.exit(1)

    print('Card Sample Collector  (automatic)')
    print('  Type label → ENTER to start → samples auto-save until 304')
    print('  ESC stops collecting early.  ESC on empty label = quit.\n')

    mode          = 'FULL'
    state         = IDLE
    frozen        = False
    frozen_frame  = None
    typed         = ''
    last_msg      = ''
    counts        = count_samples()

    # Collecting state vars
    active_label  = None
    saved_count   = 0      # images on disk for active_label at collection start
    frame_counter = 0      # counts frames since collection started
    flash_frames  = 0      # remaining frames to show green flash

    while True:
        if state == IDLE and not frozen:
            ret, frame = cap.read()
            if not ret:
                break
        elif state == COLLECTING:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = frozen_frame.copy()

        display = frame.copy()

        # ── Detection ────────────────────────────────────────────────────────
        if mode == 'FULL':
            cards    = detect_cards(frame)
            detected = len(cards) > 0
            for corners, _, _ in cards:
                cv2.drawContours(display, [corners], -1, (0, 255, 0), 2)
            status_text = f'FULL: {len(cards)} card(s)'
            status_col  = (0, 255, 0)
        else:
            pips     = detect_corners(frame)
            detected = len(pips) > 0
            for corners, _ in pips:
                cv2.drawContours(display, [corners], -1, (0, 220, 255), 2)
            status_text = f'PEEK: {len(pips)} pip(s)'
            status_col  = (0, 220, 255)

        # ── Preview panel ────────────────────────────────────────────────────
        cam_h = display.shape[0]
        if mode == 'FULL' and cards:
            full_warped  = cards[0][1]
            corner_crop  = cards[0][2]
            prev_h       = cam_h
            prev_w       = int(prev_h * CARD_W / CARD_H)
            preview_full = cv2.resize(full_warped, (prev_w, prev_h))
            crop_h       = cam_h // 2
            crop_w       = int(crop_h * CORNER_W / CORNER_H)
            preview_crop = cv2.resize(corner_crop, (crop_w, crop_h))
            blank        = np.zeros((cam_h - crop_h, crop_w, 3), dtype=np.uint8)
            cv2.putText(blank, 'corner', (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            preview = np.hstack([preview_full,
                                  np.vstack([preview_crop, blank])])
        elif mode == 'PEEK' and detected:
            pip_warped = pips[0][1]
            prev_h     = cam_h
            prev_w     = int(prev_h * CORNER_W / CORNER_H)
            preview    = cv2.resize(pip_warped, (prev_w, prev_h))
        else:
            if mode == 'FULL':
                prev_w = int(cam_h * CARD_W / CARD_H) + int(cam_h // 2 * CORNER_W / CORNER_H)
            else:
                prev_w = int(cam_h * CORNER_W / CORNER_H)
            preview = np.zeros((cam_h, prev_w, 3), dtype=np.uint8)
            cv2.putText(preview, 'No card', (4, cam_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

        # ── Auto-collect logic ────────────────────────────────────────────────
        if state == COLLECTING:
            frame_counter += 1
            current_on_disk = counts.get(active_label, 0)

            # Stop if target reached
            if current_on_disk >= TARGET_SAMPLES:
                state    = IDLE
                last_msg = f'{active_label} done!  {current_on_disk} imgs'
                print(f'  Done: {active_label} has {current_on_disk} images.')
                active_label = None
            elif frame_counter % CAPTURE_INTERVAL == 0 and detected:
                # Save
                if mode == 'FULL':
                    save_image(cards[0][1], active_label, 'full')
                    save_image(cards[0][2], active_label, 'crop')
                    counts[active_label] = counts.get(active_label, 0) + 2
                else:
                    save_image(pips[0][1], active_label, 'peek')
                    counts[active_label] = counts.get(active_label, 0) + 1
                flash_frames = 4

            if flash_frames > 0:
                flash_frames -= 1

            display = draw_collecting_overlay(
                display, active_label,
                counts.get(active_label, 0),
                detected,
                flash=(flash_frames > 0),
            )
        else:
            # Normal idle status bar
            if frozen:
                status_text = 'FROZEN  ' + status_text
                status_col  = (0, 120, 255)
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

        # Pad preview to match display height (collecting overlay adds rows)
        if preview.shape[0] != display.shape[0]:
            diff    = display.shape[0] - preview.shape[0]
            pad     = np.zeros((diff, preview.shape[1], 3), dtype=np.uint8)
            preview = np.vstack([pad, preview])

        sidebar   = draw_sidebar(counts, typed, last_msg,
                                  display.shape[0], mode, state,
                                  counts.get(normalize_label(typed) or '', 0))
        composite = np.hstack([display, preview, sidebar])
        cv2.imshow('Card Sample Collector', composite)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if state == COLLECTING:
            if key == 27:   # ESC — stop early
                state    = IDLE
                n        = counts.get(active_label, 0)
                last_msg = f'Stopped {active_label} at {n} imgs'
                print(f'  Stopped early: {active_label} = {n} images')
                active_label = None
            # all other keys ignored while collecting

        else:  # IDLE
            if key == 27:                   # ESC — clear or quit
                if typed:
                    typed    = ''
                    last_msg = ''
                else:
                    break

            elif key == 9:                  # TAB — switch mode
                mode     = 'PEEK' if mode == 'FULL' else 'FULL'
                last_msg = f'Switched to {mode} mode'
                print(f'  Mode: {mode}')

            elif key == 32:                 # SPACE — freeze
                frozen = not frozen
                if frozen:
                    frozen_frame = frame.copy()
                last_msg = 'Frozen' if frozen else 'Live'

            elif key in (8, 127):           # Backspace
                typed = typed[:-1]

            elif key == 13:                 # ENTER — start collecting
                label = normalize_label(typed)
                if label is None:
                    last_msg = 'Bad label — type e.g. Ah or Kd'
                elif counts.get(label, 0) >= TARGET_SAMPLES:
                    last_msg = f'{label} already at {counts[label]} — done!'
                else:
                    state        = COLLECTING
                    active_label = label
                    frame_counter = 0
                    flash_frames  = 0
                    n = counts.get(label, 0)
                    print(f'  Starting auto-collect: {label}  '
                          f'(currently {n}, target {TARGET_SAMPLES})')

            elif key != 255:                # Printable — build label
                ch = chr(key)
                if len(typed) == 0 and ch.upper() in RANKS:
                    typed = ch.upper()
                elif len(typed) == 1 and ch.lower() in SUITS:
                    typed += ch.lower()

    cap.release()
    cv2.destroyAllWindows()

    counts     = count_samples()
    total      = sum(counts.values())
    done_count = sum(1 for v in counts.values() if v >= TARGET_SAMPLES)
    print(f'\nSession ended.')
    print(f'  Total samples : {total}')
    print(f'  Classes done  : {done_count}/52')
    missing = [lb for lb in ALL_LABELS if counts[lb] == 0]
    if missing:
        print(f'  Still empty   : {" ".join(missing)}')


if __name__ == '__main__':
    main()
