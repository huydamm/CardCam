#!/usr/bin/env python3
"""
Poker Live — Camera-based poker game tracker
Shows each card to the camera one at a time as you deal.
Tracks hole cards → flop → turn → river and calculates equity live.

Auto-confirm: hold a card flat for ~1 second at >85% confidence
Manual confirm: ENTER key
Undo last card: BACKSPACE
Reset game: R
Quit: Q

Requirements:
  model/card_model.tflite  and  model/labels.json  (run train_model.py first)
  pip install opencv-python numpy tensorflow
"""

import cv2
import numpy as np
import sys
from collections import deque

from card_detector  import detect_cards, CARD_W, CARD_H
from recognize_card import CardRecognizer, _conf_color, CONF_HIGH
from poker_equity   import calc_equity, card_str, parse_card, HAND_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
N_PLAYERS      = 5
AUTO_FRAMES    = 22      # consecutive stable frames before auto-confirm
AUTO_CONF      = 0.85    # minimum confidence for auto-confirm
SIDEBAR_W      = 330
CAM_W, CAM_H   = 800, 480

# ── Game slot definitions ─────────────────────────────────────────────────────
# Each slot: (label, street_boundary_after)
# street_boundary_after=True means recalculate equity after this slot

def build_slots(n):
    """Build ordered list of (display_name, is_street_end) for n players."""
    slots = []
    for p in range(1, n + 1):
        slots.append((f'P{p} Card 1', False))
        slots.append((f'P{p} Card 2', p == n))   # boundary after last player's 2nd card
    slots.append(('Flop 1',  False))
    slots.append(('Flop 2',  False))
    slots.append(('Flop 3',  True))
    slots.append(('Turn',    True))
    slots.append(('River',   True))
    return slots

SLOTS      = build_slots(N_PLAYERS)
TOTAL_SLOTS = len(SLOTS)          # N*2 + 5
HOLE_END   = N_PLAYERS * 2 - 1   # index of last hole card slot
FLOP_END   = HOLE_END + 3
TURN_IDX   = FLOP_END + 1
RIVER_IDX  = TURN_IDX + 1


# ── Drawing helpers ───────────────────────────────────────────────────────────

def eq_bar(pct, width=18):
    filled = round(pct / 100 * width)
    return '|' + '#' * filled + '-' * (width - filled) + '|'


def draw_sidebar(dealt: list, slot_idx: int, equity: list,
                 lock_pct: float, pending: str, height: int) -> np.ndarray:
    panel = np.full((height, SIDEBAR_W, 3), 22, dtype=np.uint8)

    def txt(msg, x, y, scale=0.42, color=(200, 200, 200), thick=1):
        cv2.putText(panel, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick)

    def hline(y, col=(60, 60, 60)):
        cv2.line(panel, (0, y), (SIDEBAR_W, y), col, 1)

    # Title
    txt('POKER LIVE', 10, 22, 0.6, (220, 220, 220), 2)
    hline(30)

    # Next slot indicator
    if slot_idx < TOTAL_SLOTS:
        slot_name, _ = SLOTS[slot_idx]
        txt(f'DEAL: {slot_name}', 10, 50, 0.52, (80, 220, 255), 2)
    else:
        txt('GAME COMPLETE', 10, 50, 0.52, (80, 255, 80), 2)

    # Auto-confirm lock bar
    bar_y = 60
    cv2.rectangle(panel, (10, bar_y), (SIDEBAR_W - 10, bar_y + 14),
                  (50, 50, 50), -1)
    if pending and lock_pct > 0:
        filled = int((SIDEBAR_W - 20) * lock_pct)
        col = (0, int(100 + 155 * lock_pct), 0)
        cv2.rectangle(panel, (10, bar_y), (10 + filled, bar_y + 14), col, -1)
        txt(f'Locking: {pending}  {lock_pct*100:.0f}%',
            14, bar_y + 11, 0.36, (200, 255, 200))
    else:
        txt('Hold card flat to auto-confirm', 14, bar_y + 11,
            0.33, (100, 100, 100))
    hline(bar_y + 18)

    y = bar_y + 32

    # ── Player hands ─────────────────────────────────────────────────────────
    txt('HOLE CARDS', 10, y, 0.40, (140, 140, 140))
    y += 16
    for p in range(N_PLAYERS):
        i0, i1 = p * 2, p * 2 + 1
        c0 = card_str(dealt[i0]) if i0 < len(dealt) else '??'
        c1 = card_str(dealt[i1]) if i1 < len(dealt) else '??'

        col = (180, 180, 180)
        if i0 < len(dealt):
            col = (100, 220, 100)

        # Equity bar if available
        if equity and p < len(equity):
            pct  = equity[p] * 100
            ecol = (0, 200, 0) if pct > 30 else (0, 140, 200)
            bar  = eq_bar(pct, 12)
            txt(f'P{p+1} {c0} {c1}  {bar} {pct:4.1f}%',
                10, y, 0.38, ecol)
        else:
            txt(f'P{p+1}:  {c0}  {c1}', 10, y, 0.40, col)
        y += 16

    hline(y + 2)
    y += 12

    # ── Board ─────────────────────────────────────────────────────────────────
    txt('BOARD', 10, y, 0.40, (140, 140, 140))
    y += 16
    board_slots = dealt[N_PLAYERS * 2:]
    flop  = [card_str(c) for c in board_slots[:3]]
    turn  = card_str(board_slots[3]) if len(board_slots) > 3 else '??'
    river = card_str(board_slots[4]) if len(board_slots) > 4 else '??'

    while len(flop) < 3:
        flop.append('??')

    txt(f'Flop:  {flop[0]}  {flop[1]}  {flop[2]}', 10, y, 0.42, (180, 180, 100))
    y += 16
    txt(f'Turn:  {turn}', 10, y, 0.42, (180, 180, 100))
    y += 16
    txt(f'River: {river}', 10, y, 0.42, (180, 180, 100))
    y += 20

    hline(y)
    y += 10

    # ── Winner / best hand labels ─────────────────────────────────────────────
    if slot_idx >= TOTAL_SLOTS and equity:
        winners = [i for i, e in enumerate(equity) if e > 0.99]
        splits  = [i for i, e in enumerate(equity) if 0.005 < e < 0.995]
        if winners:
            for w in winners:
                txt(f'WINNER: Player {w+1}!', 10, y, 0.52, (0, 255, 100), 2)
                y += 20
        elif splits:
            ps = '+'.join(f'P{i+1}' for i in splits)
            txt(f'SPLIT: {ps}', 10, y, 0.50, (0, 200, 255), 2)
            y += 20

    # ── Controls ──────────────────────────────────────────────────────────────
    hline(height - 58)
    ctrl_y = height - 50
    for line in ['ENTER=confirm  BKSP=undo',
                 'R=reset  D=debug  Q=quit']:
        txt(line, 10, ctrl_y, 0.36, (90, 90, 90))
        ctrl_y += 14

    return panel


def draw_conf_bar(frame: np.ndarray, label: str, conf: float,
                  top3: list, lock_pct: float):
    """Draw bottom confidence bar on the camera frame."""
    h, w = frame.shape[:2]
    BAR_H = 52
    bar   = np.full((BAR_H, w, 3), (30, 30, 30), dtype=np.uint8)

    color = _conf_color(conf)

    # Big label
    cv2.putText(bar, label, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Filled confidence bar
    bx, by, bw, bh = 100, 12, w - 250, 28
    cv2.rectangle(bar, (bx, by), (bx + bw, by + bh), (65, 65, 65), -1)
    fw_fill = int(bw * conf)
    cv2.rectangle(bar, (bx, by), (bx + fw_fill, by + bh), color, -1)
    cv2.rectangle(bar, (bx, by), (bx + bw, by + bh), (110, 110, 110), 1)

    # Lock progress overlay on top of confidence bar
    if lock_pct > 0:
        lw = int(bw * lock_pct)
        overlay = bar.copy()
        cv2.rectangle(overlay, (bx, by), (bx + lw, by + bh),
                      (0, 255, 100), -1)
        cv2.addWeighted(overlay, 0.35, bar, 0.65, 0, bar)
        cv2.putText(bar, f'LOCKING {lock_pct*100:.0f}%',
                    (bx + 4, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)

    # Percentage
    cv2.putText(bar, f'{conf*100:.1f}%', (bx + bw + 8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Top-3 mini
    for i, (lb, pr) in enumerate(top3):
        c = _conf_color(pr)
        cv2.putText(bar, f'{lb} {pr*100:.0f}%', (w - 120, 16 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1)

    return np.vstack([frame, bar])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load recognizer
    try:
        rec = CardRecognizer()
    except (FileNotFoundError, ImportError) as e:
        print(f'Error: {e}')
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam.')
        sys.exit(1)

    print('Poker Live — deal cards one at a time to the camera')
    print('  Auto-confirm: hold steady for ~1 sec at >85% confidence')
    print('  ENTER=manual confirm  BACKSPACE=undo  R=reset  Q=quit\n')

    # Game state
    dealt       = []          # list of raw card ints (from parse_card)
    dealt_strs  = []          # list of 'Ah'-style strings
    equity      = []          # latest equity list
    slot_idx    = 0

    # Lock / auto-confirm state
    lock_history = deque(maxlen=AUTO_FRAMES)
    lock_label   = None

    show_debug = False

    def recalculate_equity():
        nonlocal equity
        if len(dealt) < N_PLAYERS * 2:
            equity = []
            return
        hands = [dealt[p*2 : p*2+2] for p in range(N_PLAYERS)]
        board = dealt[N_PLAYERS*2:]
        eq, _ = calc_equity(hands, board)
        equity = eq

    def undo():
        nonlocal slot_idx, equity
        if dealt:
            dealt.pop()
            dealt_strs.pop()
            slot_idx = max(0, slot_idx - 1)
            lock_history.clear()
            recalculate_equity()
            print(f'  Undo — back to slot {slot_idx}: {SLOTS[slot_idx][0]}')

    def reset():
        nonlocal dealt, dealt_strs, equity, slot_idx, lock_label
        dealt.clear(); dealt_strs.clear(); equity = []; slot_idx = 0
        lock_history.clear(); lock_label = None
        print('  Game reset.')

    def confirm_card(label: str):
        nonlocal slot_idx, lock_label
        if slot_idx >= TOTAL_SLOTS:
            return
        # Check not already in game
        if label in dealt_strs:
            print(f'  {label} already dealt — skipping')
            return
        card_int = parse_card(label)
        dealt.append(card_int)
        dealt_strs.append(label)
        _, is_boundary = SLOTS[slot_idx]
        slot_idx += 1
        lock_history.clear()
        lock_label = None
        print(f'  Confirmed: {label}  (slot {slot_idx}/{TOTAL_SLOTS})')
        if is_boundary:
            print('  Calculating equity...')
            recalculate_equity()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (CAM_W, CAM_H))

        # Detect + recognise
        cards = detect_cards(frame)
        best_label, best_conf, top3 = None, 0.0, []

        if cards and slot_idx < TOTAL_SLOTS:
            _, warped, corner_crop = cards[0]
            # Use corner crop — rank digit is far more legible at 96x96
            results = rec.predict(corner_crop, top_k=3)
            best_label, best_conf = results[0]
            top3 = results

            # Draw outline
            corners = cards[0][0]
            color   = _conf_color(best_conf)
            cv2.drawContours(frame, [corners], -1, color, 3)
            cx, cy = corners.reshape(4, 2).mean(axis=0).astype(int)

            # Label above card
            label_txt = f'{best_label}  {best_conf*100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label_txt,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            tx = max(0, min(cx - tw//2, CAM_W - tw))
            ty = max(th+6, cy - 16)
            cv2.rectangle(frame, (tx-4, ty-th-4), (tx+tw+4, ty+4),
                          (0,0,0), -1)
            cv2.putText(frame, label_txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Auto-confirm lock logic
            if best_conf >= AUTO_CONF and best_label not in dealt_strs:
                lock_history.append(best_label)
            else:
                lock_history.clear()

            if len(lock_history) == AUTO_FRAMES and len(set(lock_history)) == 1:
                confirm_card(lock_history[0])
            lock_label = best_label if len(lock_history) > 0 else None
        else:
            lock_history.clear()
            lock_label = None

        # Lock progress
        lock_pct = len(lock_history) / AUTO_FRAMES if lock_label else 0.0

        # Debug mask overlay
        if show_debug:
            from card_detector import _build_mask
            mask = _build_mask(frame)
            tint = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            tint[:,:,0] = 0
            frame = cv2.addWeighted(frame, 0.65, tint, 0.35, 0)

        # Compose: camera + conf bar | sidebar
        cam_with_bar = draw_conf_bar(
            frame,
            best_label or '---',
            best_conf,
            top3,
            lock_pct,
        )
        sidebar = draw_sidebar(
            dealt, slot_idx, equity,
            lock_pct, lock_label or '', cam_with_bar.shape[0],
        )
        composite = np.hstack([cam_with_bar, sidebar])
        cv2.imshow('Poker Live', composite)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset()
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == 13:      # ENTER — manual confirm
            if best_label and best_conf >= 0.50 and slot_idx < TOTAL_SLOTS:
                confirm_card(best_label)
            elif best_label:
                print(f'  Confidence too low ({best_conf*100:.0f}%) — try again')
        elif key in (8, 127):  # BACKSPACE
            undo()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
