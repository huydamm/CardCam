#!/usr/bin/env python3
"""
Board Cam — player hole cards + community card tracker

Detection is copied verbatim from recognize_card.py.
Auto-confirm: same label for AUTO_FRAMES consecutive frames.
Manual confirm: ENTER.

Workflow:
  1. Player cards   — show BOTH hole cards at once
  2. Community cards — one card at a time (Flop 1/2/3, Turn, River)

Controls:
  ENTER      confirm current prediction(s) immediately
  BACKSPACE  undo last confirmed card
  R          reset everything
  D          toggle debug mask
  1-5        fold / unfold player
  Q          quit
"""

import cv2
import numpy as np
import sys

from card_detector  import detect_cards
from recognize_card import CardRecognizer, _conf_color

try:
    from poker_equity import calc_equity, parse_card
    _EQUITY = True
except ImportError:
    _EQUITY = False

# ── Config ────────────────────────────────────────────────────────────────────
N_PLAYERS    = 5
AUTO_FRAMES        = 20    # consecutive matching frames to auto-confirm
CONF_INSTANT       = 0.99  # high-confidence threshold
CONF_INSTANT_FRAMES = 3    # need this many consecutive high-conf frames to confirm
CAM_W, CAM_H  = 800, 480
SIDEBAR_W    = 360

BOARD_SLOTS  = ['Flop 1', 'Flop 2', 'Flop 3', 'Turn', 'River']
TOTAL_BOARD  = len(BOARD_SLOTS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def next_active_player(start, players):
    for i in range(start, N_PLAYERS):
        if not players[i]['folded']:
            return i
    return N_PLAYERS


def recalc_equity(board, players):
    if not _EQUITY:
        return None
    active = [(i, p) for i, p in enumerate(players)
              if not p['folded'] and p['cards'][0] and p['cards'][1]]
    if len(active) < 2:
        return None
    try:
        hands     = [[parse_card(p['cards'][0]), parse_card(p['cards'][1])]
                     for _, p in active]
        board_int = [parse_card(c) for c in board]
        eq, _     = calc_equity(hands, board_int)
        return {idx: e for (idx, _), e in zip(active, eq)}
    except Exception:
        return None


def _eq_bar(pct, width=13):
    filled = round(pct / 100 * width)
    return '|' + '#' * filled + '-' * (width - filled) + '|'


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_sidebar(phase, current_player, community_slot,
                 players, board, equity_map, height):
    panel = np.full((height, SIDEBAR_W, 3), 22, dtype=np.uint8)

    def txt(msg, x, y, scale=0.40, color=(200, 200, 200), thick=1):
        cv2.putText(panel, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick)

    def hline(y, col=(60, 60, 60)):
        cv2.line(panel, (0, y), (SIDEBAR_W, y), col, 1)

    txt('BOARD CAM', 10, 24, 0.60, (220, 220, 220), 2)
    hline(32)
    y = 48

    if phase == 'players' and current_player < N_PLAYERS:
        p_cur = players[current_player]
        if p_cur['cards'][0] and p_cur['cards'][1]:
            txt(f'P{current_player+1} ready — ENTER to continue',
                10, y, 0.44, (0, 255, 140), 2)
        else:
            txt(f'READ P{current_player + 1}  (show both cards)',
                10, y, 0.48, (80, 220, 255), 2)
    elif phase == 'community' and community_slot < TOTAL_BOARD:
        txt(f'DEAL  ->  {BOARD_SLOTS[community_slot]}',
            10, y, 0.50, (80, 220, 255), 2)
    else:
        txt('COMPLETE', 10, y, 0.50, (80, 255, 80), 2)

    y += 22
    hline(y)
    y += 12

    txt('PLAYERS  [1-5 fold/unfold]', 10, y, 0.34, (120, 120, 120))
    y += 16

    for i, p in enumerate(players):
        is_current = (phase == 'players' and i == current_player)
        if p['folded']:
            txt(f'P{i+1}  FOLDED', 10, y, 0.42, (55, 55, 55))
        else:
            c0 = p['cards'][0] or '--'
            c1 = p['cards'][1] or '--'
            eq = equity_map.get(i) if equity_map else None
            if is_current:
                txt(f'P{i+1}: {c0}  {c1}  <', 10, y, 0.42, (80, 220, 255), 2)
            elif eq is not None:
                pct  = eq * 100
                ecol = (0, 200, 0) if pct > 30 else (0, 140, 200)
                txt(f'P{i+1}: {c0} {c1} {_eq_bar(pct)}{pct:4.0f}%', 10, y, 0.38, ecol)
            else:
                txt(f'P{i+1}: {c0}  {c1}', 10, y, 0.42, (180, 180, 180))
        y += 18

    hline(y + 2)
    y += 14

    txt('COMMUNITY CARDS', 10, y, 0.34, (120, 120, 120))
    y += 16
    flop  = [(board[i] if i < len(board) else '??') for i in range(3)]
    turn  = board[3] if len(board) > 3 else '??'
    river = board[4] if len(board) > 4 else '??'
    txt(f"Flop:  {' '.join(flop)}", 10, y, 0.44, (180, 180, 100)); y += 18
    txt(f'Turn:  {turn}',           10, y, 0.44, (180, 180, 100)); y += 18
    txt(f'River: {river}',          10, y, 0.44, (180, 180, 100)); y += 20

    hline(height - 70)
    cy = height - 62
    for line in ['ENTER=confirm  BKSP=undo  R=reset',
                 '1-5=fold/unfold player',
                 'D=debug  Q=quit']:
        txt(line, 10, cy, 0.34, (80, 80, 80))
        cy += 16

    return panel


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        rec = CardRecognizer()
    except (FileNotFoundError, ImportError) as e:
        print(f'Error: {e}')
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera.')
        sys.exit(1)

    print('Board Cam')
    print('  Phase 1: show each player\'s 2 cards at once')
    print('  Phase 2: community cards one at a time')
    print('  ENTER=confirm  BKSP=undo  R=reset  1-5=fold  Q=quit\n')

    players    = [{'cards': [None, None], 'folded': False}
                  for _ in range(N_PLAYERS)]
    board      = []
    equity_map = None
    history    = []

    phase          = 'players'
    current_player = next_active_player(0, players)
    community_slot = 0

    # Consecutive-frame stability counters — one per card position (0=left, 1=right)
    stable_count = [0, 0]
    stable_label = [None, None]

    show_debug   = False

    # ── Actions ───────────────────────────────────────────────────────────────

    def _used():
        used = set(board)
        for p in players:
            used.update(c for c in p['cards'] if c)
        return used

    def confirm_player_card(pos, label):
        nonlocal equity_map
        if current_player >= N_PLAYERS or phase != 'players':
            return False
        p = players[current_player]
        if p['cards'][pos] is not None:
            return False
        if label in _used():
            return False
        p['cards'][pos] = label
        stable_count[pos] = 0
        stable_label[pos] = None
        history.append(('player_card', current_player, pos, label))
        print(f'  P{current_player + 1} card {pos + 1}: {label}')
        if p['cards'][0] and p['cards'][1]:
            print(f'  P{current_player + 1}: {p["cards"][0]} {p["cards"][1]}  — press ENTER for next player')
        equity_map = recalc_equity(board, players)
        return True

    def _advance_player():
        nonlocal current_player, phase
        current_player = next_active_player(current_player + 1, players)
        if current_player >= N_PLAYERS:
            phase = 'community'
            print('  All players done — dealing community cards')
        stable_count[0] = stable_count[1] = 0
        stable_label[0] = stable_label[1] = None

    def confirm_community(label):
        nonlocal community_slot, equity_map
        if community_slot >= TOTAL_BOARD or phase != 'community':
            return False
        if label in _used():
            return False
        board.append(label)
        history.append(('board', community_slot, label))
        print(f'  {BOARD_SLOTS[community_slot]}: {label}')
        community_slot += 1
        stable_count[0] = 0
        stable_label[0] = None
        equity_map = recalc_equity(board, players)
        return True

    def undo():
        nonlocal current_player, phase, community_slot, equity_map
        if not history:
            return
        stable_count[0] = stable_count[1] = 0
        stable_label[0] = stable_label[1] = None
        action = history.pop()
        if action[0] == 'board':
            board.pop()
            community_slot = action[1]
            phase = 'community'
            print(f'  Undo: removed {action[2]}')
        else:
            players[action[1]]['cards'][action[2]] = None
            current_player = action[1]
            phase = 'players'
            print(f'  Undo: cleared P{action[1]+1} card {action[2]+1} ({action[3]})')
        equity_map = recalc_equity(board, players)

    def reset_all():
        nonlocal phase, current_player, community_slot, equity_map
        for p in players:
            p['cards'] = [None, None]
            p['folded'] = False
        board.clear(); history.clear()
        phase = 'players'; current_player = 0; community_slot = 0
        stable_count[0] = stable_count[1] = 0
        stable_label[0] = stable_label[1] = None
        equity_map = None
        print('  Reset.')

    # ── Capture loop ───────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.resize(frame, (CAM_W, CAM_H))
        fh, fw = frame.shape[:2]

        # ── Single inference pass (used for both display and tracking) ──────────
        cards        = detect_cards(frame)
        best_overall = None
        card_preds   = []   # (cx, label, conf, corners) sorted left-to-right

        for corners, warped, corner_crop in cards:
            results          = rec.predict(corner_crop, top_k=3)
            label, conf      = results[0]
            color            = _conf_color(conf)
            cx, cy           = corners.reshape(4, 2).mean(axis=0).astype(int)

            if best_overall is None or conf > best_overall[1]:
                best_overall = (label, conf, results)

            # Card outline + label — same as recognize_card.py
            cv2.drawContours(frame, [corners], -1, color, 3)
            label_text = f'{label}  {conf * 100:.0f}%'
            (tw, th), _ = cv2.getTextSize(label_text,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            tx = max(0, min(cx - tw // 2, fw - tw))
            ty = max(th + 4, cy - 14)
            cv2.rectangle(frame, (tx - 4, ty - th - 4),
                          (tx + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            card_preds.append((cx, label, conf, corners))

        card_preds.sort(key=lambda x: x[0])   # left-to-right → card 1, card 2

        # Label each card on-frame as "Card 1" / "Card 2" (or "Flop 1" etc.)
        slot_names = []
        if phase == 'players' and current_player < N_PLAYERS:
            p           = players[current_player]
            unconfirmed = [i for i in range(2) if p['cards'][i] is None]
            slot_names  = [f'Card {pos+1}' for pos in unconfirmed]
        elif phase == 'community' and community_slot < TOTAL_BOARD:
            slot_names = [BOARD_SLOTS[community_slot]]

        for det_i, (cx, label, conf, corners) in enumerate(card_preds):
            if det_i < len(slot_names):
                sname = slot_names[det_i]
                scolor = (80, 220, 255)
                cv2.putText(frame, sname,
                            (int(cx) - 30, corners.reshape(4,2)[:,1].min().astype(int) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, scolor, 2)

        # ── Auto-confirm logic ─────────────────────────────────────────────────
        def _try_confirm(pos, label, conf):
            if label in _used():
                return
            # Accumulate stable frames regardless of confidence
            if label == stable_label[pos]:
                stable_count[pos] += 1
            else:
                stable_count[pos] = 1
                stable_label[pos] = label
            # High confidence: confirm after CONF_INSTANT_FRAMES consecutive frames
            # Low confidence: confirm after AUTO_FRAMES consecutive frames
            threshold = CONF_INSTANT_FRAMES if conf >= CONF_INSTANT else AUTO_FRAMES
            if stable_count[pos] >= threshold:
                if phase == 'players':
                    confirm_player_card(pos, label)
                else:
                    confirm_community(label)

        if phase == 'players' and current_player < N_PLAYERS:
            p           = players[current_player]
            unconfirmed = [i for i in range(2) if p['cards'][i] is None]

            if not unconfirmed:
                pass   # both cards confirmed — waiting for ENTER, skip tracking
            elif not card_preds:
                stable_count[0] = stable_count[1] = 0
                stable_label[0] = stable_label[1] = None
            else:
                for slot_i, pos in enumerate(unconfirmed):
                    if slot_i >= len(card_preds):
                        continue   # card not visible this frame — pause, don't reset
                    _, label, conf, _ = card_preds[slot_i]
                    _try_confirm(pos, label, conf)

        elif phase == 'community' and community_slot < TOTAL_BOARD:
            if card_preds:
                # Highest-confidence detection for community
                _, label, conf, _ = max(card_preds, key=lambda x: x[2])
                _try_confirm(0, label, conf)
            else:
                stable_count[0] = 0
                stable_label[0] = None

        # Debug mask
        if show_debug:
            from card_detector import _build_mask
            mask  = _build_mask(frame)
            tint  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            tint[:, :, 0] = 0
            frame = cv2.addWeighted(frame, 0.65, tint, 0.35, 0)

        # Bottom bar (reuse recognize_card.py style)
        BAR_H = 56
        bar   = np.full((BAR_H, fw, 3), (30, 30, 30), dtype=np.uint8)
        if best_overall:
            label, conf, top3 = best_overall
            color = _conf_color(conf)
            cv2.putText(bar, label, (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            bx, by, bw, bh = 110, 14, fw - 260, 28
            cv2.rectangle(bar, (bx, by), (bx + bw, by + bh), (70, 70, 70), -1)
            cv2.rectangle(bar, (bx, by), (bx + int(bw * conf), by + bh), color, -1)
            cv2.rectangle(bar, (bx, by), (bx + bw, by + bh), (120, 120, 120), 1)
            cv2.putText(bar, f'{conf*100:.1f}%', (bx + bw + 8, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            for i, (lb, pr, *_) in enumerate(top3):
                cv2.putText(bar, f'{lb} {pr*100:.0f}%', (fw - 130, 16 + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _conf_color(pr), 1)
        else:
            cv2.putText(bar, 'No card detected', (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        cam_with_bar = np.vstack([frame, bar])
        sidebar = draw_sidebar(phase, current_player, community_slot,
                               players, board, equity_map, cam_with_bar.shape[0])
        cv2.imshow('Board Cam', np.hstack([cam_with_bar, sidebar]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_all()
        elif key == ord('d'):
            show_debug = not show_debug
        elif key in (8, 127):
            undo()
        elif key == 13:   # ENTER
            if phase == 'players' and current_player < N_PLAYERS:
                p = players[current_player]
                unconfirmed = [i for i in range(2) if p['cards'][i] is None]
                if not unconfirmed:
                    # Both cards confirmed — advance to next player
                    _advance_player()
                else:
                    # Force-confirm any unconfirmed slots with current best
                    for slot_i, pos in enumerate(unconfirmed):
                        if slot_i < len(card_preds) and stable_label[pos]:
                            confirm_player_card(pos, stable_label[pos])
            elif phase == 'community' and community_slot < TOTAL_BOARD:
                if stable_label[0]:
                    confirm_community(stable_label[0])
        elif ord('1') <= key <= ord('0') + N_PLAYERS:
            pnum = key - ord('0')
            pi   = pnum - 1
            p    = players[pi]
            p['folded'] = not p['folded']
            print(f"  Player {pnum} {'folded' if p['folded'] else 'active'}")
            if p['folded'] and phase == 'players' and pi == current_player:
                p['cards'] = [None, None]
                history[:] = [a for a in history
                               if not (a[0] == 'player_card' and a[1] == pi)]
                current_player = next_active_player(pi + 1, players)
                if current_player >= N_PLAYERS:
                    phase = 'community'
                stable_count[0] = stable_count[1] = 0
                stable_label[0] = stable_label[1] = None
            equity_map = recalc_equity(board, players)

    cap.release()
    cv2.destroyAllWindows()
    if board:
        print(f'\nFinal board: {" ".join(board)}')


if __name__ == '__main__':
    main()
