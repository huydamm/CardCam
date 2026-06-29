#!/usr/bin/env python3
"""
Card Detector — OpenCV playing card detection and extraction

Two detection modes work together:
  1. FULL CARD  — detects the whole card (community cards flat on table)
  2. CORNER PIP — detects just the rank+suit corner (hole cards being peeked)

Both modes output the same warped images for the ML model to classify.

Press D in the preview window to toggle a debug view of the edge mask.

Tips for best detection:
  - Use a plain dark background
  - Good lighting, no glare
  - For full card: keep all 4 corners visible
  - For corner pip: peel the card so only the corner rank/suit is visible

Requirements: pip install opencv-python numpy
"""

import cv2
import numpy as np

# Full card output dimensions
CARD_W = 200
CARD_H = 300

# Corner pip output dimensions (rank + suit printed in card corner)
# Standard card corner area is roughly 15% of card width, 22% of card height
# We warp it to a fixed size for the ML model
CORNER_W = 70
CORNER_H = 100

# ── Area thresholds ───────────────────────────────────────────────────────────
# Full card (flat on table or held flat)
MIN_CARD_AREA    = 2_000
MAX_CARD_AREA    = 400_000

# Corner pip (just the bent corner visible during a peek)
MIN_CORNER_AREA  = 400
MAX_CORNER_AREA  = 8_000

# Aspect ratio of a playing card corner pip — roughly square-ish
CORNER_ASPECT_MIN = 0.35
CORNER_ASPECT_MAX = 1.20   # can be taller than wide when peeked at steep angle

# Polygon approximation tolerance
POLY_EPS = 0.03

# Full card aspect ratio (w/h) with slack for perspective
ASPECT_MIN = 0.40
ASPECT_MAX = 0.95


# ── Corner crop from a warped full card ───────────────────────────────────────
# When we DO detect the full card, we crop the top-left corner pip
# so the ML model also trains on corner-only views.
CORNER_CROP_X  = int(CARD_W * 0.00)   # left edge
CORNER_CROP_X2 = int(CARD_W * 0.28)  # ~28% of card width  (tighter — pip only)
CORNER_CROP_Y  = int(CARD_H * 0.00)  # top edge
CORNER_CROP_Y2 = int(CARD_H * 0.30)  # ~30% of card height (rank + full suit pip)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype('float32')
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype='float32')


def warp_region(frame: np.ndarray, corners: np.ndarray,
                out_w: int, out_h: int) -> np.ndarray:
    """Perspective-warp any 4-corner region to out_w × out_h."""
    src = order_points(corners)
    w = (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2
    h = (np.linalg.norm(src[3] - src[0]) + np.linalg.norm(src[2] - src[1])) / 2

    if w > h:
        # Landscape detected → rotate to portrait
        dst = np.array([
            [0,         out_h - 1],
            [0,         0        ],
            [out_w - 1, 0        ],
            [out_w - 1, out_h - 1],
        ], dtype='float32')
    else:
        dst = np.array([
            [0,         0        ],
            [out_w - 1, 0        ],
            [out_w - 1, out_h - 1],
            [0,         out_h - 1],
        ], dtype='float32')

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (out_w, out_h))


def crop_corner_from_card(warped_card: np.ndarray) -> np.ndarray:
    """
    Crop the top-left corner pip from a full warped card image.
    Returns a CORNER_W × CORNER_H image ready for ML inference.
    """
    crop = warped_card[CORNER_CROP_Y:CORNER_CROP_Y2,
                       CORNER_CROP_X:CORNER_CROP_X2]
    return cv2.resize(crop, (CORNER_W, CORNER_H))


# ── Edge mask ─────────────────────────────────────────────────────────────────

def _white_card_mask(frame: np.ndarray) -> np.ndarray:
    """
    Isolate white/light card regions from any background using HSV color.
    Playing cards are nearly white: low saturation, high brightness.
    This works regardless of background color — the card stands out by whiteness.
    Returns a binary mask of white regions with card-border edges.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White: any hue, very low saturation, high value
    white_mask = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))

    # Morphology: fill holes inside the card, keep the solid white rectangle
    k_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    filled = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k_fill)

    # Get edges of the white region — that's the card border
    edges = cv2.Canny(filled, 50, 150)
    k_d   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k_d, iterations=2)
    return edges


def _build_mask(frame_or_gray) -> np.ndarray:
    """
    Three methods combined for maximum robustness:
      1. Adaptive threshold — great on flat surfaces / even lighting
      2. Canny edges        — good contrast between card and background
      3. White card mask    — works on ANY background (isolates by card colour)
    Accepts either a BGR frame or a grayscale image.
    """
    if len(frame_or_gray.shape) == 3:
        gray  = cv2.cvtColor(frame_or_gray, cv2.COLOR_BGR2GRAY)
        frame = frame_or_gray
    else:
        gray  = frame_or_gray
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    adapt = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )
    canny = cv2.Canny(blur, 30, 100)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canny = cv2.dilate(canny, k, iterations=2)

    white = _white_card_mask(frame)

    combined = cv2.bitwise_or(adapt, canny)
    combined = cv2.bitwise_or(combined, white)

    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k2)


# ── Full card detection ───────────────────────────────────────────────────────

def detect_cards(frame: np.ndarray, debug: bool = False):
    """
    Detect full playing cards (community cards on the table).
    Returns list of (corners, warped_full [CARD_H×CARD_W], corner_crop [CORNER_H×CORNER_W]).
    corner_crop is the top-left pip cropped from each full card.

    If debug=True, returns (cards_list, edge_mask).
    """
    mask = _build_mask(frame)   # pass full BGR so white-mask method has colour info

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CARD_AREA < area < MAX_CARD_AREA):
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, POLY_EPS * peri, True)
        if len(approx) != 4:
            continue

        ordered = order_points(approx)
        w = (np.linalg.norm(ordered[1] - ordered[0]) +
             np.linalg.norm(ordered[2] - ordered[3])) / 2
        h = (np.linalg.norm(ordered[3] - ordered[0]) +
             np.linalg.norm(ordered[2] - ordered[1])) / 2
        aspect = min(w, h) / (max(w, h) + 1e-6)
        if not (ASPECT_MIN < aspect < ASPECT_MAX):
            continue

        warped = warp_region(frame, approx, CARD_W, CARD_H)
        corner = crop_corner_from_card(warped)
        cards.append((approx, warped, corner, area))

    cards.sort(key=lambda x: x[3], reverse=True)
    result = [(c, w, cr) for c, w, cr, _ in cards]

    if debug:
        return result, mask
    return result


# ── Corner pip detection (peeked hole cards) ──────────────────────────────────

def detect_corners(frame: np.ndarray, debug: bool = False):
    """
    Detect corner pips only — for hole cards being peeked by players.
    Returns list of (corners, warped_corner [CORNER_H×CORNER_W]).

    This finds smaller rectangles that match the aspect ratio of
    a card's corner pip area.
    """
    mask = _build_mask(frame)   # pass full BGR so white-mask method has colour info

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pips = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CORNER_AREA < area < MAX_CORNER_AREA):
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, POLY_EPS * peri, True)
        if len(approx) != 4:
            continue

        ordered = order_points(approx)
        w = (np.linalg.norm(ordered[1] - ordered[0]) +
             np.linalg.norm(ordered[2] - ordered[3])) / 2
        h = (np.linalg.norm(ordered[3] - ordered[0]) +
             np.linalg.norm(ordered[2] - ordered[1])) / 2
        aspect = min(w, h) / (max(w, h) + 1e-6)
        if not (CORNER_ASPECT_MIN < aspect < CORNER_ASPECT_MAX):
            continue

        warped = warp_region(frame, approx, CORNER_W, CORNER_H)
        pips.append((approx, warped, area))

    pips.sort(key=lambda x: x[2], reverse=True)
    result = [(c, w) for c, w, _ in pips]

    if debug:
        return result, mask
    return result


def draw_detections(frame: np.ndarray, cards: list,
                    pips: list = None) -> np.ndarray:
    """Draw full card outlines (green) and corner pip outlines (yellow)."""
    out = frame.copy()
    for i, (corners, _, _) in enumerate(cards):
        cv2.drawContours(out, [corners], -1, (0, 255, 0), 2)
        cx, cy = corners.reshape(4, 2).mean(axis=0).astype(int)
        cv2.putText(out, f'Card {i + 1}', (cx - 28, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if pips:
        for i, (corners, _) in enumerate(pips):
            cv2.drawContours(out, [corners], -1, (0, 220, 255), 2)
            cx, cy = corners.reshape(4, 2).mean(axis=0).astype(int)
            cv2.putText(out, f'Pip {i + 1}', (cx - 25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    return out


# ── Standalone preview mode ───────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: cannot open webcam.')
        return

    print('Card Detector live preview')
    print('  D  — toggle debug mask view')
    print('  S  — save screenshot')
    print('  Q  — quit')
    print()
    print('GREEN outline  = full card detected')
    print('YELLOW outline = corner pip detected (peeked card)')

    show_debug = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cards, mask = detect_cards(frame, debug=True)
        pips        = detect_corners(frame)
        out         = draw_detections(frame, cards, pips)

        # Tile full cards along the top
        tile_x = 5
        for _, warped, corner in cards[:5]:
            small_h = 90
            small_w = int(small_h * CARD_W / CARD_H)
            small   = cv2.resize(warped, (small_w, small_h))
            # Show corner crop beside the full card
            c_small = cv2.resize(corner, (int(90 * CORNER_W / CORNER_H), 90))
            y1 = 5
            if tile_x + small_w + c_small.shape[1] + 4 < out.shape[1]:
                out[y1:y1+small_h, tile_x:tile_x+small_w] = small
                cv2.rectangle(out, (tile_x, y1),
                              (tile_x+small_w, y1+small_h), (0,255,0), 1)
                cx2 = tile_x + small_w + 2
                out[y1:y1+90, cx2:cx2+c_small.shape[1]] = c_small
                cv2.rectangle(out, (cx2, y1),
                              (cx2+c_small.shape[1], y1+90), (0,220,255), 1)
            tile_x += small_w + c_small.shape[1] + 8

        # Tile corner pips below full cards (second row)
        tile_x2 = 5
        for _, warped in pips[:8]:
            small = cv2.resize(warped, (int(90 * CORNER_W / CORNER_H), 90))
            y1 = 100
            if tile_x2 + small.shape[1] < out.shape[1]:
                out[y1:y1+90, tile_x2:tile_x2+small.shape[1]] = small
                cv2.rectangle(out, (tile_x2, y1),
                              (tile_x2+small.shape[1], y1+90), (0,220,255), 1)
            tile_x2 += small.shape[1] + 4

        n_c = len(cards)
        n_p = len(pips)
        cv2.putText(out,
                    f'{n_c} full card(s)  {n_p} pip(s)  |  D=debug  Q=quit',
                    (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if show_debug:
            mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            tint      = mask_bgr.copy(); tint[:, :, 0] = 0
            dbg       = cv2.addWeighted(out, 0.6, tint, 0.4, 0)
            cv2.putText(dbg, 'DEBUG: edge mask (cyan)',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            cv2.imshow('Card Detector', dbg)
        else:
            cv2.imshow('Card Detector', out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('s'):
            cv2.imwrite('screenshot.jpg', out)
            print('Saved screenshot.jpg')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
