# CardCam

**Point a camera at the table and know your odds in real time.**

CardCam is a poker hand equity calculator with a twist: it can *see* the cards. A camera spots the playing cards, a neural net reads them, and an exact equity engine tells every player their real chance of winning the pot — at every street, as the hand plays out. Built to run on a Raspberry Pi 5 with an ArduCam, so the whole thing fits next to the table.

## What it does

- 🃏 **Recognizes real cards** — detects playing cards from a live camera feed and classifies all 52 of them with a trained CNN.
- 🎯 **Calculates true equity** — exact win probability for 5-player Texas Hold'em, recalculated as the flop, turn, and river come down.
- 📈 **Shows the swings** — ▲▼ arrows after every street show who the new card helped and who it sank.
- 🛠️ **End-to-end pipeline** — tools to collect your own training data, train the model, and run inference, all included.
- 🥧 **Runs on a Pi** — designed for Raspberry Pi 5 + ArduCam, with a quantized TFLite model for on-device speed.

## How the equity engine works

Every card is a single integer (0–51), and any 5-card hand collapses into a comparable tuple — so Python's built-in tuple comparison settles every tiebreaker for free, no lookup tables. From there CardCam picks the method that's both fast *and* exact:

| Street | Cards to come | Method | Result |
|--------|---------------|--------|--------|
| River | 0 | direct | exact |
| Turn | 1 | check all ~38 rivers | exact |
| Flop | 2 | check all ~741 turn+river combos | exact |
| Preflop | 5 | Monte Carlo, 20,000 boards | ±0.7% |

Exhaustive once the board is small enough to be instant; Monte Carlo only preflop, where the full 850k-combo enumeration would be too slow. The full derivation lives in [`EQUITY_EXPLAINED.md`](EQUITY_EXPLAINED.md).

## How the vision pipeline works

1. **Detect** (`card_detector.py`) — OpenCV finds cards via adaptive thresholding, contour + quadrilateral filtering, and an aspect-ratio check, then perspective-warps each one to a clean 200×300 image.
2. **Collect** (`collect_samples.py`) — label detected cards on the fly to build a dataset (`samples/Ah/`, `samples/Kd/`, … — aim for 50+ per class).
3. **Train** (`train_model.py`) — a **MobileNetV2** classifier trained in two phases (frozen head, then fine-tuned), exported to Keras *and* quantized TFLite for the Pi.
4. **Recognize** (`recognize_card.py`) — live webcam feed with predicted card and confidence.

## Quick start

**Just the calculator (no camera):**

```bash
pip install numpy
python poker_equity.py
```

Enter each player's hole cards, then the flop, turn, and river as they're dealt.

**The full pipeline:**

```bash
pip install opencv-python numpy        # detection + collection
python collect_samples.py              # build a dataset
pip install tensorflow matplotlib      # training (run on a PC, not the Pi)
python train_model.py
python recognize_card.py               # test recognition
```

## Card format

Two characters, `[Rank][Suit]` — ranks `2-9 T J Q K A`, suits `c d h s`. So `Ah` = Ace of Hearts, `Kd` = King of Diamonds.

## Roadmap

- [x] Equity calculator (preflop → river)
- [x] OpenCV card detection + perspective extraction
- [x] Training-data collector
- [x] MobileNetV2 training + TFLite export
- [x] Inference module (PC Keras + Pi TFLite)
- [ ] Integration: camera → model → auto equity
- [ ] Pi 5 + ArduCam deployment

## Tech stack

Python · OpenCV · TensorFlow / Keras (MobileNetV2) · TFLite · NumPy · Raspberry Pi 5 + ArduCam

---

*Use it to study the math, not to cheat at the table.*
