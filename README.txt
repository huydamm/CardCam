================================================================
  POKERCAM — Poker Hand Equity Calculator with Card Recognition
================================================================

A three-part system that detects playing cards from a camera,
collects training data for an ML model, and calculates real-time
poker hand equity for 5-player Texas Hold'em.

Designed to run on a Raspberry Pi 5 with an ArduCam module.

----------------------------------------------------------------
  FILES
----------------------------------------------------------------

  poker_equity.py
    Interactive equity calculator for 5-player Texas Hold'em.
    Enter each player's hole cards manually, then input the
    flop, turn, and river as they are dealt. Displays equity
    percentages and hand labels at every street with change
    arrows showing how equity shifts.

  card_detector.py
    OpenCV-based card detection engine (used as a library by
    collect_samples.py and future ML integration).
    Detects playing cards in a webcam/camera feed using:
      - Adaptive thresholding
      - Contour detection with quadrilateral filtering
      - Aspect ratio validation (standard card ~0.71)
      - Perspective transform to a clean 200x300 portrait image
    Run directly for a live detection preview.

  collect_samples.py
    Dataset builder for training the card recognition ML model.
    Shows the camera feed, detects cards, and lets you label
    and save each one. Saves to samples/<label>/ folders
    (e.g. samples/Ah/, samples/Kd/).
    Target: 50+ samples per class (52 classes = 2,600+ images).

----------------------------------------------------------------
  REQUIREMENTS
----------------------------------------------------------------

  Python 3.13  (recommended — 3.14 has unstable numpy wheels)

  Install dependencies:
    pip install opencv-python numpy

----------------------------------------------------------------
  HOW TO USE
----------------------------------------------------------------

  Step 1 — Collect training samples
    python collect_samples.py

    Controls:
      Type rank (2-9 T J Q K A) then suit (c d h s) to label
      ENTER   save the detected card with that label
      SPACE   freeze / unfreeze the camera frame
      ESC     clear current label  (or quit if label is empty)

    Example: press A then H to label "Ah" (Ace of Hearts),
    then press ENTER to save.

  Step 2 — Train the CNN  (run on PC, not Pi 5)
    pip install tensorflow matplotlib
    python train_model.py

    Trains a MobileNetV2 classifier in two phases:
      Phase 1 — head only (base frozen)
      Phase 2 — fine-tune top base layers
    Outputs to model/ folder:
      card_model.keras   full Keras model (PC)
      card_model.tflite  quantized model  (copy to Pi 5)
      labels.json        class name list  (copy to Pi 5)
      training_curves.png

  Step 3 — Test card recognition
    python recognize_card.py
    Shows live webcam feed with predicted card label and confidence.

  Step 4 — Run live equity calculator  (coming soon)
    Camera feed → ML model → auto-input into poker_equity.py

  Manual equity calculator (no camera needed):
    python poker_equity.py
    Follow the prompts to enter cards for each player and street.

----------------------------------------------------------------
  CARD FORMAT
----------------------------------------------------------------

  Two characters: [Rank][Suit]

  Ranks : 2 3 4 5 6 7 8 9 T J Q K A
  Suits : c (clubs)  d (diamonds)  h (hearts)  s (spades)

  Examples:
    Ah  = Ace of Hearts
    Kd  = King of Diamonds
    Tc  = Ten of Clubs
    2s  = Two of Spades

----------------------------------------------------------------
  HARDWARE TARGET
----------------------------------------------------------------

  Raspberry Pi 5 + ArduCam module
  All dependencies (opencv-python, numpy) run natively on Pi OS.
  Use cv2.VideoCapture(0) or cv2.VideoCapture(0, cv2.CAP_V4L2)
  depending on your ArduCam connection type.

----------------------------------------------------------------
  PROJECT ROADMAP
----------------------------------------------------------------

  [x] Poker equity calculator (preflop / flop / turn / river)
  [x] Card detection and perspective extraction (OpenCV)
  [x] Training data collector with progress tracking
  [x] CNN training script (MobileNetV2, TFLite export)
  [x] Inference module (PC Keras + Pi 5 TFLite)
  [ ] Integration: camera -> ML -> auto equity calculation
  [ ] Pi 5 + ArduCam deployment

================================================================
