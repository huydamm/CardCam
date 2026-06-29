#!/usr/bin/env python3
"""
Card Recognition CNN Training
Trains a MobileNetV2 classifier on the samples/ dataset.

  Run on PC (not Pi 5) — training is CPU/GPU heavy.
  Outputs:
    model/card_model.keras   — full Keras model (PC use)
    model/card_model.tflite  — quantized TFLite model (copy to Pi 5)
    model/labels.json        — class name list (copy to Pi 5)
    model/training_curves.png

Two-phase training:
  Phase 1 — freeze MobileNetV2 base, train only the head
  Phase 2 — unfreeze top layers, fine-tune with a low learning rate

Requirements:
  pip install tensorflow matplotlib
"""

import os
import sys
import json

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
except ImportError:
    print('TensorFlow not found.  Run:  pip install tensorflow')
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')           # save to file, no display required
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLES_DIR    = 'samples'
MODEL_DIR      = 'model'
IMG_SIZE       = (96, 96)       # MobileNetV2 input (faster than 224x224 on Pi)
BATCH_SIZE     = 32
VAL_SPLIT      = 0.20
SEED           = 42
EPOCHS_PHASE1  = 20             # head-only training
EPOCHS_PHASE2  = 20             # fine-tuning (early stopping will cut these short)
FINE_TUNE_FROM = -30            # unfreeze last N layers of base in phase 2

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
ALL_LABELS = [r + s for r in RANKS for s in SUITS]


# ── Dataset helpers ───────────────────────────────────────────────────────────

def check_dataset() -> tuple:
    """Print a dataset summary. Returns (counts dict, empty labels, low labels)."""
    counts = {}
    for label in ALL_LABELS:
        folder = os.path.join(SAMPLES_DIR, label)
        if os.path.isdir(folder):
            counts[label] = len([
                f for f in os.listdir(folder) if f.lower().endswith('.jpg')
            ])
        else:
            counts[label] = 0

    total   = sum(counts.values())
    present = sum(1 for v in counts.values() if v > 0)
    empty   = [lb for lb, n in counts.items() if n == 0]
    low     = [lb for lb, n in counts.items() if 0 < n < 15]

    print(f'\n  Dataset summary')
    print(f'  ---------------')
    print(f'  Total images : {total}')
    print(f'  Classes with data : {present} / 52')
    if low:
        print(f'  Low sample count (<15) : {" ".join(low)}')
    if empty:
        print(f'  No samples at all : {" ".join(empty[:10])}{"..." if len(empty) > 10 else ""}')

    return counts, empty, low


def remove_empty_class_dirs():
    """Remove empty label folders so image_dataset_from_directory ignores them."""
    removed = []
    for label in ALL_LABELS:
        folder = os.path.join(SAMPLES_DIR, label)
        if os.path.isdir(folder):
            imgs = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
            if len(imgs) == 0:
                os.rmdir(folder)
                removed.append(label)
    if removed:
        print(f'  Skipping {len(removed)} empty class folders.')


def make_datasets() -> tuple:
    """
    Load images from SAMPLES_DIR and return (train_ds, val_ds, class_names).
    Only folders with at least 1 image are included.
    Applies augmentation to the training split.
    """
    remove_empty_class_dirs()

    # Augmentation: no flips (cards have asymmetric ranks/suits)
    augment = keras.Sequential([
        layers.RandomRotation(factor=0.05),   # ±18° for slight tilts
        layers.RandomZoom(height_factor=0.10),
        layers.RandomBrightness(factor=0.25),
        layers.RandomContrast(factor=0.20),
    ], name='augmentation')

    def augment_fn(x, y):
        return augment(x, training=True), y

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = keras.utils.image_dataset_from_directory(
        SAMPLES_DIR,
        validation_split=VAL_SPLIT,
        subset='training',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )
    val_ds = keras.utils.image_dataset_from_directory(
        SAMPLES_DIR,
        validation_split=VAL_SPLIT,
        subset='validation',
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )

    class_names = train_ds.class_names

    train_ds = (train_ds
                .map(augment_fn, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_classes: int) -> tuple:
    """
    MobileNetV2 base (ImageNet weights) + custom head.
    Returns (model, base_model).
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False      # frozen for phase 1

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name='card_input')
    # MobileNetV2 expects inputs scaled to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    return keras.Model(inputs, outputs, name='card_classifier'), base


# ── Plot ──────────────────────────────────────────────────────────────────────

def save_training_plot(h1, h2, out_path: str):
    if not HAS_MATPLOTLIB:
        return
    acc   = h1.history['accuracy']     + h2.history['accuracy']
    vacc  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss  = h1.history['loss']         + h2.history['loss']
    vloss = h1.history['val_loss']     + h2.history['val_loss']
    ep    = range(1, len(acc) + 1)
    split = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    for ax, train, val, title in [
        (ax1, acc,  vacc,  'Accuracy'),
        (ax2, loss, vloss, 'Loss'),
    ]:
        ax.plot(ep, train, label='Train')
        ax.plot(ep, val,   label='Val')
        ax.axvline(split, color='gray', linestyle='--', linewidth=1,
                   label='Fine-tune start')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'  Training curves -> {out_path}')


# ── TFLite export ─────────────────────────────────────────────────────────────

def export_tflite(model: keras.Model, out_path: str):
    """Convert and save a quantized TFLite model for Pi 5 deployment."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quant
    tflite_bytes = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_bytes)
    kb = os.path.getsize(out_path) / 1024
    print(f'  TFLite model    -> {out_path}  ({kb:.0f} KB)')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 57)
    print('  CARD RECOGNITION CNN  —  Training')
    print('=' * 57)

    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f'  GPU : {gpus[0].name}')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    else:
        print('  GPU : none — using CPU (training may take 30–90 min)')

    # Dataset check
    counts, empty, low = check_dataset()
    if not any(v > 0 for v in counts.values()):
        print('\n  No training images found.')
        print('  Run collect_samples.py first to build the dataset.')
        sys.exit(1)

    if empty:
        print(f'\n  {len(empty)} classes have no data — skipping them, training on present classes only.')

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    print('\n  Loading dataset...')
    train_ds, val_ds, class_names = make_datasets()
    n_classes = len(class_names)
    print(f'  Classes : {n_classes}')
    print(f'  Batch   : {BATCH_SIZE}')

    # Save label map (required for inference)
    labels_path = os.path.join(MODEL_DIR, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f'  Labels  -> {labels_path}')

    model, base = build_model(n_classes)

    common_callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=6,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.4, patience=3, verbose=1,
        ),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'checkpoint_best.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=0,
        ),
    ]

    # ── Phase 1: head only ────────────────────────────────────────────────
    print(f'\n  Phase 1 — Training head  (base frozen, up to {EPOCHS_PHASE1} epochs)')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_PHASE1, callbacks=common_callbacks,
    )

    # ── Phase 2: fine-tune top base layers ────────────────────────────────
    print(f'\n  Phase 2 — Fine-tuning top {abs(FINE_TUNE_FROM)} base layers'
          f'  (up to {EPOCHS_PHASE2} epochs)')
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_FROM]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS_PHASE2, callbacks=common_callbacks,
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    print('\n  Saving outputs...')
    keras_path  = os.path.join(MODEL_DIR, 'card_model.keras')
    tflite_path = os.path.join(MODEL_DIR, 'card_model.tflite')
    plot_path   = os.path.join(MODEL_DIR, 'training_curves.png')

    model.save(keras_path)
    print(f'  Keras model     -> {keras_path}')
    export_tflite(model, tflite_path)
    save_training_plot(h1, h2, plot_path)

    # ── Final evaluation ──────────────────────────────────────────────────
    print('\n  Evaluating on validation set...')
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f'  Val accuracy : {val_acc * 100:.1f}%')
    print(f'  Val loss     : {val_loss:.4f}')

    print('\n' + '=' * 57)
    print('  Done!  Files to copy to Pi 5:')
    print(f'    {tflite_path}')
    print(f'    {labels_path}')
    print('=' * 57 + '\n')


if __name__ == '__main__':
    main()
