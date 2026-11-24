"""
Train a CNN hand-gesture classifier on the LeapGestRecog dataset.

Dataset: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
Layout : leapGestRecog/<subject 00-09>/<gesture folder>/*.png  (10 gestures)

The split is done BY SUBJECT (train: 00-06, val: 07-08, test: 09) so the test
score reflects performance on unseen people, not just unseen frames.

Usage:  python train.py --data path/to/leapGestRecog --epochs 15
"""
import argparse, json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models

IMG = 64


def find_root(p: Path) -> Path:
    # Kaggle zips often nest leapGestRecog/leapGestRecog/00/...
    while not (p / "00").exists():
        inner = [d for d in p.iterdir() if d.is_dir()]
        if not inner:
            raise FileNotFoundError("Could not find subject folders (00..09).")
        p = inner[0]
    return p


def load_dataset(root: Path):
    subjects = sorted(d for d in root.iterdir() if d.is_dir())
    classes = sorted(d.name for d in subjects[0].iterdir() if d.is_dir())
    X, y, subj = [], [], []
    for si, sdir in enumerate(subjects):
        for ci, cname in enumerate(classes):
            for f in (sdir / cname).glob("*.png"):
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                X.append(cv2.resize(img, (IMG, IMG)))
                y.append(ci)
                subj.append(si)
    X = np.asarray(X, dtype="uint8")[..., None]
    return X, np.asarray(y), np.asarray(subj), classes


def build_model(n_classes: int) -> tf.keras.Model:
    aug = models.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomContrast(0.2),
    ], name="augment")

    def block(x, f):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        return layers.Dropout(0.25)(x)

    inp = layers.Input((IMG, IMG, 1))
    x = layers.Rescaling(1 / 255.0)(inp)
    x = aug(x)
    for f in (32, 64, 128):
        x = block(x, f)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="gesture_model.keras")
    a = ap.parse_args()

    X, y, subj, classes = load_dataset(find_root(Path(a.data)))
    print(f"Loaded {len(X)} images, classes: {classes}")

    tr, va, te = subj <= 6, (subj == 7) | (subj == 8), subj == 9
    model = build_model(len(classes))
    cbs = [
        callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]
    hist = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]),
                     epochs=a.epochs, batch_size=a.batch, callbacks=cbs)

    # --- evaluation on unseen subject ---
    pred = model.predict(X[te]).argmax(1)
    print(classification_report(y[te], pred, target_names=classes))
    cm = confusion_matrix(y[te], pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    plt.figure(figsize=(9, 3.5))
    for i, k in enumerate(("accuracy", "loss")):
        plt.subplot(1, 2, i + 1)
        plt.plot(hist.history[k], label="train")
        plt.plot(hist.history["val_" + k], label="val")
        plt.title(k); plt.legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)

    model.save(a.out)
    json.dump(classes, open("classes.json", "w"))
    print("Saved", a.out, "and classes.json")


if __name__ == "__main__":
    main()
