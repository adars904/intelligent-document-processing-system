"""
train_v2.py
-----------
Same architecture, same callbacks, same epochs as train.py.
ONLY difference: class_weight is computed from the train split and
passed into model.fit(). This isolates class weighting as the single
variable being tested against the baseline run.

Usage (Colab):
    !python train_v2.py

Then compare against baseline with:
    !python evaluate.py --model final_model_v2.keras --tag v2_class_weighted
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
    Rescaling,
)

from data_pipeline import get_datasets, MANIFEST_PATH
import pandas as pd

# ---- Config ----
EPOCHS = 30
BEST_MODEL_PATH = "best_model_v2.keras"
FINAL_MODEL_PATH = "final_model_v2.keras"
HISTORY_PLOT_PATH = "training_curves_v2.png"


def build_model(num_classes: int) -> Sequential:
    # Identical to train.py -- kept in sync deliberately so the only
    # difference between baseline and v2 is class_weight.
    model = Sequential()
    model.add(Rescaling(1 / 255, input_shape=(160, 160, 3)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Dropout(0.25))

    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def compute_weights(class_names):
    """Compute class weights from the TRAIN split only (never val/test)."""
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df["split"] == "train"]

    label_map = {name: idx for idx, name in enumerate(class_names)}
    y_train = train_df["class_name"].map(label_map).values

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=y_train,
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
    print("Computed class weights:", dict(zip(class_names, [class_weight_dict[i] for i in range(len(class_names))])))
    return class_weight_dict


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Accuracy (v2, class-weighted)')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss (v2, class-weighted)')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.show()
    print(f"Saved curves to {HISTORY_PLOT_PATH}")


def main():
    train_ds, val_ds, test_ds, class_names = get_datasets()
    num_classes = len(class_names)

    class_weights = compute_weights(class_names)

    model = build_model(num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,   # <-- the only real difference from train.py
    )

    model.save(FINAL_MODEL_PATH)
    print(f"Saved final model to {FINAL_MODEL_PATH}")
    print(f"Best checkpoint saved separately to {BEST_MODEL_PATH}")

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    with open("v2_results.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "epochs_run": len(history.history["loss"]),
                "class_weights_used": True,
                "class_weights": class_weights,
                "class_names": class_names,
            },
            f,
            indent=2,
        )
    print("Saved v2_results.json")

    plot_history(history)


if __name__ == "__main__":
    main()
