"""
train.py
--------
Trains the 5-class Document AI CNN on the pipeline from data_pipeline.py.

Baseline run: NO class weights yet (deliberate -- see project notes).
Class weights will be added as a documented v2 improvement afterwards,
so you can report a before/after accuracy delta.

Usage (Colab):
    !python train.py
"""

import json

import matplotlib.pyplot as plt
import tensorflow as tf
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

from data_pipeline import get_datasets

# ---- Config ----
EPOCHS = 30
BEST_MODEL_PATH = "best_model.keras"
FINAL_MODEL_PATH = "final_model.keras"
HISTORY_PLOT_PATH = "training_curves.png"


def build_model(num_classes: int) -> Sequential:
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
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    plt.show()
    print(f"Saved curves to {HISTORY_PLOT_PATH}")


def main():
    train_ds, val_ds, test_ds, class_names = get_datasets()
    num_classes = len(class_names)

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
        # no class_weight here on purpose -- baseline run, see project notes
    )

    model.save(FINAL_MODEL_PATH)
    print(f"Saved final model (post-EarlyStopping restore_best_weights) to {FINAL_MODEL_PATH}")
    print(f"Best checkpoint during training saved separately to {BEST_MODEL_PATH}")

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    with open("baseline_results.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "epochs_run": len(history.history["loss"]),
                "class_weights_used": False,
                "class_names": class_names,
            },
            f,
            indent=2,
        )
    print("Saved baseline_results.json")

    plot_history(history)


if __name__ == "__main__":
    main()
