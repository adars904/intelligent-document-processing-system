import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from data_pipeline import get_datasets


train_ds, val_ds, test_ds, class_names = get_datasets()

base_model = MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # freeze the pretrained base, only train our head

model = Sequential([
    Rescaling(scale=1./127.5, offset=-1, input_shape=(160, 160, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.25),
    Dense(5, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint("transfer_model.keras", monitor="val_loss", save_best_only=True),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_ds)
print(f"test accuracy: {test_acc:.4f}")

results = {
    "test_accuracy": test_acc,
    "test_loss": test_loss,
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
}

with open("transfer_results.json", "w") as f:
    json.dump(results, f, indent=2)
