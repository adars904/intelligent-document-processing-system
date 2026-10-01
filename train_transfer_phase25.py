import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from data_pipeline import get_datasets


# ---- Phase 2.5: combined old + modern data ----
TAG = "phase25"
MODEL_PHASE1 = f"transfer_model_{TAG}_phase1.keras"
MODEL_PHASE2 = f"transfer_model_{TAG}.keras"
RESULTS_PATH = f"transfer_results_{TAG}.json"


train_ds, val_ds, test_ds, class_names = get_datasets()
NUM_CLASSES = len(class_names)
print(f"Number of classes: {NUM_CLASSES}")
print(f"Tag: {TAG}")
print(f"Will save to: {MODEL_PHASE1}, {MODEL_PHASE2}, {RESULTS_PATH}")


# ---------- Class weights (fix ID Card vs Certificate imbalance) ----------
df = pd.read_csv("/content/manifest.csv")
train_df = df[df["split"] == "train"]

with open("class_indices.json") as f:
    label_map = json.load(f)

y_train = train_df["class_name"].map(label_map).values

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train,
)
class_weight = dict(enumerate(weights))
print("\nClass weights:",
      {class_names[i]: round(w, 3) for i, w in class_weight.items()})


# ---------- Data augmentation ----------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
], name="augmentation")

train_ds_aug = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)


# ---------- Model ----------
base_model = MobileNetV2(
    input_shape=(320, 320, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

model = Sequential([
    Rescaling(scale=1./127.5, offset=-1, input_shape=(320, 320, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.25),
    Dense(NUM_CLASSES, activation="softmax"),
])


# ============================================================
# PHASE 1: Train the head only (base frozen)
# ============================================================
print("\n=== Phase 1: Training head only ===")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint(MODEL_PHASE1, monitor="val_loss", save_best_only=True),
]

history1 = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_phase1,
    class_weight=class_weight,
)


# ============================================================
# PHASE 2: Fine-tune the top of the base model
# ============================================================
print("\n=== Phase 2: Fine-tuning top layers of MobileNetV2 ===")

base_model.trainable = True
FINE_TUNE_AT = len(base_model.layers) - 50

for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

for layer in base_model.layers[FINE_TUNE_AT:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_PHASE2, monitor="val_loss", save_best_only=True),
]

history2 = model.fit(
    train_ds_aug,
    validation_data=val_ds,
    epochs=25,
    callbacks=callbacks_phase2,
    class_weight=class_weight,
)


# ---------- Evaluate ----------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.4f}")

merged_history = {}
for k in history1.history:
    merged_history[k] = (
        [float(v) for v in history1.history[k]]
        + [float(v) for v in history2.history.get(k, [])]
    )

results = {
    "tag": TAG,
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "class_names": class_names,
    "class_weights": {class_names[i]: float(w) for i, w in class_weight.items()},
    "history": merged_history,
    "phase1_epochs": len(history1.history["loss"]),
    "phase2_epochs": len(history2.history["loss"]),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {RESULTS_PATH}")
print(f"Saved models: {MODEL_PHASE1}, {MODEL_PHASE2}")