import json
import os

import pandas as pd
import tensorflow as tf

# ---- Config ----
DATASET_ROOT = "/content/dataset/dataset"
MANIFEST_PATH = "/content/manifest.csv"
IMAGE_SIZE = (320, 320)
BATCH_SIZE = 16
CLASS_INDICES_PATH = "class_indices.json"


def _build_label_mapping(df: pd.DataFrame) -> dict:
    class_names = sorted(df["class_name"].unique())
    return {name: idx for idx, name in enumerate(class_names)}


def _load_and_preprocess(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label


def _make_dataset(df: pd.DataFrame, label_map: dict, shuffle: bool) -> tf.data.Dataset:
    filepaths = (DATASET_ROOT + os.sep + df["filepath"]).tolist()
    labels = df["class_name"].map(label_map).tolist()

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(manifest_path: str = MANIFEST_PATH):
    df = pd.read_csv(manifest_path)

    required_cols = {"filepath", "class_name", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv is missing required columns: {missing}")

    label_map = _build_label_mapping(df)
    class_names = sorted(label_map, key=label_map.get)

    with open(CLASS_INDICES_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label mapping to {CLASS_INDICES_PATH}: {label_map}")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_ds = _make_dataset(train_df, label_map, shuffle=True)
    val_ds = _make_dataset(val_df, label_map, shuffle=False)
    test_ds = _make_dataset(test_df, label_map, shuffle=False)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print("Classes (index order):", class_names)
    for images, labels in train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
        print("Sample labels:", labels.numpy()[:10])