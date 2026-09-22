"""
evaluate.py
-----------
Evaluates a trained model on the test set with per-class metrics,
not just aggregate accuracy. Use this on both the baseline model and
the class-weighted v2 model so you can compare them class-by-class.

Usage (Colab):
    !python evaluate.py                        # evaluates final_model.keras
    !python evaluate.py --model best_model.keras
    !python evaluate.py --model final_model_v2.keras --tag v2_class_weighted
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_pipeline import get_datasets


def main(model_path: str, tag: str):
    _, _, test_ds, class_names = get_datasets()

    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    report_path = f"classification_report_{tag}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    cm_path = f"confusion_matrix_{tag}.png"
    plt.savefig(cm_path)
    plt.show()
    print(f"Saved {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="final_model.keras", help="Path to the .keras model file"
    )
    parser.add_argument(
        "--tag", default="baseline", help="Label used in output filenames (e.g. baseline, v2_class_weighted)"
    )
    args = parser.parse_args()
    main(args.model, args.tag)
