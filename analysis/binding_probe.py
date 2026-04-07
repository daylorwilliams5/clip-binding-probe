"""
Linear probe on CLIP vision embeddings for attribute binding.

Task: given a pair image embedding, can a linear classifier decode
which color-shape binding is present?

This is distinct from the retrieval test — it asks whether binding
information exists in the vision encoder at all, independent of
text alignment.

Outputs:
  analysis/binding_probe_results.json
  figures/binding_probe_accuracy.png
"""

import os
import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder

VISION_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_vision.npz")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_JSON    = os.path.join(os.path.dirname(__file__), "binding_probe_results.json")


def run():
    vision_data  = np.load(VISION_PATH, allow_pickle=True)
    vision_embs  = vision_data["embeddings"]
    vision_files = vision_data["filenames"]
    img_idx      = {f: i for i, f in enumerate(vision_files)}

    with open(PAIRS_CSV) as f:
        pairs = list(csv.DictReader(f))

    # Build dataset: X = vision embedding of pair image, y = binding label
    # binding label: "{color_left}_{shape_left}__{color_right}_{shape_right}"
    X, y_binding, y_color_left, y_color_right = [], [], [], []

    for p in pairs:
        for img_key, col_l, col_r in [
            ("image_1", p["color_1"], p["color_2"]),
            ("image_2", p["color_2"], p["color_1"]),
        ]:
            emb = vision_embs[img_idx[p[img_key]]]
            X.append(emb)
            y_binding.append(f"{col_l}__{col_r}")
            y_color_left.append(col_l)
            y_color_right.append(col_r)

    X = np.array(X)

    results = {}
    for task_name, y_raw in [
        ("binding",     y_binding),
        ("color_left",  y_color_left),
        ("color_right", y_color_right),
    ]:
        le = LabelEncoder()
        y  = le.fit_transform(y_raw)
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        results[task_name] = {
            "accuracy": float(scores.mean()),
            "std":      float(scores.std()),
            "n_classes": int(len(le.classes_)),
            "chance":   float(1 / len(le.classes_)),
        }
        print(f"{task_name:15s}: acc = {scores.mean():.3f} ± {scores.std():.3f}  "
              f"(chance = {1/len(le.classes_):.3f})")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {OUT_JSON}")

    _plot(results)


def _plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    tasks   = list(results.keys())
    accs    = [results[t]["accuracy"] for t in tasks]
    stds    = [results[t]["std"]      for t in tasks]
    chances = [results[t]["chance"]   for t in tasks]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tasks))
    ax.bar(x, accs, yerr=stds, capsize=5, color="steelblue", alpha=0.8, label="Accuracy")
    ax.scatter(x, chances, color="red", zorder=5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1)
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_title("Linear Probe: Binding Information in CLIP Vision Embeddings")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "binding_probe_accuracy.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
