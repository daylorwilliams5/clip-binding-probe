"""
Layer-wise binding probe: at which ViT layer does binding information
first appear (or disappear) in CLIP's vision encoder?

For each layer, train a linear probe to decode color_left from the
CLS token at that layer. Plot accuracy vs layer depth.

This is the novel contribution — existing work shows CLIP fails
behaviorally; this shows *where* in the network the failure occurs.

Outputs:
  analysis/layer_probe_results.json
  figures/layer_probe_accuracy.png
"""

import os
import csv
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

VISION_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_vision.npz")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_JSON    = os.path.join(os.path.dirname(__file__), "layer_probe_results.json")


def run():
    vision_data  = np.load(VISION_PATH, allow_pickle=True)
    layer_feats  = vision_data["layer_features"]   # (N, n_layers, D)
    vision_files = vision_data["filenames"]
    img_idx      = {f: i for i, f in enumerate(vision_files)}

    with open(PAIRS_CSV) as f:
        pairs = list(csv.DictReader(f))

    # Build labels for pair images
    pair_indices, y_color_left, y_binding = [], [], []
    for p in pairs:
        for img_key, col_l, col_r in [
            ("image_1", p["color_1"], p["color_2"]),
            ("image_2", p["color_2"], p["color_1"]),
        ]:
            pair_indices.append(img_idx[p[img_key]])
            y_color_left.append(col_l)
            y_binding.append(f"{col_l}__{col_r}")

    pair_indices = np.array(pair_indices)
    n_layers     = layer_feats.shape[1]

    results = {"color_left": [], "binding": []}

    for layer_i in range(n_layers):
        X = layer_feats[pair_indices, layer_i, :]   # (N_pairs, D)

        for task_name, y_raw in [("color_left", y_color_left), ("binding", y_binding)]:
            le  = LabelEncoder()
            y   = le.fit_transform(y_raw)
            clf = LogisticRegression(max_iter=1000)
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            results[task_name].append({
                "layer":    layer_i,
                "accuracy": float(scores.mean()),
                "std":      float(scores.std()),
                "chance":   float(1 / len(le.classes_)),
            })

        print(f"  Layer {layer_i:2d}: color_left acc = "
              f"{results['color_left'][-1]['accuracy']:.3f}  "
              f"binding acc = {results['binding'][-1]['accuracy']:.3f}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {OUT_JSON}")

    _plot(results, n_layers)


def _plot(results, n_layers):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, task_name in zip(axes, ["color_left", "binding"]):
        accs    = [r["accuracy"] for r in results[task_name]]
        stds    = [r["std"]      for r in results[task_name]]
        chance  = results[task_name][0]["chance"]
        layers  = list(range(n_layers))

        ax.plot(layers, accs, marker="o", color="steelblue", label="Accuracy")
        ax.fill_between(layers,
                        [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)],
                        alpha=0.2, color="steelblue")
        ax.axhline(chance, color="red", linestyle="--", label=f"Chance ({chance:.2f})")
        ax.set_xlabel("ViT Layer")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.set_title(f"Layer Probe: {task_name.replace('_', ' ').title()}")
        ax.legend()
        ax.set_ylim(0, 1)

    fig.suptitle("Where Does Binding Information Live in CLIP's Vision Encoder?", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "layer_probe_accuracy.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
