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

    N_PERM   = 1000
    results  = {"color_left": [], "binding": []}

    for layer_i in range(n_layers):
        X = layer_feats[pair_indices, layer_i, :]   # (N_pairs, D)

        for task_name, y_raw in [("color_left", y_color_left), ("binding", y_binding)]:
            le  = LabelEncoder()
            y   = le.fit_transform(y_raw)
            clf = LogisticRegression(max_iter=1000)

            # Observed accuracy
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            observed = float(scores.mean())

            # Permutation null distribution
            null_accs = []
            rng = np.random.default_rng(42)
            for _ in range(N_PERM):
                y_shuf  = rng.permutation(y)
                s       = cross_val_score(clf, X, y_shuf, cv=5, scoring="accuracy")
                null_accs.append(s.mean())
            null_accs = np.array(null_accs)
            p_val = float(np.mean(null_accs >= observed))

            results[task_name].append({
                "layer":    layer_i,
                "accuracy": observed,
                "std":      float(scores.std()),
                "chance":   float(1 / len(le.classes_)),
                "p_value":  p_val,
                "null_mean": float(null_accs.mean()),
                "null_std":  float(null_accs.std()),
            })

        print(f"  Layer {layer_i:2d}: color_left acc = "
              f"{results['color_left'][-1]['accuracy']:.3f} "
              f"(p={results['color_left'][-1]['p_value']:.3f})  "
              f"binding acc = {results['binding'][-1]['accuracy']:.3f} "
              f"(p={results['binding'][-1]['p_value']:.3f})")

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
        accs       = [r["accuracy"]  for r in results[task_name]]
        null_means = [r["null_mean"] for r in results[task_name]]
        null_stds  = [r["null_std"]  for r in results[task_name]]
        p_vals     = [r["p_value"]   for r in results[task_name]]
        chance     = results[task_name][0]["chance"]
        layers     = list(range(n_layers))

        # Null band
        ax.fill_between(layers,
                        [m - 2*s for m, s in zip(null_means, null_stds)],
                        [m + 2*s for m, s in zip(null_means, null_stds)],
                        alpha=0.2, color="gray", label="Null ±2SD")

        # Observed — color by significance
        colors = ["steelblue" if p < 0.05 else "lightsteelblue" for p in p_vals]
        ax.scatter(layers, accs, c=colors, zorder=5, s=60)
        ax.plot(layers, accs, color="steelblue", linewidth=1.5, label="Observed accuracy")

        ax.axhline(chance, color="red", linestyle="--", label=f"Chance ({chance:.2f})")
        ax.set_xlabel("ViT Layer")
        ax.set_ylabel("5-fold CV Accuracy")
        ax.set_title(f"Layer Probe: {task_name.replace('_', ' ').title()}")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

    fig.suptitle("Where Does Binding Information Live in CLIP's Vision Encoder?", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "layer_probe_accuracy.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
