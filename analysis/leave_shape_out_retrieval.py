"""
Leave-one-shape-out retrieval test.

For each of the 4 original shapes, hold out all foil pairs containing
that shape, train the Ridge projection on the remaining pairs, and
test retrieval on the held-out pairs.

This is a fairer generalization test than novel shapes — more data,
and tests whether the projection learns binding structure vs shape-specific
patterns.

Outputs:
  analysis/leave_shape_out_results.json
  figures/leave_shape_out_retrieval.png
"""

import os
import csv
import json

import numpy as np
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cosine

VISION_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_vision.npz")
TEXT_PATH   = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_text.npz")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_JSON    = os.path.join(os.path.dirname(__file__), "leave_shape_out_results.json")

SHAPES = ["circle", "square", "triangle", "star"]


def cosine_sim(a, b):
    return 1 - cosine(a, b)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pair_contains_shape(pair, shape):
    """Check if a foil pair involves a given shape."""
    return shape in pair["shape_a"] or shape in pair["shape_b"]


def run():
    vision_data = np.load(VISION_PATH, allow_pickle=True)
    text_data   = np.load(TEXT_PATH, allow_pickle=True)

    layer_feats  = vision_data["layer_features"]        # (N, n_layers, 768)
    proj_embs    = vision_data["projected_embeddings"]   # (N, 512)
    vision_files = vision_data["filenames"]
    text_embs    = text_data["embeddings"]
    text_descs   = text_data["descriptions"]

    img_idx  = {f: i for i, f in enumerate(vision_files)}
    text_idx = {t: i for i, t in enumerate(text_descs)}

    pairs    = load_csv(PAIRS_CSV)
    n_layers = layer_feats.shape[1]

    # Best layer from previous analysis
    BEST_LAYER = 12

    results = {"per_shape": [], "summary": {}}

    for held_out_shape in SHAPES:
        train_pairs = [p for p in pairs if not pair_contains_shape(p, held_out_shape)]
        test_pairs  = [p for p in pairs if pair_contains_shape(p, held_out_shape)]

        # Build training data
        train_indices, train_targets = [], []
        for p in train_pairs:
            for ik, tk in [("image_1", "text_1"), ("image_2", "text_2")]:
                train_indices.append(img_idx[p[ik]])
                train_targets.append(text_embs[text_idx[p[tk]]])
        train_indices = np.array(train_indices)
        train_targets = np.stack(train_targets)

        # Train projection
        X_train = layer_feats[train_indices, BEST_LAYER, :]
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, train_targets)

        # Baseline CLIP on test pairs
        bl_correct_img, bl_correct_grp = 0, 0
        # Learned projection on test pairs
        lp_correct_img, lp_correct_grp = 0, 0
        total = 0

        for p in test_pairs:
            i1, i2 = img_idx[p["image_1"]], img_idx[p["image_2"]]
            t1, t2 = text_idx[p["text_1"]], text_idx[p["text_2"]]
            tx1, tx2 = text_embs[t1], text_embs[t2]

            # Baseline
            bv1, bv2 = proj_embs[i1], proj_embs[i2]
            bl_img1 = cosine_sim(bv1, tx1) > cosine_sim(bv1, tx2)
            bl_img2 = cosine_sim(bv2, tx2) > cosine_sim(bv2, tx1)
            bl_correct_img += bl_img1 + bl_img2
            bl_correct_grp += (bl_img1 and bl_img2)

            # Learned projection
            lv1 = ridge.predict(layer_feats[i1:i1+1, BEST_LAYER, :])[0]
            lv2 = ridge.predict(layer_feats[i2:i2+1, BEST_LAYER, :])[0]
            lp_img1 = cosine_sim(lv1, tx1) > cosine_sim(lv1, tx2)
            lp_img2 = cosine_sim(lv2, tx2) > cosine_sim(lv2, tx1)
            lp_correct_img += lp_img1 + lp_img2
            lp_correct_grp += (lp_img1 and lp_img2)

            total += 1

        result = {
            "held_out":       held_out_shape,
            "n_test_pairs":   total,
            "n_train_pairs":  len(train_pairs),
            "baseline_image": bl_correct_img / (total * 2),
            "baseline_group": bl_correct_grp / total,
            "learned_image":  lp_correct_img / (total * 2),
            "learned_group":  lp_correct_grp / total,
        }
        results["per_shape"].append(result)

        print(f"Hold out {held_out_shape:10s} "
              f"(train={len(train_pairs)}, test={total}): "
              f"baseline group={result['baseline_group']:.3f}  "
              f"learned group={result['learned_group']:.3f}")

    # Summary
    bl_mean = np.mean([r["baseline_group"] for r in results["per_shape"]])
    lp_mean = np.mean([r["learned_group"]  for r in results["per_shape"]])
    results["summary"] = {
        "mean_baseline_group": float(bl_mean),
        "mean_learned_group":  float(lp_mean),
        "layer_used": BEST_LAYER,
    }
    print(f"\nMean baseline group: {bl_mean:.3f}")
    print(f"Mean learned group:  {lp_mean:.3f}")

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

    shapes   = [r["held_out"]       for r in results["per_shape"]]
    bl_grps  = [r["baseline_group"] for r in results["per_shape"]]
    lp_grps  = [r["learned_group"]  for r in results["per_shape"]]

    x = np.arange(len(shapes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, bl_grps, w, label="Baseline CLIP", color="salmon")
    ax.bar(x + w/2, lp_grps, w, label="Learned projection", color="steelblue")
    ax.axhline(0.25, color="gray", linestyle=":", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Hold out\n{s}" for s in shapes])
    ax.set_ylabel("Group-level Retrieval Accuracy")
    ax.set_title("Leave-One-Shape-Out Generalization:\n"
                 "Train projection on 3 shapes, test on held-out shape")
    ax.set_ylim(0, 1)
    ax.legend()

    bl_mean = results["summary"]["mean_baseline_group"]
    lp_mean = results["summary"]["mean_learned_group"]
    ax.text(0.98, 0.95,
            f"Mean baseline: {bl_mean:.3f}\nMean learned: {lp_mean:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "leave_shape_out_retrieval.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
