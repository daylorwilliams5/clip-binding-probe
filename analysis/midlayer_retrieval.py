"""
Mid-layer retrieval: can we fix CLIP's binding failure by reading from
earlier layers instead of the final one?

The layer probe showed binding info peaks at layers 3-5. This script
tests whether using those mid-layer representations for image-text
retrieval improves binding accuracy over the standard final-layer approach.

For each ViT layer:
  1. Take the CLS token from that layer as the image representation
  2. Project it to the text embedding space (learned linear projection)
  3. Run the same foil retrieval test

If mid-layer retrieval beats final-layer retrieval, that's an actionable
finding: CLIP already has binding info, you just need to read it from
the right layer.

Outputs:
  analysis/midlayer_retrieval_results.json
  figures/midlayer_retrieval.png
"""

import os
import csv
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cosine

VISION_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_vision.npz")
TEXT_PATH   = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_text.npz")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
METADATA_CSV = os.path.join(os.path.dirname(__file__), "..", "stimuli", "metadata.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_JSON    = os.path.join(os.path.dirname(__file__), "midlayer_retrieval_results.json")


def cosine_sim(a, b):
    return 1 - cosine(a, b)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def run():
    vision_data = np.load(VISION_PATH, allow_pickle=True)
    text_data   = np.load(TEXT_PATH, allow_pickle=True)

    layer_feats     = vision_data["layer_features"]       # (N_images, n_layers, 768)
    proj_embs       = vision_data["projected_embeddings"] # (N_images, 512)
    vision_files    = vision_data["filenames"]
    text_embs       = text_data["embeddings"]             # (N_texts, 512)
    text_descs      = text_data["descriptions"]

    img_idx  = {f: i for i, f in enumerate(vision_files)}
    text_idx = {t: i for i, t in enumerate(text_descs)}

    pairs    = load_csv(PAIRS_CSV)
    metadata = load_csv(METADATA_CSV)
    n_layers = layer_feats.shape[1]

    # ── Baseline: final projected embeddings (standard CLIP) ──────────────────
    baseline = eval_retrieval(proj_embs, text_embs, pairs, img_idx, text_idx)
    print(f"Baseline (final layer, projected): "
          f"image={baseline['image_acc']:.3f}  group={baseline['group_acc']:.3f}")

    # ── Per-layer retrieval with learned projection ───────────────────────────
    # Strategy: learn a Ridge regression from layer CLS → text embedding space
    # using all pair images and their correct text embeddings as supervision.
    # Use cross-validated predictions to avoid train/test leakage.

    # Build training data: pair image indices → their correct text embeddings
    pair_img_indices = []
    pair_text_targets = []
    for p in pairs:
        for img_key, text_key in [("image_1", "text_1"), ("image_2", "text_2")]:
            pair_img_indices.append(img_idx[p[img_key]])
            pair_text_targets.append(text_embs[text_idx[p[text_key]]])

    pair_img_indices  = np.array(pair_img_indices)
    pair_text_targets = np.stack(pair_text_targets)  # (N_pair_images, 512)

    results = {"baseline": baseline, "layers": []}

    for layer_i in range(n_layers):
        X = layer_feats[pair_img_indices, layer_i, :]  # (N_pair_images, 768)

        # Learn projection: 768 → 512 with cross-validation
        ridge = Ridge(alpha=1.0)
        # Cross-val predict gives out-of-fold predictions for each sample
        projected = cross_val_predict(ridge, X, pair_text_targets, cv=5)

        # Now evaluate retrieval using these projected embeddings
        # Build a lookup from filename to projected embedding
        proj_lookup = {}
        for idx_in_list, img_global_idx in enumerate(pair_img_indices):
            fname = vision_files[img_global_idx]
            proj_lookup[fname] = projected[idx_in_list]

        layer_result = eval_retrieval_from_lookup(
            proj_lookup, text_embs, pairs, text_idx)

        results["layers"].append({
            "layer":     layer_i,
            "image_acc": layer_result["image_acc"],
            "group_acc": layer_result["group_acc"],
        })

        sig = "***" if layer_result["group_acc"] > baseline["group_acc"] else ""
        print(f"  Layer {layer_i:2d}: image={layer_result['image_acc']:.3f}  "
              f"group={layer_result['group_acc']:.3f}  {sig}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")

    _plot(results)


def eval_retrieval(vision_embs, text_embs, pairs, img_idx, text_idx):
    """Standard retrieval evaluation using precomputed embeddings."""
    correct_img, correct_group, total = 0, 0, 0

    for p in pairs:
        i1, i2 = img_idx[p["image_1"]], img_idx[p["image_2"]]
        t1, t2 = text_idx[p["text_1"]], text_idx[p["text_2"]]

        v1, v2   = vision_embs[i1], vision_embs[i2]
        tx1, tx2 = text_embs[t1], text_embs[t2]

        img1_ok = cosine_sim(v1, tx1) > cosine_sim(v1, tx2)
        img2_ok = cosine_sim(v2, tx2) > cosine_sim(v2, tx1)

        correct_img += img1_ok + img2_ok
        correct_group += (img1_ok and img2_ok)
        total += 1

    return {
        "image_acc": correct_img / (total * 2),
        "group_acc": correct_group / total,
    }


def eval_retrieval_from_lookup(proj_lookup, text_embs, pairs, text_idx):
    """Retrieval evaluation using a dict of projected embeddings."""
    correct_img, correct_group, total = 0, 0, 0

    for p in pairs:
        if p["image_1"] not in proj_lookup or p["image_2"] not in proj_lookup:
            continue

        v1 = proj_lookup[p["image_1"]]
        v2 = proj_lookup[p["image_2"]]
        tx1 = text_embs[text_idx[p["text_1"]]]
        tx2 = text_embs[text_idx[p["text_2"]]]

        img1_ok = cosine_sim(v1, tx1) > cosine_sim(v1, tx2)
        img2_ok = cosine_sim(v2, tx2) > cosine_sim(v2, tx1)

        correct_img += img1_ok + img2_ok
        correct_group += (img1_ok and img2_ok)
        total += 1

    return {
        "image_acc": correct_img / (total * 2),
        "group_acc": correct_group / total,
    }


def _plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    layers    = [r["layer"]     for r in results["layers"]]
    img_accs  = [r["image_acc"] for r in results["layers"]]
    grp_accs  = [r["group_acc"] for r in results["layers"]]
    bl_img    = results["baseline"]["image_acc"]
    bl_grp    = results["baseline"]["group_acc"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image-level
    axes[0].plot(layers, img_accs, marker="o", color="steelblue",
                 label="Mid-layer + projection")
    axes[0].axhline(bl_img, color="red", linestyle="--",
                    label=f"Baseline final layer ({bl_img:.3f})")
    axes[0].axhline(0.5, color="gray", linestyle=":", label="Chance")
    axes[0].set_xlabel("ViT Layer (source of CLS token)")
    axes[0].set_ylabel("Image-level Retrieval Accuracy")
    axes[0].set_title("Image-level Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    # Group-level
    axes[1].plot(layers, grp_accs, marker="o", color="coral",
                 label="Mid-layer + projection")
    axes[1].axhline(bl_grp, color="red", linestyle="--",
                    label=f"Baseline final layer ({bl_grp:.3f})")
    axes[1].axhline(0.25, color="gray", linestyle=":", label="Chance")
    axes[1].set_xlabel("ViT Layer (source of CLS token)")
    axes[1].set_ylabel("Group-level Retrieval Accuracy")
    axes[1].set_title("Group-level Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1)

    fig.suptitle("Can Mid-Layer Representations Fix CLIP's Binding Failure?",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "midlayer_retrieval.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
