"""
Behavioral retrieval test: does CLIP rank the correct text above the foil?

For each foil pair (image_1, text_1) vs (image_2, text_2):
  - Compute cosine similarity: sim(image_1, text_1) vs sim(image_1, text_2)
  - CLIP "succeeds" if it ranks the correct text higher

This replicates the Winoground-style evaluation on our controlled stimuli.

Outputs:
  analysis/retrieval_results.json
  figures/retrieval_accuracy.png
"""

import os
import csv
import json

import numpy as np
from scipy.spatial.distance import cosine

VISION_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_vision.npz")
TEXT_PATH   = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_text.npz")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_JSON    = os.path.join(os.path.dirname(__file__), "retrieval_results.json")


def cosine_sim(a, b):
    return 1 - cosine(a, b)


def run():
    vision_data = np.load(VISION_PATH, allow_pickle=True)
    text_data   = np.load(TEXT_PATH,   allow_pickle=True)

    vision_embs  = vision_data["projected_embeddings"]  # (N_images, 512)
    vision_files = vision_data["filenames"]
    text_embs    = text_data["embeddings"]            # (N_texts, D)
    text_descs   = text_data["descriptions"]

    # Build lookup dicts
    img_idx  = {f: i for i, f in enumerate(vision_files)}
    text_idx = {t: i for i, t in enumerate(text_descs)}

    with open(PAIRS_CSV) as f:
        pairs = list(csv.DictReader(f))

    correct = 0
    results = []

    for p in pairs:
        i1 = img_idx[p["image_1"]]
        i2 = img_idx[p["image_2"]]
        t1 = text_idx[p["text_1"]]
        t2 = text_idx[p["text_2"]]

        v1, v2 = vision_embs[i1], vision_embs[i2]
        tx1, tx2 = text_embs[t1], text_embs[t2]

        # Image 1 should match text 1 more than text 2
        sim_correct = cosine_sim(v1, tx1)
        sim_foil    = cosine_sim(v1, tx2)
        img1_correct = sim_correct > sim_foil

        # Image 2 should match text 2 more than text 1
        sim_correct2 = cosine_sim(v2, tx2)
        sim_foil2    = cosine_sim(v2, tx1)
        img2_correct = sim_correct2 > sim_foil2

        # Both must be correct (group score, stricter)
        both_correct = img1_correct and img2_correct
        if both_correct:
            correct += 1

        results.append({
            "pair_id":      p["pair_id"],
            "shape_a":      p["shape_a"],
            "shape_b":      p["shape_b"],
            "color_1":      p["color_1"],
            "color_2":      p["color_2"],
            "img1_correct": bool(img1_correct),
            "img2_correct": bool(img2_correct),
            "both_correct": bool(both_correct),
        })

    group_acc = correct / len(pairs)
    img_acc   = np.mean([r["img1_correct"] for r in results] +
                        [r["img2_correct"] for r in results])

    print(f"Retrieval accuracy (image-level): {img_acc:.3f}")
    print(f"Retrieval accuracy (group-level): {group_acc:.3f}")
    print(f"Chance = 0.50 (image), 0.25 (group)")

    output = {
        "image_level_accuracy": float(img_acc),
        "group_level_accuracy": float(group_acc),
        "n_pairs": len(pairs),
        "per_pair": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved → {OUT_JSON}")

    _plot(results)


def _plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Accuracy broken down by shape pair
    from collections import defaultdict
    shape_acc = defaultdict(list)
    for r in results:
        key = f"{r['shape_a']}-{r['shape_b']}"
        shape_acc[key].append(r["both_correct"])

    keys   = sorted(shape_acc)
    accs   = [np.mean(shape_acc[k]) for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(keys, accs, color=["steelblue" if a > 0.5 else "salmon" for a in accs])
    ax.axhline(0.25, color="gray", linestyle="--", label="Chance (group)")
    ax.axhline(0.50, color="gray", linestyle=":",  label="Chance (image)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Group retrieval accuracy")
    ax.set_title("CLIP Binding Retrieval Accuracy by Shape Pair")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "retrieval_accuracy.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
