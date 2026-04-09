"""
Nonlinear projection for binding retrieval — leave-one-shape-out.

Same setup as leave_shape_out_retrieval.py, but replaces Ridge with
a small MLP (768 → 256 → 512) to test whether a nonlinear mapping
can disentangle binding from object identity.

Outputs:
  analysis/nonlinear_retrieval_results.json
  figures/nonlinear_retrieval.png
"""

import os
import csv
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import cosine

VISION_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_vision.npz")
TEXT_PATH   = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_text.npz")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_JSON    = os.path.join(os.path.dirname(__file__), "nonlinear_retrieval_results.json")

SHAPES     = ["circle", "square", "triangle", "star"]
BEST_LAYER = 12
EPOCHS     = 200
LR         = 1e-3
HIDDEN     = 256


class BindingMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=HIDDEN, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def cosine_sim(a, b):
    return 1 - cosine(a, b)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pair_contains_shape(pair, shape):
    return shape in pair["shape_a"] or shape in pair["shape_b"]


def train_mlp(X_train, Y_train):
    """Train a small MLP to map vision features to text space."""
    device = torch.device("cpu")
    model  = BindingMLP(in_dim=X_train.shape[1], out_dim=Y_train.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CosineEmbeddingLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    Y_t = torch.tensor(Y_train, dtype=torch.float32)
    target = torch.ones(X_t.shape[0])  # all pairs are positive matches

    dataset = TensorDataset(X_t, Y_t, target)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for xb, yb, tb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb, tb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    model.eval()
    return model


def run():
    vision_data = np.load(VISION_PATH, allow_pickle=True)
    text_data   = np.load(TEXT_PATH, allow_pickle=True)

    layer_feats  = vision_data["layer_features"]
    proj_embs    = vision_data["projected_embeddings"]
    vision_files = vision_data["filenames"]
    text_embs    = text_data["embeddings"]
    text_descs   = text_data["descriptions"]

    img_idx  = {f: i for i, f in enumerate(vision_files)}
    text_idx = {t: i for i, t in enumerate(text_descs)}

    pairs = load_csv(PAIRS_CSV)

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

        X_train = layer_feats[train_indices, BEST_LAYER, :]

        # Train MLP
        model = train_mlp(X_train, train_targets)

        # Evaluate on test pairs
        bl_correct_grp, lp_correct_grp = 0, 0
        total = 0

        with torch.no_grad():
            for p in test_pairs:
                i1, i2 = img_idx[p["image_1"]], img_idx[p["image_2"]]
                t1, t2 = text_idx[p["text_1"]], text_idx[p["text_2"]]
                tx1, tx2 = text_embs[t1], text_embs[t2]

                # Baseline
                bv1, bv2 = proj_embs[i1], proj_embs[i2]
                bl_img1 = cosine_sim(bv1, tx1) > cosine_sim(bv1, tx2)
                bl_img2 = cosine_sim(bv2, tx2) > cosine_sim(bv2, tx1)
                bl_correct_grp += (bl_img1 and bl_img2)

                # MLP projection
                lv1 = model(torch.tensor(layer_feats[i1, BEST_LAYER, :],
                            dtype=torch.float32).unsqueeze(0)).numpy()[0]
                lv2 = model(torch.tensor(layer_feats[i2, BEST_LAYER, :],
                            dtype=torch.float32).unsqueeze(0)).numpy()[0]
                lp_img1 = cosine_sim(lv1, tx1) > cosine_sim(lv1, tx2)
                lp_img2 = cosine_sim(lv2, tx2) > cosine_sim(lv2, tx1)
                lp_correct_grp += (lp_img1 and lp_img2)

                total += 1

        result = {
            "held_out":       held_out_shape,
            "n_test_pairs":   total,
            "baseline_group": bl_correct_grp / total,
            "mlp_group":      lp_correct_grp / total,
        }
        results["per_shape"].append(result)

        print(f"Hold out {held_out_shape:10s}: "
              f"baseline={result['baseline_group']:.3f}  "
              f"linear={0:.3f}  "  # placeholder
              f"MLP={result['mlp_group']:.3f}")

    bl_mean  = np.mean([r["baseline_group"] for r in results["per_shape"]])
    mlp_mean = np.mean([r["mlp_group"]      for r in results["per_shape"]])
    results["summary"] = {
        "mean_baseline_group": float(bl_mean),
        "mean_mlp_group":      float(mlp_mean),
    }
    print(f"\nMean baseline: {bl_mean:.3f}")
    print(f"Mean MLP:      {mlp_mean:.3f}")

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
    mlp_grps = [r["mlp_group"]      for r in results["per_shape"]]

    x = np.arange(len(shapes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, bl_grps, w, label="Baseline CLIP", color="salmon")
    ax.bar(x + w/2, mlp_grps, w, label="MLP projection", color="steelblue")
    ax.axhline(0.25, color="gray", linestyle=":", label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Hold out\n{s}" for s in shapes])
    ax.set_ylabel("Group-level Retrieval Accuracy")
    ax.set_title("Leave-One-Shape-Out: MLP Projection\n"
                 "Train on 3 shapes, test on held-out shape")
    ax.set_ylim(0, 1)
    ax.legend()

    bl_mean  = results["summary"]["mean_baseline_group"]
    mlp_mean = results["summary"]["mean_mlp_group"]
    ax.text(0.98, 0.95,
            f"Mean baseline: {bl_mean:.3f}\nMean MLP: {mlp_mean:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "nonlinear_retrieval.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
