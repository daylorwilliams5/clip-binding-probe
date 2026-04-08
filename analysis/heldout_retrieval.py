"""
Held-out shape generalization test for the learned projection fix.

Generates new stimuli with novel shapes (pentagon, diamond) that
the projection never saw during training. Trains the projection
on the original 4 shapes, tests on foil pairs made from the new ones.

If retrieval accuracy stays high on held-out shapes, the fix generalizes
and isn't just memorizing shape-specific patterns.

Outputs:
  analysis/heldout_retrieval_results.json
  figures/heldout_retrieval.png
"""

import os
import sys
import csv
import json
from itertools import permutations, combinations

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.linear_model import Ridge
from scipy.spatial.distance import cosine
from transformers import CLIPProcessor, CLIPModel

STIMULI_DIR = os.path.join(os.path.dirname(__file__), "..", "stimuli", "images")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
VISION_PATH = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_vision.npz")
TEXT_PATH   = os.path.join(os.path.dirname(__file__), "..", "embeddings", "clip_text.npz")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
OUT_JSON    = os.path.join(os.path.dirname(__file__), "heldout_retrieval_results.json")
HELDOUT_DIR = os.path.join(os.path.dirname(__file__), "..", "stimuli", "heldout_images")

MODEL_ID       = "openai/clip-vit-base-patch32"
IMG_SIZE       = 224
SHAPE_SIZE     = 45
BACKGROUND     = (245, 245, 245)
TRAIN_SHAPES   = ["circle", "square", "triangle", "star"]
HELDOUT_SHAPES = ["pentagon", "diamond"]
COLORS = {
    "red":    (220, 50,  50),
    "blue":   (50,  100, 220),
    "green":  (50,  180, 70),
    "yellow": (230, 200, 40),
}


def draw_shape(draw, shape, cx, cy, r, color):
    if shape == "circle":
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif shape == "square":
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif shape == "triangle":
        pts = [(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)]
        draw.polygon(pts, fill=color)
    elif shape == "star":
        n, outer, inner = 5, r, r * 0.4
        pts = []
        for i in range(n * 2):
            angle = np.pi / n * i - np.pi / 2
            radius = outer if i % 2 == 0 else inner
            pts.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
        draw.polygon(pts, fill=color)
    elif shape == "pentagon":
        n = 5
        pts = []
        for i in range(n):
            angle = 2 * np.pi / n * i - np.pi / 2
            pts.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))
        draw.polygon(pts, fill=color)
    elif shape == "diamond":
        pts = [(cx, cy-r), (cx+r*0.6, cy), (cx, cy+r), (cx-r*0.6, cy)]
        draw.polygon(pts, fill=color)


def make_pair_image(shape_a, color_a, shape_b, color_b):
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape_a, IMG_SIZE // 4,     IMG_SIZE // 2, SHAPE_SIZE, color_a)
    draw_shape(draw, shape_b, 3 * IMG_SIZE // 4, IMG_SIZE // 2, SHAPE_SIZE, color_b)
    return img


def cosine_sim(a, b):
    return 1 - cosine(a, b)


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID} …")

    model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # ── Load training data (original shapes) ──────────────────────────────────
    vision_data = np.load(VISION_PATH, allow_pickle=True)
    text_data   = np.load(TEXT_PATH, allow_pickle=True)

    layer_feats  = vision_data["layer_features"]       # (N, n_layers, 768)
    vision_files = vision_data["filenames"]
    text_embs    = text_data["embeddings"]             # (N_texts, 512)
    text_descs   = text_data["descriptions"]

    img_idx  = {f: i for i, f in enumerate(vision_files)}
    text_idx = {t: i for i, t in enumerate(text_descs)}

    pairs = []
    with open(PAIRS_CSV) as f:
        pairs = list(csv.DictReader(f))

    # Build training set from original pairs
    train_img_indices, train_text_targets = [], []
    for p in pairs:
        for ik, tk in [("image_1", "text_1"), ("image_2", "text_2")]:
            train_img_indices.append(img_idx[p[ik]])
            train_text_targets.append(text_embs[text_idx[p[tk]]])

    train_img_indices  = np.array(train_img_indices)
    train_text_targets = np.stack(train_text_targets)

    # ── Generate held-out stimuli with new shapes ─────────────────────────────
    os.makedirs(HELDOUT_DIR, exist_ok=True)
    all_shapes = HELDOUT_SHAPES  # only new shapes paired with each other
    color_items = list(COLORS.items())

    heldout_pairs = []
    pair_id = 0
    for shape_a, shape_b in permutations(all_shapes, 2):
        for (c1_name, c1_rgb), (c2_name, c2_rgb) in combinations(color_items, 2):
            img1 = make_pair_image(shape_a, c1_rgb, shape_b, c2_rgb)
            fname1 = f"heldout_{pair_id:04d}_{shape_a}_{c1_name}_{shape_b}_{c2_name}.png"
            img1.save(os.path.join(HELDOUT_DIR, fname1))

            img2 = make_pair_image(shape_a, c2_rgb, shape_b, c1_rgb)
            fname2 = f"heldout_{pair_id:04d}_{shape_a}_{c2_name}_{shape_b}_{c1_name}.png"
            img2.save(os.path.join(HELDOUT_DIR, fname2))

            text1 = f"a {c1_name} {shape_a} and a {c2_name} {shape_b}"
            text2 = f"a {c2_name} {shape_a} and a {c1_name} {shape_b}"

            heldout_pairs.append({
                "pair_id": pair_id,
                "image_1": fname1, "text_1": text1,
                "image_2": fname2, "text_2": text2,
            })
            pair_id += 1

    print(f"Generated {len(heldout_pairs)} held-out foil pairs "
          f"({len(HELDOUT_SHAPES)} new shapes)")

    # ── Extract CLIP embeddings for held-out images ───────────────────────────
    heldout_vision = {}   # fname → (n_layers, 768)
    heldout_proj   = {}   # fname → (512,)
    heldout_text   = {}   # text  → (512,)

    with torch.no_grad():
        for hp in heldout_pairs:
            for ik, tk in [("image_1", "text_1"), ("image_2", "text_2")]:
                fname = hp[ik]
                if fname not in heldout_vision:
                    img = Image.open(os.path.join(HELDOUT_DIR, fname)).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    outputs = model.vision_model(**inputs, output_hidden_states=True)

                    layer_cls = np.stack([h[0, 0, :].cpu().numpy()
                                          for h in outputs.hidden_states])
                    heldout_vision[fname] = layer_cls

                    proj = model.visual_projection(
                        outputs.pooler_output).squeeze(0).cpu().numpy()
                    heldout_proj[fname] = proj

                t = hp[tk]
                if t not in heldout_text:
                    inputs = processor(text=[t], return_tensors="pt", padding=True).to(device)
                    text_out = model.text_model(**inputs)
                    projected = model.text_projection(text_out.pooler_output)
                    heldout_text[t] = projected.squeeze(0).cpu().numpy()

    print(f"Extracted embeddings for {len(heldout_vision)} held-out images")

    # ── Baseline: standard CLIP on held-out pairs ─────────────────────────────
    baseline_correct_img, baseline_correct_grp, total = 0, 0, 0
    for hp in heldout_pairs:
        v1  = heldout_proj[hp["image_1"]]
        v2  = heldout_proj[hp["image_2"]]
        tx1 = heldout_text[hp["text_1"]]
        tx2 = heldout_text[hp["text_2"]]

        img1_ok = cosine_sim(v1, tx1) > cosine_sim(v1, tx2)
        img2_ok = cosine_sim(v2, tx2) > cosine_sim(v2, tx1)
        baseline_correct_img += img1_ok + img2_ok
        baseline_correct_grp += (img1_ok and img2_ok)
        total += 1

    bl_img = baseline_correct_img / (total * 2)
    bl_grp = baseline_correct_grp / total
    print(f"\nBaseline (held-out): image={bl_img:.3f}  group={bl_grp:.3f}")

    # ── Per-layer: train on original shapes, test on held-out shapes ──────────
    n_layers = layer_feats.shape[1]
    results = {
        "baseline": {"image_acc": bl_img, "group_acc": bl_grp},
        "layers": [],
    }

    for layer_i in range(n_layers):
        # Train projection on original shapes
        X_train = layer_feats[train_img_indices, layer_i, :]
        ridge   = Ridge(alpha=1.0)
        ridge.fit(X_train, train_text_targets)

        # Test on held-out shapes
        correct_img, correct_grp, total = 0, 0, 0
        for hp in heldout_pairs:
            v1 = ridge.predict(heldout_vision[hp["image_1"]][layer_i:layer_i+1])
            v2 = ridge.predict(heldout_vision[hp["image_2"]][layer_i:layer_i+1])
            tx1 = heldout_text[hp["text_1"]]
            tx2 = heldout_text[hp["text_2"]]

            img1_ok = cosine_sim(v1[0], tx1) > cosine_sim(v1[0], tx2)
            img2_ok = cosine_sim(v2[0], tx2) > cosine_sim(v2[0], tx1)
            correct_img += img1_ok + img2_ok
            correct_grp += (img1_ok and img2_ok)
            total += 1

        img_acc = correct_img / (total * 2)
        grp_acc = correct_grp / total
        results["layers"].append({
            "layer": layer_i, "image_acc": img_acc, "group_acc": grp_acc,
        })

        sig = "***" if grp_acc > bl_grp else ""
        print(f"  Layer {layer_i:2d}: image={img_acc:.3f}  group={grp_acc:.3f}  {sig}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")

    _plot(results)


def _plot(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    layers   = [r["layer"]     for r in results["layers"]]
    grp_accs = [r["group_acc"] for r in results["layers"]]
    bl_grp   = results["baseline"]["group_acc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, grp_accs, marker="o", color="coral",
            label="Learned projection (trained on original shapes)")
    ax.axhline(bl_grp, color="red", linestyle="--",
               label=f"Baseline CLIP ({bl_grp:.3f})")
    ax.axhline(0.25, color="gray", linestyle=":", label="Chance")
    ax.set_xlabel("ViT Layer (source of CLS token)")
    ax.set_ylabel("Group-level Retrieval Accuracy")
    ax.set_title("Held-Out Shape Generalization:\n"
                 "Projection trained on original shapes, tested on novel shapes")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "heldout_retrieval.png")
    fig.savefig(out, dpi=150)
    print(f"Figure → {out}")


if __name__ == "__main__":
    run()
