"""
Visualize CLIP's attention maps across layers for binding stimuli.

For a foil pair (e.g. "red circle + blue square" vs "blue circle + red square"),
extract attention maps from each ViT layer and visualize where the model is
looking. The question: do middle layers (where binding info peaks) attend
differently than final layers (where binding is erased)?

Outputs:
  figures/attention_maps/    — per-layer attention heatmaps for sample pairs
  figures/attention_summary.png — average attention entropy across layers
"""

import os
import csv

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

STIMULI_DIR = os.path.join(os.path.dirname(__file__), "..", "stimuli", "images")
PAIRS_CSV   = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
ATTN_DIR    = os.path.join(os.path.dirname(__file__), "..", "figures", "attention_maps")
MODEL_ID    = "openai/clip-vit-base-patch32"

# ViT-B/32: 224px image, 32px patches → 7×7 = 49 patch tokens + 1 CLS = 50 total
PATCH_GRID = 7


def get_attention_maps(model, processor, img_path, device):
    """Return attention maps for all layers and heads.

    Returns: (n_layers, n_heads, seq_len, seq_len) numpy array
    """
    img    = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.vision_model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
    attn = np.stack([a.squeeze(0).cpu().numpy() for a in outputs.attentions])
    return attn, img


def cls_to_patch_attention(attn_layer_head):
    """Extract CLS→patch attention and reshape to 2D grid.

    attn_layer_head: (seq_len, seq_len) — full attention matrix for one head
    Returns: (PATCH_GRID, PATCH_GRID) — CLS attention over spatial patches
    """
    # Row 0 = CLS token attending to all tokens; skip column 0 (CLS→CLS)
    cls_attn = attn_layer_head[0, 1:]  # (49,)
    return cls_attn.reshape(PATCH_GRID, PATCH_GRID)


def plot_pair_attention(model, processor, pair, device, pair_idx):
    """Plot attention maps for one foil pair across all layers."""
    os.makedirs(ATTN_DIR, exist_ok=True)

    img1_path = os.path.join(STIMULI_DIR, pair["image_1"])
    img2_path = os.path.join(STIMULI_DIR, pair["image_2"])

    attn1, img1 = get_attention_maps(model, processor, img1_path, device)
    attn2, img2 = get_attention_maps(model, processor, img2_path, device)

    n_layers, n_heads = attn1.shape[0], attn1.shape[1]

    # Average over heads for cleaner visualization
    for layer_i in range(n_layers):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Original image 1
        axes[0].imshow(img1.resize((224, 224)))
        axes[0].set_title(pair["text_1"] if "text_1" in pair else "Image 1", fontsize=9)
        axes[0].axis("off")

        # Attention map image 1 (averaged over heads)
        avg_attn1 = np.mean([cls_to_patch_attention(attn1[layer_i, h])
                             for h in range(n_heads)], axis=0)
        axes[1].imshow(img1.resize((224, 224)))
        axes[1].imshow(avg_attn1, cmap="hot", alpha=0.6,
                       extent=[0, 224, 224, 0], interpolation="bilinear")
        axes[1].set_title(f"Layer {layer_i} attention", fontsize=9)
        axes[1].axis("off")

        # Attention map image 2 (averaged over heads)
        avg_attn2 = np.mean([cls_to_patch_attention(attn2[layer_i, h])
                             for h in range(n_heads)], axis=0)
        axes[2].imshow(img2.resize((224, 224)))
        axes[2].imshow(avg_attn2, cmap="hot", alpha=0.6,
                       extent=[0, 224, 224, 0], interpolation="bilinear")
        axes[2].set_title(f"Layer {layer_i} attention (foil)", fontsize=9)
        axes[2].axis("off")

        fig.suptitle(f"Pair {pair_idx} — Layer {layer_i}", fontsize=11)
        fig.tight_layout()
        out = os.path.join(ATTN_DIR, f"pair{pair_idx:02d}_layer{layer_i:02d}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Pair {pair_idx}: saved {n_layers} layer maps")


def plot_summary(model, processor, pairs, device):
    """Plot how attention patterns change across layers.

    For all pair images, compute:
    1. CLS attention entropy (how spread out is attention?) per layer
    2. Left-vs-right attention ratio (does CLIP attend more to one object?)
    """
    with open(PAIRS_CSV) as f:
        all_pairs = list(csv.DictReader(f))

    n_sample = min(20, len(all_pairs))
    sample   = all_pairs[:n_sample]

    entropy_per_layer    = []
    left_right_per_layer = []

    for p in sample:
        img_path = os.path.join(STIMULI_DIR, p["image_1"])
        attn, _  = get_attention_maps(model, processor, img_path, device)
        n_layers, n_heads = attn.shape[0], attn.shape[1]

        for layer_i in range(n_layers):
            avg_attn = np.mean([cls_to_patch_attention(attn[layer_i, h])
                                for h in range(n_heads)], axis=0)  # (7, 7)

            # Entropy of flattened attention
            flat = avg_attn.flatten()
            flat = flat / flat.sum()
            ent  = -np.sum(flat * np.log(flat + 1e-10))

            # Left vs right ratio
            left  = avg_attn[:, :PATCH_GRID // 2].sum()
            right = avg_attn[:, PATCH_GRID // 2 + 1:].sum()
            ratio = left / (left + right + 1e-10)

            while len(entropy_per_layer) <= layer_i:
                entropy_per_layer.append([])
                left_right_per_layer.append([])
            entropy_per_layer[layer_i].append(ent)
            left_right_per_layer[layer_i].append(ratio)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    layers = list(range(len(entropy_per_layer)))
    ent_means = [np.mean(e) for e in entropy_per_layer]
    ent_stds  = [np.std(e)  for e in entropy_per_layer]
    axes[0].plot(layers, ent_means, marker="o", color="steelblue")
    axes[0].fill_between(layers,
                         [m - s for m, s in zip(ent_means, ent_stds)],
                         [m + s for m, s in zip(ent_means, ent_stds)],
                         alpha=0.2, color="steelblue")
    axes[0].set_xlabel("ViT Layer")
    axes[0].set_ylabel("Attention Entropy")
    axes[0].set_title("CLS Attention Entropy by Layer")

    lr_means = [np.mean(lr) for lr in left_right_per_layer]
    lr_stds  = [np.std(lr)  for lr in left_right_per_layer]
    axes[1].plot(layers, lr_means, marker="o", color="coral")
    axes[1].fill_between(layers,
                         [m - s for m, s in zip(lr_means, lr_stds)],
                         [m + s for m, s in zip(lr_means, lr_stds)],
                         alpha=0.2, color="coral")
    axes[1].axhline(0.5, color="gray", linestyle="--", label="Equal L/R")
    axes[1].set_xlabel("ViT Layer")
    axes[1].set_ylabel("Left / (Left + Right)")
    axes[1].set_title("CLS Attention: Left vs Right Object")
    axes[1].legend()

    fig.suptitle("How CLIP's Attention Evolves Across Layers", fontsize=13)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "attention_summary.png")
    fig.savefig(out, dpi=150)
    print(f"Summary → {out}")


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID} …")

    model     = CLIPModel.from_pretrained(MODEL_ID, attn_implementation="eager").to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    with open(PAIRS_CSV) as f:
        pairs = list(csv.DictReader(f))

    # Plot detailed attention maps for first 3 pairs
    print("Generating per-layer attention maps for sample pairs…")
    for i in range(min(3, len(pairs))):
        plot_pair_attention(model, processor, pairs[i], device, i)

    # Plot summary across all pairs
    print("Generating attention summary…")
    plot_summary(model, processor, pairs, device)


if __name__ == "__main__":
    run()
