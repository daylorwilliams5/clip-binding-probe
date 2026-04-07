"""
Extract CLIP vision and text embeddings for all stimuli.

Uses CLIP ViT-B/32 via HuggingFace transformers.
Also extracts intermediate ViT layer features for the layer-wise binding probe.

Outputs:
  embeddings/clip_vision.npz  — CLS embeddings + per-layer features for all images
  embeddings/clip_text.npz    — text embeddings for all unique pair descriptions
"""

import os
import csv

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

STIMULI_DIR  = os.path.join(os.path.dirname(__file__), "..", "stimuli", "images")
METADATA_CSV = os.path.join(os.path.dirname(__file__), "..", "stimuli", "metadata.csv")
PAIRS_CSV    = os.path.join(os.path.dirname(__file__), "..", "stimuli", "pairs.csv")
VISION_OUT   = os.path.join(os.path.dirname(__file__), "clip_vision.npz")
TEXT_OUT     = os.path.join(os.path.dirname(__file__), "clip_text.npz")

MODEL_ID = "openai/clip-vit-base-patch32"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def extract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID} …")

    model     = CLIPModel.from_pretrained(MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    records = load_csv(METADATA_CSV)

    # ── Vision embeddings ─────────────────────────────────────────────────────
    vision_embeddings    = []
    projected_embeddings = []
    layer_features       = []
    filenames            = []

    with torch.no_grad():
        for i, rec in enumerate(records):
            img_path = os.path.join(STIMULI_DIR, rec["filename"])
            img      = Image.open(img_path).convert("RGB")
            inputs   = processor(images=img, return_tensors="pt").to(device)

            outputs = model.vision_model(**inputs, output_hidden_states=True)

            # Raw CLS for layer probe (768-dim)
            raw_cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            vision_embeddings.append(raw_cls)

            # Projected embedding for retrieval
            proj = model.visual_projection(outputs.pooler_output).squeeze(0).cpu().numpy()
            projected_embeddings.append(proj)

            # CLS from each layer: tuple of (1, seq_len, D)
            layer_cls = np.stack([h[0, 0, :].cpu().numpy() for h in outputs.hidden_states])
            layer_features.append(layer_cls)  # (n_layers, D)

            filenames.append(rec["filename"])
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(records)}")

    vision_embeddings    = np.stack(vision_embeddings)     # (N, 768)
    projected_embeddings = np.stack(projected_embeddings)  # (N, 512)
    layer_features       = np.stack(layer_features)        # (N, n_layers, 768)

    np.savez(VISION_OUT,
             embeddings=vision_embeddings,
             projected_embeddings=projected_embeddings,
             layer_features=layer_features,
             filenames=np.array(filenames))
    print(f"\nvisual_projection: {model.visual_projection}")
    print(f"Vision embeddings    {vision_embeddings.shape} → {VISION_OUT}")
    print(f"Projected embeddings {projected_embeddings.shape}")
    print(f"Layer features       {layer_features.shape}")

    # ── Text embeddings ───────────────────────────────────────────────────────
    pairs = load_csv(PAIRS_CSV)
    seen, text_embeddings, descriptions = {}, [], []

    with torch.no_grad():
        for p in pairs:
            for key in ("text_1", "text_2"):
                t = p[key]
                if t not in seen:
                    inputs = processor(text=[t], return_tensors="pt", padding=True).to(device)
                    text_out = model.text_model(**inputs)
                    projected = model.text_projection(text_out.pooler_output)
                    emb       = projected.squeeze(0).cpu().numpy()
                    seen[t] = emb
                    text_embeddings.append(emb)
                    descriptions.append(t)

    text_embeddings = np.stack(text_embeddings)
    np.savez(TEXT_OUT,
             embeddings=text_embeddings,
             descriptions=np.array(descriptions))
    print(f"Text embeddings   {text_embeddings.shape} → {TEXT_OUT}")


if __name__ == "__main__":
    extract()
