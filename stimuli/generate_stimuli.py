"""
Generate synthetic binding stimuli for the CLIP attribute binding probe.

For each (shape_A, shape_B, color_1, color_2) combination:
  - binding_1: shape_A in color_1 (left),  shape_B in color_2 (right)
  - binding_2: shape_A in color_2 (left),  shape_B in color_1 (right)  ← foil

Also generates singleton images for each (shape, color) combo.

Outputs:
  stimuli/images/      — PNG files
  stimuli/metadata.csv — full image manifest
  stimuli/pairs.csv    — foil pair manifest (image_1 vs image_2)
"""

import os
import csv
from itertools import permutations, combinations

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "star"]
COLORS = {
    "red":    (220, 50,  50),
    "blue":   (50,  100, 220),
    "green":  (50,  180, 70),
    "yellow": (230, 200, 40),
}

IMG_SIZE   = 224
SHAPE_SIZE = 45
BACKGROUND = (245, 245, 245)
OUT_DIR    = os.path.join(os.path.dirname(__file__), "images")


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


def make_singleton(shape, color_rgb):
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape, IMG_SIZE // 2, IMG_SIZE // 2, SHAPE_SIZE, color_rgb)
    return img


def make_pair_image(shape_a, color_a, shape_b, color_b):
    """Place shape_a on the left half, shape_b on the right half."""
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape_a, IMG_SIZE // 4,     IMG_SIZE // 2, SHAPE_SIZE, color_a)
    draw_shape(draw, shape_b, 3 * IMG_SIZE // 4, IMG_SIZE // 2, SHAPE_SIZE, color_b)
    return img


def generate_all(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    singleton_records = []
    pair_records      = []
    foil_pairs        = []

    color_items = list(COLORS.items())

    # ── Singletons ────────────────────────────────────────────────────────────
    for shape in SHAPES:
        for cname, crgb in COLORS.items():
            img   = make_singleton(shape, crgb)
            fname = f"singleton_{shape}_{cname}.png"
            img.save(os.path.join(out_dir, fname))
            singleton_records.append({
                "filename": fname, "type": "singleton",
                "shape": shape, "color": cname,
                "pair_id": "", "shape_left": "", "color_left": "",
                "shape_right": "", "color_right": "", "text": "",
            })

    # ── Pairs + Foils ─────────────────────────────────────────────────────────
    pair_id = 0
    for shape_a, shape_b in permutations(SHAPES, 2):
        for (c1_name, c1_rgb), (c2_name, c2_rgb) in combinations(color_items, 2):

            img1   = make_pair_image(shape_a, c1_rgb, shape_b, c2_rgb)
            fname1 = f"pair{pair_id:04d}_{shape_a}_{c1_name}_{shape_b}_{c2_name}.png"
            img1.save(os.path.join(out_dir, fname1))

            img2   = make_pair_image(shape_a, c2_rgb, shape_b, c1_rgb)
            fname2 = f"pair{pair_id:04d}_{shape_a}_{c2_name}_{shape_b}_{c1_name}.png"
            img2.save(os.path.join(out_dir, fname2))

            text1 = f"a {c1_name} {shape_a} and a {c2_name} {shape_b}"
            text2 = f"a {c2_name} {shape_a} and a {c1_name} {shape_b}"

            for fname, shape_l, col_l, shape_r, col_r, text in [
                (fname1, shape_a, c1_name, shape_b, c2_name, text1),
                (fname2, shape_a, c2_name, shape_b, c1_name, text2),
            ]:
                pair_records.append({
                    "filename": fname, "type": "pair", "pair_id": pair_id,
                    "shape_left": shape_l, "color_left": col_l,
                    "shape_right": shape_r, "color_right": col_r,
                    "text": text, "shape": "", "color": "",
                })

            foil_pairs.append({
                "pair_id": pair_id,
                "image_1": fname1, "text_1": text1,
                "image_2": fname2, "text_2": text2,
                "shape_a": shape_a, "shape_b": shape_b,
                "color_1": c1_name, "color_2": c2_name,
            })
            pair_id += 1

    # ── Write manifests ───────────────────────────────────────────────────────
    fieldnames = ["filename", "type", "pair_id", "shape", "color",
                  "shape_left", "color_left", "shape_right", "color_right", "text"]
    manifest_path = os.path.join(os.path.dirname(__file__), "metadata.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(singleton_records + pair_records)

    foil_path = os.path.join(os.path.dirname(__file__), "pairs.csv")
    with open(foil_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(foil_pairs[0].keys()))
        writer.writeheader()
        writer.writerows(foil_pairs)

    print(f"Generated {len(singleton_records)} singletons")
    print(f"Generated {len(pair_records)} pair images ({len(foil_pairs)} foil pairs)")
    print(f"Manifest → {manifest_path}")
    print(f"Pairs    → {foil_path}")


if __name__ == "__main__":
    generate_all()
