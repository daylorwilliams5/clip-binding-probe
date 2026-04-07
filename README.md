# CLIP Binding Probe

**Does CLIP's vision encoder encode attribute binding — and if so, where does it fail?**

CLIP can't tell *"a red cube and a blue sphere"* apart from *"a blue cube and a red sphere."* Prior work (Winoground, ARO) shows this failure behaviorally. This project probes **where inside the vision encoder** binding information appears and disappears.

## Key Finding

CLIP's vision encoder *does* encode which color goes with which shape — but only in its middle layers. Layers 3–5 of the ViT contain statistically significant binding information (confirmed via permutation testing, p < 0.001). Later layers progressively erase it. By the final layer, binding accuracy collapses to chance.

The failure isn't perceptual — CLIP's intermediate representations do see binding. It's architectural: the later layers discard binding information, likely because CLIP's contrastive training objective doesn't reward preserving fine-grained attribute-object associations.

This matters for any product using CLIP embeddings (search, content understanding, AR): the model literally cannot distinguish scenes that differ only in how attributes are assigned to objects.

## Results

### Retrieval test — does CLIP pick the right caption?

| Metric | Accuracy | Chance |
|--------|----------|--------|
| Image-level | 0.549 | 0.50 |
| Group-level | 0.125 | 0.25 |

Group-level below chance means CLIP tends to match *both* images in a foil pair to the same caption — it sees them as interchangeable.

### Binding probe — is binding info in the final embedding?

| Task | Accuracy | Chance |
|------|----------|--------|
| Color-shape binding | 0.166 | 0.083 |
| Color left | 0.276 | 0.250 |
| Color right | 0.284 | 0.250 |

The final embedding carries almost no usable binding information. Color identity alone is at chance.

### Layer probe — where does binding info live and die?

![Layer probe results](figures/layer_probe_accuracy.png)

Binding information peaks at layers 3–5, is statistically significant through layer 10 (permutation test, 1000 iterations, p < 0.001), and falls back to chance by layer 12. The gray band shows the null distribution from shuffled labels.

## Pipeline

```
Synthetic stimuli (shape pairs + foils)
        ↓
CLIP ViT-B/32 embeddings (final layer + all intermediate layers)
        ↓
Retrieval test     — does CLIP rank correct text above foil?
Binding probe      — linear probe on final vision embeddings
Layer probe        — linear probe at each ViT layer (with permutation test)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python stimuli/generate_stimuli.py       # synthetic shape pairs
python embeddings/extract_clip.py        # CLIP vision + text embeddings
python analysis/retrieval_test.py        # behavioral retrieval accuracy
python analysis/binding_probe.py         # linear probe on final embeddings
python analysis/layer_probe.py           # layer-wise binding probe (slow — ~20 min)
```

## Stimuli

Synthetic images of shape pairs (circle, square, triangle, star) in four colors (red, blue, green, yellow). For each pair, a foil image swaps the color-shape bindings while keeping object positions fixed.

- 16 singleton images (one shape, one color)
- 144 pair images (72 foil pairs)

## Method

**Retrieval test:** For each foil pair, compute cosine similarity between CLIP vision and text embeddings. CLIP succeeds if it ranks the correct description above the foil. Group-level requires both images in a pair to be correct.

**Binding probe:** Logistic regression on CLIP's final vision embeddings to decode which color goes with which shape. 5-fold cross-validation.

**Layer probe:** Same logistic regression applied to the CLS token at each of the 13 ViT layers. 1000-permutation null distribution at each layer confirms whether observed accuracy is significantly above chance.

## Related work

- Thrush et al. (2022). [Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality.](https://arxiv.org/abs/2204.03162)
- Yuksekgonul et al. (2022). [When and Why Vision-Language Models Behave like Bag-of-Words Models, and What to Do About It.](https://arxiv.org/abs/2210.01936)
