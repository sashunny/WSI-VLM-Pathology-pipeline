# WSI + VLM Digital Pathology Pipeline

A hands-on, end-to-end pipeline for whole-slide image (WSI) analysis in digital pathology:
tissue-aware tiling, a patch-level tissue classifier, slide-level heatmap visualization, and a
CLIP + FAISS vision-language retrieval layer for automated report generation.

## Overview

Whole-slide images are gigapixel-scale digitized microscope slides that can't be processed as a
single image. This project implements the standard patch-based workflow used across digital
pathology deep learning:

```
tile the slide → filter tissue vs. background → classify patches → stitch predictions
into a slide-level heatmap → embed patches with a VLM → retrieve/report on findings
```

## Features

- **WSI I/O & tiling** — reads pyramidal `.svs`/`.tiff` slides via OpenSlide, with Otsu-threshold
  tissue detection to skip background regions
- **Patch classification** — ResNet18 fine-tuned on PathMNIST (9-class histopathology tissue
  types), with accuracy/AUROC evaluation
- **Explainability** — Grad-CAM localization maps for weakly-supervised region-of-interest
  visualization
- **Slide-level heatmaps** — stitches patch predictions back into a spatial overlay on the
  original slide
- **VLM-based retrieval & reporting** — CLIP image/text embeddings indexed with FAISS for
  content-based patch retrieval, combined into a structured markdown report

## Project structure

```
├── src/
│   ├── utils.py                  # tissue detection & tiling helpers
│   ├── wsi_tiling.py           # WSI → tissue-filtered patches
│   ├── train_classifier.py     # fine-tune ResNet18 on PathMNIST
│   ├── inference_heatmap.py    # patch inference → slide heatmap + Grad-CAM
│   └── vlm_clip_report.py      # CLIP embeddings → FAISS → generated report
├── data/                         # place WSI files here
├── outputs/                      # generated patches, heatmaps, reports
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

All dependencies, including OpenSlide, install via pip (`openslide-bin` ships precompiled
binaries — no system/apt packages required).

Download a sample WSI (public Aperio test slide, ~180MB):

```bash
mkdir -p data
wget -O data/CMU-1.svs "http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
```

## Usage

Run the scripts in order:

```bash
python src/wsi_tiling.py
python src/train_classifier.py
python src/inference_heatmap.py
python src/vlm_clip_report.py
```

Each script writes its outputs to `outputs/`, including tissue masks, extracted patches, trained
model checkpoints, heatmap overlays, Grad-CAM visualizations, CLIP retrieval contact sheets, and
a generated markdown report.

## Scope & limitations

- The classifier is trained on PathMNIST (colorectal tissue) and applied to a different WSI to
  demonstrate the pipeline mechanics; production use would require training on task-matched
  WSI-derived patches.
- Retrieval uses general-purpose CLIP rather than a pathology-specific vision-language model
  (e.g. PLIP, CONCH); the architecture (embed → FAISS → retrieve → ground a report) transfers
  directly to those domain-specific encoders.

## License

Sample data from OpenSlide's public test data repository and PathMNIST (MedMNIST) are used under
their respective open licenses. See individual dataset sources for details.
