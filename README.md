# Specify and Edit: Overcoming Ambiguity in Text‑Based Image Editing

> **BMVC 2025** · [arXiv:2407.20232](https://arxiv.org/abs/2407.20232)

This repository contains the official implementation of **“Specify and Edit: Overcoming Ambiguity in Text‑Based Image Editing”**. Our method, **SANE** (Specify‑And‑Edit), resolves ambiguous user instructions by first *decomposing* them into precise sub‑edits and then executing each step with an image‑editing diffusion model.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fabvio/SANE.git
cd SANE
```

### 2. Create the conda environment

```bash
conda env create -f requirements.yml   # creates "sane" env by default
conda activate sane
```

*The `requirements.yml` already pins Python ≥3.10, PyTorch ≥2.2, Diffusers, and all other dependencies tested for this paper.*

---

## Dataset format

```
<dataset_path>/
├── original_images/    # JPEG/PNG input images
│   ├── xxx.jpg
│   └── …
└── instructions/       # plain‑text editing prompts (one per image)
    ├── xxx.txt
    └── …
    
    
```

**Naming consistency:** For every `xxx.jpg` (or `.png`) in **`original_images/`**, there must be a matching `xxx.txt` in **`instructions/`**.

---

## Pre‑processing

Before training or evaluation you may want to generate

1. **captions** (if your dataset lacks them), and
2. **decomposed instructions** for disambiguation.

### 1. Generate captions *(optional)*

```bash
# Environment variables (only need to export once per session)
export PYTHONPATH=.
export OPENAI_API_KEY=<your_openai_api_key>

python preprocess/caption.py \
  --ds_path <dataset_path>
```

By default we query the OpenAI Vision model to caption every image into a folder in `<dataset_path>`.

### 2. Decompose instructions

```bash
python preprocess/decompose.py --ds_path <dataset_path>
```

This creates a decomposed instruction folder containing specific edits for SANE.

---

## Inference

### 1. Run SANE

```bash
python infer/sane_inference.py --ds_path <dataset_path>
```

Outputs are written to `<output_dir>/results/sane/`.

### 2. Swap in another diffusion model

```bash
python infer/sane_inference.py \
  --ds_path <dataset_path> \
  --model_id <model_name>
```

Any [Hugging Face Diffusers](https://huggingface.co/docs/diffusers) model based on *InstructPix2Pix* will work (e.g. `timbrooks/instruct‑pix2pix`).

---

## Citation

If you build upon this work, please cite:

```bibtex
@inproceedings{iakovleva2024specify,
  title={Specify and Edit: Overcoming Ambiguity in Text-Based Image Editing},
  author={Iakovleva, Ekaterina and Pizzati, Fabio and Torr, Philip and Lathuili{\`e}re, St{\'e}phane},
  booktitle={The British Machine Vision Conference},
  year={2025}
}
```
