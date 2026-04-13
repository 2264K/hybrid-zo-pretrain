# Hybrid ZO+SGD Pretraining — Companion Code

Companion code for the preprint:

> **Noise Over Gradients: Hybrid Backpropagation and Forward-Only Zeroth-Order Optimization for Memory-Efficient LLM Pretraining**
> Raen2264 (independent researcher, pseudonymous).
> Zenodo preprint v1, 2026. **DOI:** `10.5281/zenodo.XXXXXXX` *(to be filled on upload)*.

This repository is released as a **priority record (flag v1)** together with the Zenodo preprint. Subsequent versions will be tagged `v2`, `v3`, ... with their own Zenodo DOIs.

---

## How to cite

If you use this code or build on the ideas in the paper, please cite the Zenodo record:

```bibtex
@misc{raen2264_hybrid_zo_2026,
  author       = {Raen2264},
  title        = {{Noise Over Gradients}: Hybrid Backpropagation and Forward-Only Zeroth-Order Optimization for Memory-Efficient LLM Pretraining},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

The author name **Raen2264** is a pseudonym. When referencing this work, please retain the pseudonym exactly as written.

---

## What is this?

A single-GPU pretraining scheme that applies standard backpropagation + AdamW to the first **25%** of a network's transformer/SSM blocks, and a single-perturbation zeroth-order (ZO) estimate with plain SGD to the remaining **75%**. In the early-training regime studied (~20M tokens), this matches or improves validation perplexity against full backpropagation across:

- Llama 1B, GPT-2 Medium, Mamba 1B, Qwen3-Next 1B
- WikiText-103, C4, FineWeb-Edu

while saving **28–43% VRAM**. See the paper for accounting, ablations, and limitations.

**Important caveat (also in paper §Limitations).** The current split places the BP group on the *input side* of the network; backward therefore still traverses the back stack to deliver gradients to the front. The measured VRAM savings come primarily from removed optimizer state and back-group parameter gradients, not from fully eliminating back-layer activations. Results are single-seed, on a single RTX 5090, and restricted to the early-training regime.

---

## Layout

```
hybrid-zo-pretrain/
├── LICENSE                # Apache-2.0
├── NOTICE                 # attribution / pseudonym preservation clause
├── README.md              # this file
├── requirements.txt
├── scripts/
│   └── run_generality.py  # main experiment runner (all 4 architectures, 3 datasets)
└── results/
    └── gate2/             # raw JSON logs from the runs reported in the paper
```

---

## Quickstart

```bash
pip install -r requirements.txt

# GPT-2 Medium + WikiText, backprop baseline
python scripts/run_generality.py \
    --name G1a_gpt2m_backprop --model gpt2m --data wikitext --mode backprop

# GPT-2 Medium + WikiText, Hybrid (front 25% BP + back 75% ZO+SGD, k=1)
python scripts/run_generality.py \
    --name G1b_gpt2m_hybrid --model gpt2m --data wikitext --mode hybrid

# Llama 1B + C4, Hybrid
python scripts/run_generality.py \
    --name G2b_llama_c4_hybrid --model llama1b --data c4 --mode hybrid
```

Supported flags: `--model {gpt2m,gpt2l,llama1b,qwen3next,mamba1b}`, `--data {wikitext,c4,fineweb-edu}`, `--mode {backprop,hybrid}`, `--steps`, `--batch_size`, `--lr_front`, `--lr_back`, `--data_tokens`.

Outputs go to `results/gate2/gate2_<name>.json` with per-eval perplexity, wall time, and peak VRAM.

---

## Reproducing the paper tables

The JSON files under `results/gate2/` are the exact runs used in the paper's tables. File names follow this scheme:

| Prefix | Meaning |
| ------ | ------- |
| `G1a`, `G1b` | GPT-2 Medium baseline / hybrid on WikiText |
| `G2a`, `G2b` | Llama 1B on C4 |
| `G3a`, `G3b` | Llama 1B on FineWeb-Edu |
| `M1a`, `M1b` | Mamba 1B on WikiText |
| `Q1a`, `Q1b` | Qwen3-Next 1B on WikiText |
| `J*`, `L*`   | ZO ratio sweep (50/75/83/90 %) |
| `E*`         | ε magnitude sweep |
| `F*`         | back-Adam variant |

Every run was a single seed on a single RTX 5090 (32 GB), bf16 parameters, sequence length 512, micro-batch 2, 20,000 optimizer steps (~20 M tokens consumed).

---

## Licensing

- **Code** (this repository): Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). You may use, modify, and redistribute, but you must retain the copyright line and attribution clause (pseudonym `Raen2264` + Zenodo DOI).
- **Paper** (on Zenodo): Creative Commons Attribution 4.0 International (CC BY 4.0). You may reuse and adapt with credit to `Raen2264` and a link to the Zenodo DOI.

---

## Version history

- **v1 (2026)** — initial Zenodo preprint; scripts for generality runs; raw result JSONs.

---

## Contact

`infinitylawofbigbang@gmail.com`
