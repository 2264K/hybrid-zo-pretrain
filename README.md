# Which Layers Need Backpropagation? — Companion Code

Companion code for the preprint:

> **Which Layers Need Backpropagation? Position-Dependent Partial Training for Memory-Efficient LLM Pretraining**
> Raen2264 (independent researcher, pseudonymous).
> Zenodo preprint v2, 2026. **DOI:** [10.5281/zenodo.20392068](https://doi.org/10.5281/zenodo.20392068)

This is **v2**, a substantial revision of v1 (*"Noise Over Gradients"*). v1's two headline claims were re-examined: the output-side BP recommendation was overturned by a position sweep, and the "partial beats full BP" claim—reached in v1 via an under-tuned baseline at a 20M-token horizon—turns out to be **correct at scale** when baselines are properly tuned and training is extended to 10B tokens.

---

## TL;DR

When pretraining a transformer from scratch, you do not need to backpropagate through every layer. Backpropagating a well-chosen **25%** of layers (early-middle for attention models) while **freezing the rest at random initialization**:

- **wins at scale**: at 10B tokens (FineWeb-Edu), frozen-partial reaches **PPL 35.2**, beating tuned full backprop (**39.0**) and GaLore (**52.4**)—the short-horizon ranking *inverts*;
- uses **the least VRAM** (10.9 GB vs. 17.4 GB full BP at matched batch size; 6.8 GB at batch 2);
- is **the most learning-rate-robust** (6% PPL variation over a 3× lr range, vs. 110% for full BP and 1,749% for GaLore);
- is **architecture-dependent**: attention models prefer an early-middle BP window; recurrent/SSM models require the input side.

The non-BP layers can be frozen entirely—zeroth-order perturbation of them contributes at most ~4 PPL—so the method reduces to: **backpropagate a quarter, freeze three quarters.**

---

## Key result: the ranking inverts with training horizon

| Method | 20M tokens (WikiText) | **10B tokens (FineWeb-Edu)** | VRAM |
| ------ | --------------------- | ---------------------------- | ---- |
| Frozen-partial (pos6) | 185 (3rd) | **35.2 (1st)** | 6.8 GB |
| Full BP | 159 (2nd) | 39.0 (2nd) | 14.5 GB |
| GaLore (rank 256) | 149 (1st) | 52.4 (3rd) | 8.5 GB |

Zero-shot downstream (10B checkpoints, lm-eval-harness):

| Method | HellaSwag | PIQA | LAMBADA |
| ------ | --------- | ---- | ------- |
| Frozen-partial | 27.4 | 57.3 | **10.5** |
| Full BP | 27.7 | 57.4 | 8.7 |
| GaLore | 26.2 | 54.4 | 3.9 |
| *(chance)* | *25.0* | *50.0* | *~0* |

---

## Layout

```
hybrid-zo-pretrain/
├── LICENSE                    # Apache-2.0
├── NOTICE                     # attribution / pseudonym preservation clause
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_generality.py      # short-horizon runner (4 architectures, 3 datasets,
│   │                          #   position sweep, frozen/hybrid/galore modes)
│   ├── run_10b.py             # 10B-token runner with checkpoint/resume + SIGTERM
│   ├── preprocess_fineweb.py  # tokenize FineWeb-Edu → flat uint16 binary
│   ├── eval_10b.py            # zero-shot downstream eval (HellaSwag/PIQA/LAMBADA)
│   └── stability_analysis.py  # Hessian / Jacobian / gradient-covariance probes
└── results/
    ├── gate2/                 # short-horizon JSON logs (paper tables)
    └── 10b/                   # 10B training + eval JSON logs
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

**Short-horizon (20M tokens), position sweep:**
```bash
# Full backprop baseline (tune lr!)
python scripts/run_generality.py --name fullbp --model llama1b --mode backprop --lr_front 1e-4

# Frozen-partial: BP on a 6-layer window starting at layer 6, rest frozen
python scripts/run_generality.py --name frozen_pos6 --model llama1b --mode frozen \
    --split_start 6 --lr_front 2e-4

# GaLore baseline
python scripts/run_generality.py --name galore --model llama1b --mode backprop \
    --galore_rank 256 --lr_front 1e-3
```

**10B-token run (with resume + graceful SIGTERM checkpointing):**
```bash
# 1. Preprocess once (~20 GB output)
python scripts/preprocess_fineweb.py --output data/fineweb_edu_10b.bin --max_tokens 10200000000

# 2. Train (re-run the same command to resume from the latest checkpoint)
python scripts/run_10b.py --run_name frozen_pos6_10b --mode frozen --split_start 6 --lr 2e-4 \
    --data data/fineweb_edu_10b.bin --total_tokens 10000000000 --resume

# 3. Evaluate
python scripts/eval_10b.py --checkpoint checkpoints/10b/frozen_pos6_10b_final.pt --run_name frozen_pos6_10b
```

Modes: `--mode {backprop,hybrid,frozen}` plus `--galore_rank R` (R>0 enables GaLore in backprop mode).
`run_10b.py` traps `SIGTERM`/`SIGINT` to save a checkpoint and exit, so it survives SLURM wall-time limits—re-submit with `--resume` to continue.

---

## Reproducing the paper

- `results/gate2/` — short-horizon runs (position sweep, architecture comparison, lr sweep, frozen/ZO/embed ablations).
- `results/10b/` — the 10B training logs (`*_s42.json`) and downstream eval (`*_eval.json`) for all three methods.

All 10B runs: Llama 1B, seed 42, effective batch size 64 (32,768 tokens/update), warmup 100M → cosine decay, identical data and order across methods; only the per-method optimal learning rate differs.

---

## How to cite

```bibtex
@misc{raen2264_which_layers_2026,
  author       = {Raen2264},
  title        = {{Which Layers Need Backpropagation? Position-Dependent
                   Partial Training for Memory-Efficient LLM Pretraining}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v2},
  doi          = {10.5281/zenodo.20392068},
  url          = {https://doi.org/10.5281/zenodo.20392068}
}
```

The author name **Raen2264** is a pseudonym; please retain it exactly as written.

---

## Licensing

- **Code**: Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Retain the copyright line and attribution clause (pseudonym `Raen2264` + Zenodo DOI).
- **Paper** (on Zenodo): CC BY 4.0. Reuse/adapt with credit to `Raen2264` and a link to the Zenodo DOI.

The author name **Raen2264** is a pseudonym; please retain it exactly as written.

---

## Version history

- **v2 (2026)** — position sweep + architecture dependence; properly tuned baselines; GaLore comparison; **10B-token horizon showing frozen-partial overtakes full BP**; downstream eval.
- **v1 (2026)** — initial Zenodo preprint (*Noise Over Gradients*); hybrid ZO+SGD generality runs.

---

## Contact

`infinitylawofbigbang@gmail.com`
