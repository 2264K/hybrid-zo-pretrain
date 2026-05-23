"""Stability analysis: Hessian eigenvalue, Jacobian/Lipschitz, gradient covariance.

Usage:
  # Full BP checkpoint — Hessian w.r.t. all params
  python scripts/stability_analysis.py --checkpoint checkpoints/fullbp_lr1e4.pt --mode fullbp

  # Frozen checkpoint — Hessian w.r.t. trainable params only
  python scripts/stability_analysis.py --checkpoint checkpoints/frozen_lr1e4.pt --mode frozen --split_start 7

  # Full BP checkpoint — Hessian w.r.t. frozen subset (Cauchy interlacing test)
  python scripts/stability_analysis.py --checkpoint checkpoints/fullbp_lr1e4.pt --mode fullbp_subset --split_start 7
"""
import os
os.environ['HF_HOME'] = '/mnt/d/tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/d/tmp/hf_cache/datasets'
os.environ['HF_HUB_CACHE'] = '/mnt/d/tmp/hf_cache/hub'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/d/tmp/hf_cache/transformers'

import torch, torch.nn as nn, math, json, time, argparse
import numpy as np
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from datasets import load_dataset

device = 'cuda'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_LEN = 512


def load_val_data(max_seqs=200):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    texts = [t for t in ds['text'] if t.strip()]
    all_tokens = tokenizer(texts, return_attention_mask=False)['input_ids']
    flat = [tok for doc in all_tokens for tok in doc]
    total_seqs = len(flat) // SEQ_LEN
    # Use last portion as val
    start = total_seqs - max_seqs
    seqs = [flat[i*SEQ_LEN:(i+1)*SEQ_LEN] for i in range(start, start + max_seqs)]
    return torch.tensor(seqs, dtype=torch.long)


def make_model():
    config = LlamaConfig(
        vocab_size=50257, hidden_size=2048, intermediate_size=5632,
        num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16,
        max_position_embeddings=512, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).to(device)
    return model


def get_layer_idx(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i+1 < len(parts):
            try: return int(parts[i+1])
            except: pass
    return None


def get_trainable_params(model, mode, split_start=7, split_size=3):
    """Return list of (name, param) for the trainable subset."""
    bp_start, bp_end = split_start, split_start + split_size
    result = []
    for name, param in model.named_parameters():
        idx = get_layer_idx(name)
        if mode == 'fullbp':
            result.append((name, param))
        elif mode in ('frozen', 'fullbp_subset'):
            # Only embed/head/norm (idx=None) + layers in BP range
            if idx is None or (bp_start <= idx < bp_end):
                result.append((name, param))
    return result


# ── Hessian top eigenvalue via finite-difference power iteration ──

def hessian_top_eigenvalue(model, data, trainable_params, n_iter=50, eps=1e-3, n_batches=8):
    """Estimate λ_max of the Hessian w.r.t. trainable_params."""
    params = [p for _, p in trainable_params]
    n_params = sum(p.numel() for p in params)
    print(f'  Hessian: {n_params/1e6:.0f}M trainable params, {n_iter} iterations')

    # Initialize random vector v
    v = [torch.randn_like(p, dtype=torch.float32) for p in params]
    norm = sum((vi**2).sum() for vi in v).sqrt()
    v = [vi / norm for vi in v]

    eigenvalues = []
    for it in range(n_iter):
        # Average Hv over multiple batches for stability
        Hv_acc = [torch.zeros_like(vi) for vi in v]

        for b in range(n_batches):
            idx = (it * n_batches + b) % len(data)
            ids = data[idx:idx+1].to(device)

            # g(θ)
            model.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            g0 = [p.grad.float().clone() for p in params]

            # Perturb θ → θ + εv
            with torch.no_grad():
                for p, vi in zip(params, v):
                    p.data.add_(eps * vi.to(p.dtype))

            # g(θ + εv)
            model.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            g1 = [p.grad.float().clone() for p in params]

            # Restore θ
            with torch.no_grad():
                for p, vi in zip(params, v):
                    p.data.sub_(eps * vi.to(p.dtype))

            # Hv ≈ (g1 - g0) / eps
            for i in range(len(params)):
                Hv_acc[i] += (g1[i] - g0[i]) / eps

        # Average
        Hv = [h / n_batches for h in Hv_acc]

        # Rayleigh quotient: λ = v · Hv
        lam = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv)).item()

        # Update v ← Hv / ||Hv||
        norm = sum((hvi**2).sum() for hvi in Hv).sqrt()
        if norm.item() < 1e-10:
            print(f'  iter {it}: norm collapsed, stopping')
            break
        v = [hvi / norm for hvi in Hv]
        eigenvalues.append(lam)

        if it % 10 == 0:
            print(f'  iter {it:3d}: λ_max ≈ {lam:.4f}')

    model.zero_grad()
    return eigenvalues[-1] if eigenvalues else 0.0, eigenvalues


# ── Per-layer Jacobian / Lipschitz estimate ──

def jacobian_lipschitz(model, data, n_samples=50, eps=0.01):
    """Measure per-layer Lipschitz constant via input perturbation."""
    model.eval()
    hooks = []
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach()
        return hook_fn

    # Register hooks on each transformer layer
    layers = model.model.layers
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(f'layer_{i}')))

    lipschitz_per_layer = {f'layer_{i}': [] for i in range(len(layers))}

    for s in range(n_samples):
        idx = s % len(data)
        ids = data[idx:idx+1].to(device)

        # Clean forward
        with torch.no_grad():
            _ = model(input_ids=ids)
        clean_acts = {k: v.clone() for k, v in activations.items()}

        # Perturb embedding output
        embed_out = model.model.embed_tokens(ids)
        delta = torch.randn_like(embed_out) * eps
        perturbed_embed = embed_out + delta
        delta_norm = delta.norm().item()

        # Forward with perturbed embedding (manual pass through layers)
        hidden = perturbed_embed
        # Apply rotary position encoding setup
        position_ids = torch.arange(ids.shape[1], device=device).unsqueeze(0)
        # We need to use the model's internal forward for proper position handling
        # Instead, let's use a different approach: hook-based measurement

        # Alternative: measure ratio of activation change at each layer
        # by running two forward passes with slightly different inputs
        ids_pert = ids.clone()
        # Perturb a random token position
        pos = torch.randint(0, ids.shape[1], (1,)).item()
        # Change token to a random nearby token
        ids_pert[0, pos] = (ids_pert[0, pos] + torch.randint(1, 100, (1,)).item()) % 50257

        with torch.no_grad():
            _ = model(input_ids=ids_pert)
        pert_acts = {k: v.clone() for k, v in activations.items()}

        # Compute per-layer sensitivity
        for i in range(len(layers)):
            key = f'layer_{i}'
            diff = (pert_acts[key] - clean_acts[key]).norm().item()
            if i == 0:
                input_diff = (model.model.embed_tokens(ids_pert) - model.model.embed_tokens(ids)).norm().item()
            else:
                prev_key = f'layer_{i-1}'
                input_diff = (pert_acts[prev_key] - clean_acts[prev_key]).norm().item()
            if input_diff > 1e-8:
                lipschitz_per_layer[key].append(diff / input_diff)

    for h in hooks:
        h.remove()

    # Summarize
    result = {}
    for key, values in lipschitz_per_layer.items():
        if values:
            result[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'max': float(np.max(values)),
                'min': float(np.min(values)),
            }
    model.train()
    return result


# ── Gradient covariance across mini-batches ──

def gradient_covariance(model, data, trainable_params, n_batches=50):
    """Measure per-layer gradient consistency across mini-batches."""
    model.train()
    params = [p for _, p in trainable_params]
    names = [n for n, _ in trainable_params]

    # Group by layer
    layer_groups = {}
    for i, name in enumerate(names):
        idx = get_layer_idx(name)
        key = f'layer_{idx}' if idx is not None else 'embed_head'
        if key not in layer_groups:
            layer_groups[key] = []
        layer_groups[key].append(i)

    # Collect gradients per layer per batch
    layer_grads = {k: [] for k in layer_groups}

    for b in range(n_batches):
        idx = b % len(data)
        ids = data[idx:idx+1].to(device)
        model.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()

        for key, param_indices in layer_groups.items():
            grad_vec = torch.cat([params[i].grad.flatten().float() for i in param_indices])
            layer_grads[key].append(grad_vec)

    model.zero_grad()

    # Compute pairwise cosine similarity
    result = {}
    for key, grads in layer_grads.items():
        if len(grads) < 2:
            continue
        # Stack into matrix [n_batches, n_params]
        G = torch.stack(grads)
        # Normalize rows
        norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G_norm = G / norms
        # Cosine similarity matrix
        cos_sim = (G_norm @ G_norm.T)
        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
        cos_vals = cos_sim[mask]

        # Gradient norm statistics
        grad_norms = norms.squeeze()

        result[key] = {
            'cos_sim_mean': float(cos_vals.mean()),
            'cos_sim_std': float(cos_vals.std()),
            'grad_norm_mean': float(grad_norms.mean()),
            'grad_norm_std': float(grad_norms.std()),
            'grad_norm_cv': float(grad_norms.std() / grad_norms.mean()),  # coefficient of variation
        }
        print(f'  {key}: cos_sim={cos_vals.mean():.4f}±{cos_vals.std():.4f}, '
              f'grad_norm={grad_norms.mean():.2e}±{grad_norms.std():.2e}')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', required=True, choices=['fullbp', 'frozen', 'fullbp_subset'],
                        help='fullbp: all params. frozen: trainable subset. fullbp_subset: Cauchy test.')
    parser.add_argument('--split_start', type=int, default=7)
    parser.add_argument('--split_size', type=int, default=3)
    parser.add_argument('--hessian_iter', type=int, default=50)
    parser.add_argument('--hessian_batches', type=int, default=8)
    parser.add_argument('--lipschitz_samples', type=int, default=50)
    parser.add_argument('--grad_cov_batches', type=int, default=50)
    parser.add_argument('--skip_hessian', action='store_true')
    parser.add_argument('--skip_lipschitz', action='store_true')
    parser.add_argument('--skip_grad_cov', action='store_true')
    args = parser.parse_args()

    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Loading data...')
    val_data = load_val_data(max_seqs=200)
    print(f'Val data: {val_data.shape}')

    print(f'Loading model + checkpoint: {args.checkpoint}')
    model = make_model()
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    del state

    trainable = get_trainable_params(model, args.mode, args.split_start, args.split_size)
    n_trainable = sum(p.numel() for _, p in trainable)
    print(f'Mode: {args.mode}, trainable: {n_trainable/1e6:.0f}M params')

    results = {
        'checkpoint': args.checkpoint,
        'mode': args.mode,
        'split_start': args.split_start,
        'split_size': args.split_size,
        'n_trainable': n_trainable,
    }

    # 1. Hessian top eigenvalue
    if not args.skip_hessian:
        print(f'\n=== Hessian Top Eigenvalue ===')
        t0 = time.time()
        lam_max, lam_history = hessian_top_eigenvalue(
            model, val_data, trainable,
            n_iter=args.hessian_iter, n_batches=args.hessian_batches)
        hessian_time = time.time() - t0
        print(f'  λ_max = {lam_max:.6f} (η_max ≈ {2/abs(lam_max) if lam_max != 0 else float("inf"):.6f})')
        print(f'  Time: {hessian_time/60:.1f} min')
        results['hessian'] = {
            'lambda_max': lam_max,
            'eta_max': 2/abs(lam_max) if lam_max != 0 else float('inf'),
            'history': lam_history,
            'time_sec': hessian_time,
        }

    # 2. Jacobian / Lipschitz
    if not args.skip_lipschitz:
        print(f'\n=== Jacobian / Lipschitz ===')
        t0 = time.time()
        lip = jacobian_lipschitz(model, val_data, n_samples=args.lipschitz_samples)
        lip_time = time.time() - t0
        print(f'  Time: {lip_time/60:.1f} min')
        results['lipschitz'] = lip
        results['lipschitz_time_sec'] = lip_time

    # 3. Gradient covariance
    if not args.skip_grad_cov:
        print(f'\n=== Gradient Covariance ===')
        # For grad cov, need gradients only for trainable params
        # Set requires_grad accordingly
        for name, param in model.named_parameters():
            param.requires_grad_(False)
        for name, param in trainable:
            param.requires_grad_(True)
        t0 = time.time()
        gcov = gradient_covariance(model, val_data, trainable, n_batches=args.grad_cov_batches)
        gcov_time = time.time() - t0
        print(f'  Time: {gcov_time/60:.1f} min')
        results['grad_covariance'] = gcov
        results['grad_cov_time_sec'] = gcov_time

    # Save
    save_dir = os.path.join(BASE_DIR, 'results', 'stability')
    os.makedirs(save_dir, exist_ok=True)
    ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
    fname = os.path.join(save_dir, f'stability_{ckpt_name}_{args.mode}.json')
    json.dump(results, open(fname, 'w'), indent=2, default=str)
    print(f'\nSaved: {fname}')


if __name__ == '__main__':
    main()
