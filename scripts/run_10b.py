"""10B-token pretraining: Full BP / Frozen / GaLore with SIGTERM graceful shutdown.

Usage:
  # Full BP
  python scripts/run_10b.py --mode backprop --lr 1e-4 --run_name fullbp_10b

  # Frozen 3-layer (layers 6-11)
  python scripts/run_10b.py --mode frozen --split_start 6 --lr 2e-4 --run_name frozen_pos6_10b

  # GaLore
  python scripts/run_10b.py --mode galore --galore_rank 256 --lr 1e-3 --run_name galore_r256_10b

  # Resume after SIGTERM
  python scripts/run_10b.py --mode frozen --split_start 6 --lr 2e-4 --run_name frozen_pos6_10b --resume

SIGTERM handling:
  kill -TERM <pid>   → saves checkpoint, exits after current step
  kill -INT <pid>    → same (Ctrl+C)
"""
import os
os.environ.setdefault('HF_HOME', '/mnt/d/tmp/hf_cache')
os.environ.setdefault('HF_DATASETS_CACHE', '/mnt/d/tmp/hf_cache/datasets')
os.environ.setdefault('HF_HUB_CACHE', '/mnt/d/tmp/hf_cache/hub')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

import torch, torch.nn as nn, math, json, time, argparse, signal, sys
import numpy as np
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

device = 'cuda'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_LEN = 512

# ── SIGTERM handler ──
_should_stop = False

def _signal_handler(signum, frame):
    global _should_stop
    sig_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'
    print(f'\n[{sig_name}] Will save checkpoint and exit after current step...', flush=True)
    _should_stop = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ── Data loading (mmap) ──
class MMapDataLoader:
    def __init__(self, data_path, seq_len=512):
        self.tokens = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.tokens)
        self.seq_len = seq_len
        self.n_seqs = self.n_tokens // seq_len
        print(f'  Data: {self.n_tokens/1e9:.3f}B tokens, {self.n_seqs} sequences')

    def get_batch(self, batch_size, step):
        """Get a batch of sequences. Wraps around at epoch boundary."""
        seqs_per_step = batch_size
        start_seq = (step * seqs_per_step) % self.n_seqs
        batch = []
        for i in range(batch_size):
            seq_idx = (start_seq + i) % self.n_seqs
            start_tok = seq_idx * self.seq_len
            seq = self.tokens[start_tok:start_tok + self.seq_len].astype(np.int64)
            batch.append(seq)
        return torch.tensor(np.stack(batch), dtype=torch.long)


# ── Model ──
def make_model(seed=42):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=50257, hidden_size=2048, intermediate_size=5632,
        num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16,
        max_position_embeddings=512, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: Llama 1B ({n_params/1e6:.0f}M params)')
    return model


def get_layer_idx(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            try: return int(parts[i + 1])
            except: pass
    return None


# ── Optimizer setup ──
def setup_optimizer(model, args):
    """Returns optimizer and list of trainable param names."""
    n_layers = 24
    split_size = 6  # 25%

    if args.mode == 'frozen':
        bp_start = args.split_start
        bp_end = bp_start + split_size
        print(f'  Frozen: BP layers [{bp_start}, {bp_end}), rest frozen')
        trainable, frozen_names = [], []
        for name, param in model.named_parameters():
            idx = get_layer_idx(name)
            if idx is not None and (idx < bp_start or idx >= bp_end):
                param.requires_grad_(False)
                frozen_names.append(name)
            else:
                trainable.append(param)
        tp = sum(p.numel() for p in trainable)
        fp = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f'  Trainable: {tp/1e6:.0f}M, Frozen: {fp/1e6:.0f}M')
        opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    elif args.mode == 'galore':
        from galore_torch import GaLoreAdamW
        galore_params, regular_params = [], []
        for p in model.parameters():
            if p.dim() >= 2:
                galore_params.append(p)
            else:
                regular_params.append(p)
        print(f'  GaLore: {len(galore_params)} weight matrices, rank={args.galore_rank}')
        opt = GaLoreAdamW(
            [{'params': regular_params},
             {'params': galore_params, 'rank': args.galore_rank,
              'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}],
            lr=args.lr, weight_decay=0.01)
        trainable = list(model.parameters())

    else:  # backprop
        trainable = list(model.parameters())
        opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    return opt, trainable


# ── LR schedule ──
def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Eval ──
@torch.no_grad()
def evaluate(model, val_loader, max_batches=100):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(max_batches):
        ids = val_loader.get_batch(2, i).to(device)
        loss = model(input_ids=ids, labels=ids).loss.item()
        total_loss += loss * ids.numel()
        total_tok += ids.numel()
    model.train()
    return math.exp(total_loss / total_tok)


# ── Checkpoint ──
def save_checkpoint(model, optimizer, step, tokens_seen, best_ppl, log, args, path):
    print(f'  Saving checkpoint at step {step} ({tokens_seen/1e9:.3f}B tokens)...', flush=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'tokens_seen': tokens_seen,
        'best_ppl': best_ppl,
        'args': vars(args),
    }, path)
    # Save log separately (JSON, lightweight)
    log_path = path.replace('.pt', '_log.json')
    json.dump(log, open(log_path, 'w'), indent=2)
    print(f'  Checkpoint saved: {path} ({os.path.getsize(path)/1e9:.2f}GB)', flush=True)


def load_checkpoint(path, model, optimizer):
    print(f'  Loading checkpoint: {path}')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    step = ckpt['step']
    tokens_seen = ckpt['tokens_seen']
    best_ppl = ckpt.get('best_ppl', float('inf'))
    log_path = path.replace('.pt', '_log.json')
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    print(f'  Resumed from step {step} ({tokens_seen/1e9:.3f}B tokens), best_ppl={best_ppl:.2f}')
    return step, tokens_seen, best_ppl, log


# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--mode', required=True, choices=['backprop', 'frozen', 'galore'])
    parser.add_argument('--data', default='data/fineweb_edu_10b.bin')
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--total_tokens', type=int, default=10_000_000_000)
    parser.add_argument('--micro_bs', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--warmup_tokens', type=int, default=100_000_000,
                        help='Warmup in tokens (default 100M)')
    parser.add_argument('--eval_every_tokens', type=int, default=500_000_000,
                        help='Eval interval in tokens (default 500M)')
    parser.add_argument('--ckpt_every_tokens', type=int, default=1_000_000_000,
                        help='Checkpoint interval in tokens (default 1B)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_start', type=int, default=6,
                        help='Frozen: BP window start layer')
    parser.add_argument('--galore_rank', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # Derived
    tokens_per_step = args.micro_bs * SEQ_LEN
    tokens_per_update = tokens_per_step * args.grad_accum
    total_steps = args.total_tokens // tokens_per_step
    total_updates = args.total_tokens // tokens_per_update
    warmup_updates = args.warmup_tokens // tokens_per_update
    eval_every_updates = args.eval_every_tokens // tokens_per_update
    ckpt_every_updates = args.ckpt_every_tokens // tokens_per_update

    print(f'=== {args.run_name} ===')
    print(f'  Mode: {args.mode}, lr: {args.lr}, seed: {args.seed}')
    print(f'  micro_bs: {args.micro_bs}, grad_accum: {args.grad_accum}')
    print(f'  effective_bs: {args.micro_bs * args.grad_accum} seqs = {tokens_per_update} tokens/update')
    print(f'  total: {args.total_tokens/1e9:.1f}B tokens = {total_updates} updates')
    print(f'  warmup: {warmup_updates} updates, eval every {eval_every_updates} updates')
    print(f'  GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

    # Dirs
    ckpt_dir = os.path.join(BASE_DIR, 'checkpoints', '10b')
    results_dir = os.path.join(BASE_DIR, 'results', '10b')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{args.run_name}.pt')

    # Data
    val_path = args.data.replace('.bin', '_val.bin')
    print(f'Loading data: {args.data}')
    train_loader = MMapDataLoader(args.data, seq_len=SEQ_LEN)
    val_loader = MMapDataLoader(val_path, seq_len=SEQ_LEN)

    # Model + optimizer
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = make_model(seed=args.seed)
    opt, trainable = setup_optimizer(model, args)

    # Resume
    start_update = 0
    tokens_seen = 0
    best_ppl = float('inf')
    log = {'run_name': args.run_name, 'mode': args.mode, 'config': vars(args),
           'eval_tokens': [], 'eval_ppls': [], 'eval_times': []}

    if args.resume and os.path.exists(ckpt_path):
        start_update, tokens_seen, best_ppl, saved_log = load_checkpoint(
            ckpt_path, model, opt)
        start_update += 1  # resume from next update
        if saved_log:
            log = saved_log
    elif args.resume:
        print(f'  No checkpoint found at {ckpt_path}, starting fresh')

    # Initial eval
    if start_update == 0:
        ppl = evaluate(model, val_loader)
        print(f'  Initial PPL: {ppl:.2f}')
        log['initial_ppl'] = ppl

    model.train()
    t0 = time.time()
    running_loss = 0.0
    micro_step = 0
    next_eval_tokens = ((tokens_seen // args.eval_every_tokens) + 1) * args.eval_every_tokens
    next_ckpt_tokens = ((tokens_seen // args.ckpt_every_tokens) + 1) * args.ckpt_every_tokens

    print(f'\n  Training from update {start_update}, tokens_seen={tokens_seen/1e9:.3f}B\n')

    for update in range(start_update, total_updates):
        # LR schedule
        lr = get_lr(update, total_updates, warmup_updates, args.lr)
        for pg in opt.param_groups:
            pg['lr'] = lr

        # Gradient accumulation
        opt.zero_grad()
        accum_loss = 0.0
        for micro in range(args.grad_accum):
            global_micro = update * args.grad_accum + micro
            ids = train_loader.get_batch(args.micro_bs, global_micro).to(device)
            loss = model(input_ids=ids, labels=ids).loss
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            tokens_seen += tokens_per_step

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        running_loss += accum_loss / args.grad_accum

        # Logging
        if update % 100 == 0 and update > start_update:
            elapsed = time.time() - t0
            avg_loss = running_loss / min(100, update - start_update)
            tps = (tokens_seen - (start_update * tokens_per_update)) / elapsed
            eta = (args.total_tokens - tokens_seen) / tps if tps > 0 else 0
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f'  update {update:7d}/{total_updates} | loss {avg_loss:.4f} | '
                  f'lr {lr:.2e} | {tokens_seen/1e9:.3f}B tok | '
                  f'{tps/1e3:.1f}K tok/s | ETA {eta/3600:.1f}h | vram {vram:.1f}GB',
                  flush=True)
            running_loss = 0.0

        # Eval
        if tokens_seen >= next_eval_tokens:
            ppl = evaluate(model, val_loader)
            elapsed = time.time() - t0
            log['eval_tokens'].append(int(tokens_seen))
            log['eval_ppls'].append(ppl)
            log['eval_times'].append(elapsed)
            if ppl < best_ppl:
                best_ppl = ppl
            print(f'  === EVAL at {tokens_seen/1e9:.3f}B tokens: PPL={ppl:.2f} '
                  f'(best={best_ppl:.2f}) ===', flush=True)
            next_eval_tokens += args.eval_every_tokens

        # Periodic checkpoint
        if tokens_seen >= next_ckpt_tokens:
            save_checkpoint(model, opt, update, tokens_seen, best_ppl, log, args, ckpt_path)
            next_ckpt_tokens += args.ckpt_every_tokens

        # SIGTERM check
        if _should_stop:
            save_checkpoint(model, opt, update, tokens_seen, best_ppl, log, args, ckpt_path)
            print(f'Graceful exit at update {update}, tokens_seen={tokens_seen/1e9:.3f}B')
            sys.exit(0)

    # Final eval + save
    final_ppl = evaluate(model, val_loader)
    total_time = time.time() - t0
    log['final_ppl'] = final_ppl
    log['best_ppl'] = best_ppl
    log['total_time'] = total_time
    log['peak_vram'] = torch.cuda.max_memory_allocated() / 1e9

    print(f'\n=== DONE ===')
    print(f'  Final PPL: {final_ppl:.2f}, Best PPL: {best_ppl:.2f}')
    print(f'  Time: {total_time/3600:.1f}h, VRAM: {log["peak_vram"]:.1f}GB')

    # Save final results
    result_path = os.path.join(results_dir, f'{args.run_name}.json')
    json.dump(log, open(result_path, 'w'), indent=2)
    print(f'  Results: {result_path}')

    # Save final checkpoint
    final_ckpt = os.path.join(ckpt_dir, f'{args.run_name}_final.pt')
    torch.save(model.state_dict(), final_ckpt)
    print(f'  Final model: {final_ckpt}')


if __name__ == '__main__':
    main()
