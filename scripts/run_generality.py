"""Architecture + Data generality test for M1b Hybrid ZO+SGD.

Usage:
  # GPT-2 Medium + WikiText
  python scripts/run_generality.py --name G1_gpt2m_wikitext --model gpt2m --mode hybrid

  # Llama 1B + C4
  python scripts/run_generality.py --name G2_llama_c4 --model llama1b --data c4 --mode hybrid
"""
import os
# HF cache to D: (MUST be set before any HF imports)
os.environ['HF_HOME'] = '/mnt/d/tmp/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/mnt/d/tmp/hf_cache/datasets'
os.environ['HF_HUB_CACHE'] = '/mnt/d/tmp/hf_cache/hub'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/d/tmp/hf_cache/transformers'

import torch, torch.nn as nn, math, json, time, argparse
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

device = 'cuda'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "results", "gate2")
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 512

def load_data(dataset='wikitext', max_tokens=80_000_000):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    text_field = 'text'
    if dataset == 'wikitext':
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        texts = [t for t in ds[text_field] if t.strip()]
    elif dataset == 'c4':
        # Streaming to avoid full download
        ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        texts = []
        for item in ds:
            t = item[text_field]
            if t.strip():
                texts.append(t)
            if len(texts) >= 200000: break
    elif dataset == 'fineweb-edu':
        ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train', streaming=True)
        texts = []
        for item in ds:
            t = item[text_field]
            if t.strip():
                texts.append(t)
            if len(texts) >= 200000: break
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    print(f'  {len(texts)} docs loaded')
    all_tokens = tokenizer(texts, return_attention_mask=False)['input_ids']
    flat = [tok for doc in all_tokens for tok in doc]
    total_seqs = len(flat) // SEQ_LEN
    max_seqs = max_tokens // SEQ_LEN
    print(f'  Available seqs: {total_seqs}, requested: {max_seqs}')
    if total_seqs < max_seqs + 2000:
        # Not enough: use 90% train, 10% val (min 500 val seqs)
        val_n = max(500, min(2000, total_seqs // 10))
        train_n = total_seqs - val_n
        print(f'  Adjusted: train {train_n}, val {val_n}')
    else:
        train_n = max_seqs
        val_n = 2000
    seqs = [flat[i*SEQ_LEN:(i+1)*SEQ_LEN] for i in range(train_n + val_n)]
    train = torch.tensor(seqs[:train_n], dtype=torch.long)
    val = torch.tensor(seqs[train_n:train_n + val_n], dtype=torch.long)
    del flat, seqs, all_tokens, texts
    return train, val

def make_model(model_type, seed=42):
    torch.manual_seed(seed)
    if model_type == 'gpt2m':
        from transformers import GPT2LMHeadModel, GPT2Config
        config = GPT2Config(vocab_size=50257, n_positions=512, n_embd=1024, n_layer=24, n_head=16)
        model = GPT2LMHeadModel(config)
    elif model_type == 'gpt2l':
        from transformers import GPT2LMHeadModel, GPT2Config
        config = GPT2Config(vocab_size=50257, n_positions=512, n_embd=1280, n_layer=36, n_head=20)
        model = GPT2LMHeadModel(config)
    elif model_type == 'llama1b':
        from transformers import LlamaForCausalLM, LlamaConfig
        config = LlamaConfig(
            vocab_size=50257, hidden_size=2048, intermediate_size=5632,
            num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16,
            max_position_embeddings=512, tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config)
    elif model_type == 'qwen3next':
        from transformers import Qwen3NextForCausalLM, Qwen3NextConfig
        config = Qwen3NextConfig(
            hidden_size=2048, num_hidden_layers=24, intermediate_size=5632,
            vocab_size=50257, max_position_embeddings=512, tie_word_embeddings=False,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128,
            linear_num_key_heads=8, linear_num_value_heads=8,
            linear_key_head_dim=128, linear_value_head_dim=256,
            linear_conv_kernel_dim=4, full_attention_interval=4,
            num_experts=1, num_experts_per_tok=1,
            moe_intermediate_size=5632, shared_expert_intermediate_size=0,
        )
        model = Qwen3NextForCausalLM(config)
    elif model_type == 'mamba1b':
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.models.config_mamba import MambaConfig
        config = MambaConfig(
            d_model=2048, n_layer=48, vocab_size=50257,
            ssm_cfg={'layer': 'Mamba2'},
        )
        model = MambaLMHeadModel(config)
    else:
        raise ValueError(f'Unknown model: {model_type}')
    model = model.to(torch.bfloat16).to(device)
    p = sum(pp.numel() for pp in model.parameters())
    print(f'  Model: {p:,} ({p/1e6:.0f}M)')
    return model

def forward_and_loss(model, input_ids, model_type):
    """Unified forward to handle Mamba's different API."""
    if model_type == 'mamba1b':
        out = model(input_ids)
        logits = out.logits
        labels = input_ids
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1))
        return type('Out', (), {'loss': loss})()
    return model(input_ids=input_ids, labels=input_ids)

@torch.no_grad()
def evaluate(model, val_data, max_docs=500, model_type='llama1b'):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(min(len(val_data), max_docs)):
        ids = val_data[i:i+1].to(device)
        loss = forward_and_loss(model, ids, model_type).loss.item()
        total_loss += loss * ids.numel(); total_tok += ids.numel()
    model.train()
    return math.exp(total_loss / total_tok)

def get_layer_idx(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p in ('layers', 'h') and i+1 < len(parts):
            try: return int(parts[i+1])
            except: pass
    return None

def get_num_layers(model):
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return len(model.model.layers)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        return len(model.backbone.layers)
    return 0

class HybridSGDOptimizer:
    def __init__(self, model, split_layer, lr_front=3e-4, lr_back=1e-3,
                 k=1, pert_eps=1e-3, model_type='llama1b',
                 split_side='front', n_layers=None, frozen=False,
                 split_start=-1):
        """split_side='front'  -> BP on input-side layers [0, split_layer).
           split_side='last'   -> BP on output-side layers [n_layers-split_layer, n_layers).
           split_side='middle' -> BP on center layers [(n-s)/2, (n-s)/2+s).
           If split_start >= 0, overrides: BP on [split_start, split_start+split_layer)."""
        self.model = model
        self.model_type = model_type
        self.k = k
        self.pert_eps = pert_eps
        self.lr_back = lr_back
        self.frozen = frozen  # if True, skip ZO step entirely (back params never updated)
        self.front_params, self.back_params = [], []
        # Determine BP range
        if split_start >= 0:
            bp_start, bp_end = split_start, split_start + split_layer
        elif split_side == 'front':
            bp_start, bp_end = 0, split_layer
        elif split_side == 'last':
            bp_start, bp_end = n_layers - split_layer, n_layers
        elif split_side == 'middle':
            bp_start = (n_layers - split_layer) // 2
            bp_end = bp_start + split_layer
        else:
            raise ValueError(f'Unknown split_side: {split_side}')
        print(f'  BP range: layers [{bp_start}, {bp_end}) out of {n_layers}')
        for name, param in model.named_parameters():
            idx = get_layer_idx(name)
            is_back = (idx is not None and (idx < bp_start or idx >= bp_end))
            if is_back:
                self.back_params.append((name, param))
            else:
                self.front_params.append((name, param))
        fp = sum(p.numel() for _, p in self.front_params)
        bp = sum(p.numel() for _, p in self.back_params)
        print(f'  Front (BP+AdamW): {fp:,} ({fp/1e6:.0f}M)')
        print(f'  Back (ZO+SGD):    {bp:,} ({bp/1e6:.0f}M)')
        self.front_opt = torch.optim.AdamW(
            [p for _, p in self.front_params], lr=lr_front,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    @torch.no_grad()
    def _zo_sgd_step(self, input_ids):
        deltas = {}
        for n, p in self.back_params:
            d = torch.randn_like(p); d.div_(d.norm()+1e-8); deltas[n] = d
        for n, p in self.back_params: p.data.add_(self.pert_eps * deltas[n])
        lp = forward_and_loss(self.model, input_ids, getattr(self, 'model_type', 'llama1b')).loss.item()
        for n, p in self.back_params: p.data.sub_(2*self.pert_eps * deltas[n])
        lm = forward_and_loss(self.model, input_ids, getattr(self, 'model_type', 'llama1b')).loss.item()
        for n, p in self.back_params: p.data.add_(self.pert_eps * deltas[n])
        dd = (lp-lm)/(2*self.pert_eps)
        for n, p in self.back_params:
            p.data.sub_(self.lr_back * dd * deltas[n])
        del deltas

    def step(self, input_ids):
        if not self.frozen:
            self._zo_sgd_step(input_ids)
        for _, p in self.back_params: p.requires_grad_(False)
        self.front_opt.zero_grad()
        loss = forward_and_loss(self.model, input_ids, getattr(self, 'model_type', 'llama1b')).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in self.front_params], 1.0)
        self.front_opt.step()
        for _, p in self.back_params: p.requires_grad_(True)
        return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--model', default='llama1b',
                        choices=['gpt2m', 'gpt2l', 'llama1b', 'qwen3next', 'mamba1b'])
    parser.add_argument('--data', default='wikitext',
                        choices=['wikitext', 'c4', 'fineweb-edu'])
    parser.add_argument('--mode', default='hybrid', choices=['backprop', 'hybrid', 'frozen'])
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr_front', type=float, default=3e-4)
    parser.add_argument('--lr_back', type=float, default=1e-3)
    parser.add_argument('--data_tokens', type=int, default=80_000_000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_side', default='front', choices=['front', 'last', 'middle'])
    parser.add_argument('--split_start', type=int, default=-1,
                        help='Override BP start layer (for position sweep). -1 = auto from split_side.')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='Save model checkpoint at end of training')
    parser.add_argument('--grad_analysis', action='store_true',
                        help='Measure per-layer gradient norm every eval_every steps (backprop mode only)')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Linear LR warmup steps')
    parser.add_argument('--lr_schedule', default='constant', choices=['constant', 'cosine'],
                        help='LR schedule after warmup')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze embeddings and LM head (only train transformer layers)')
    parser.add_argument('--galore_rank', type=int, default=0,
                        help='GaLore rank (0=disabled). Uses GaLoreAdamW for memory-efficient training.')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
    print(f'HF cache: {os.environ.get("HF_HOME")}')
    print(f'Loading {args.data}...')
    train_data, val_data = load_data(dataset=args.data, max_tokens=args.data_tokens)
    print(f'Data: train {train_data.shape} ({train_data.numel()/1e6:.0f}M)')

    model = make_model(args.model, seed=args.seed)
    n_layers = get_num_layers(model)
    split = n_layers // 4  # 25% front
    print(f'\n=== {args.name} === (model={args.model}, data={args.data}, mode={args.mode}, split_side={args.split_side}, split={split}/{n_layers}, seed={args.seed})')

    # Optionally freeze embeddings and LM head
    if args.freeze_embed:
        for name, param in model.named_parameters():
            if get_layer_idx(name) is None:  # embed, head, final norm
                param.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        frozen_e = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f'  Embed/head frozen: {frozen_e/1e6:.0f}M params')
    else:
        trainable = list(model.parameters())

    if args.mode == 'backprop':
        if args.galore_rank > 0:
            from galore_torch import GaLoreAdamW
            galore_params = []
            regular_params = []
            for p in trainable:
                if p.dim() >= 2:
                    galore_params.append(p)
                else:
                    regular_params.append(p)
            print(f'  GaLore: {len(galore_params)} weight matrices, rank={args.galore_rank}')
            print(f'  Regular: {len(regular_params)} bias/norm params')
            opt = GaLoreAdamW(
                [{'params': regular_params},
                 {'params': galore_params, 'rank': args.galore_rank,
                  'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}],
                lr=args.lr_front, weight_decay=0.01)
        else:
            opt = torch.optim.AdamW(trainable, lr=args.lr_front, betas=(0.9,0.999), weight_decay=0.01)
    else:
        opt = HybridSGDOptimizer(model, split_layer=split, lr_front=args.lr_front,
                                  lr_back=args.lr_back, k=1, model_type=args.model,
                                  split_side=args.split_side, n_layers=n_layers,
                                  frozen=(args.mode == 'frozen'),
                                  split_start=args.split_start)

    # LR schedule
    def get_lr(step):
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return args.lr_front * step / args.warmup_steps
        if args.lr_schedule == 'cosine':
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            return args.lr_front * 0.5 * (1 + math.cos(math.pi * progress))
        return args.lr_front

    idx = 0
    log = {'name': args.name, 'model': args.model, 'data': args.data, 'mode': args.mode,
           'split': split, 'n_layers': n_layers, 'seed': args.seed, 'split_side': args.split_side,
           'lr_front': args.lr_front, 'warmup_steps': args.warmup_steps,
           'lr_schedule': args.lr_schedule, 'freeze_embed': args.freeze_embed,
           'galore_rank': args.galore_rank,
           'eval_steps': [], 'losses': [], 'val_ppls': [], 'times': []}
    t0 = time.time()

    for step in range(args.steps):
        # Update LR
        current_lr = get_lr(step)
        if args.mode == 'backprop':
            for pg in opt.param_groups:
                pg['lr'] = current_lr

        if idx + args.batch_size > len(train_data): idx = 0
        ids = train_data[idx:idx+args.batch_size].to(device); idx += args.batch_size
        if args.mode == 'backprop':
            opt.zero_grad()
            loss = forward_and_loss(model, ids, args.model).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            loss_val = loss.item()
        else:
            loss_val = opt.step(ids)
        if step % 100 == 0: log['losses'].append(loss_val)
        # Gradient analysis: measure per-layer grad norm at eval steps
        if args.grad_analysis and args.mode == 'backprop' and step % args.eval_every == 0 and step > 0:
            layer_gnorms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    lidx = get_layer_idx(name)
                    key = f'layer_{lidx}' if lidx is not None else 'other'
                    if key not in layer_gnorms:
                        layer_gnorms[key] = []
                    layer_gnorms[key].append(param.grad.norm().item())
            avg_gnorms = {k: sum(v)/len(v) for k, v in layer_gnorms.items()}
            if 'grad_norms' not in log:
                log['grad_norms'] = {}
            log['grad_norms'][str(step)] = avg_gnorms

        if step % args.eval_every == 0:
            ppl = evaluate(model, val_data, model_type=args.model)
            elapsed = time.time() - t0
            log['eval_steps'].append(step)
            log['val_ppls'].append(ppl)
            log['times'].append(elapsed)
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f'  step {step:5d}/{args.steps} | loss {loss_val:.4f} | '
                  f'ppl {ppl:.2f} | {elapsed/60:.1f}min | vram {vram:.1f}GB', flush=True)

    ppl = evaluate(model, val_data, model_type=args.model)
    log['final_ppl'] = ppl
    log['total_time'] = time.time() - t0
    log['peak_vram'] = torch.cuda.max_memory_allocated() / 1e9
    print(f'\n  FINAL: ppl={ppl:.2f}, time={log["total_time"]/60:.1f}min, vram={log["peak_vram"]:.1f}GB')
    fname = os.path.join(SAVE_DIR, f'gate2_{args.name}.json')
    json.dump(log, open(fname, 'w'), indent=2)
    print(f'  Saved: {fname}')

    if args.save_ckpt:
        ckpt_dir = os.path.join(BASE_DIR, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'{args.name}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f'  Checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path)/1e9:.2f}GB)')

if __name__ == '__main__':
    main()
