"""Downstream eval with lm-evaluation-harness for 10B checkpoints."""
import os
# Use existing HF_HOME if set (Alpine), else default to local WSL cache
os.environ.setdefault('HF_HOME', '/mnt/d/tmp/hf_cache')
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch, json, argparse
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

# Workaround for lm_eval + newer transformers compatibility
import transformers
if not hasattr(transformers, 'AutoModelForVision2Seq'):
    transformers.AutoModelForVision2Seq = None

import lm_eval
from lm_eval.models.huggingface import HFLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--tasks', default='hellaswag,piqa,lambada_openai')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda'
    config = LlamaConfig(
        vocab_size=50257, hidden_size=2048, intermediate_size=5632,
        num_hidden_layers=24, num_attention_heads=16, num_key_value_heads=16,
        max_position_embeddings=512, tie_word_embeddings=False,
    )
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(f'Loading checkpoint: {args.checkpoint}')
    model = LlamaForCausalLM(config).to(torch.bfloat16).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    del state
    print(f'Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB')

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    tasks = args.tasks.split(',')
    print(f'Running eval: {tasks}')
    out = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0)

    print(f'\n=== Results: {args.run_name} ===')
    results = {}
    for task, res in out['results'].items():
        acc = res.get('acc,none', res.get('acc_norm,none', 'N/A'))
        acc_norm = res.get('acc_norm,none', 'N/A')
        print(f'  {task}: acc={acc}, acc_norm={acc_norm}')
        results[task] = {'acc': acc, 'acc_norm': acc_norm}

    save_path = f'results/10b/{args.run_name}_eval.json'
    json.dump({'run_name': args.run_name, 'checkpoint': args.checkpoint,
               'results': results}, open(save_path, 'w'), indent=2)
    print(f'\nSaved: {save_path}')

if __name__ == '__main__':
    main()
