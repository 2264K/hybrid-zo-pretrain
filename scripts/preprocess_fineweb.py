"""Preprocess FineWeb-Edu into flat binary token file for 10B training.

Usage:
  python scripts/preprocess_fineweb.py --output data/fineweb_edu_10b.bin --max_tokens 10_000_000_000

Output:
  data/fineweb_edu_10b.bin      — flat uint16 token IDs (train)
  data/fineweb_edu_10b_val.bin  — flat uint16 token IDs (val, last 2M tokens)
  data/fineweb_edu_10b_meta.json — metadata (token count, vocab, etc.)
"""
import os, json, time, argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/fineweb_edu_10b.bin')
    parser.add_argument('--max_tokens', type=int, default=10_000_000_000)
    parser.add_argument('--val_tokens', type=int, default=2_000_000)
    parser.add_argument('--chunk_size', type=int, default=100_000,
                        help='Documents per tokenization batch')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print(f'Tokenizer: gpt2, vocab_size={tokenizer.vocab_size}')
    assert tokenizer.vocab_size <= 65535, 'vocab too large for uint16'

    ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT',
                      split='train', streaming=True)

    total_target = args.max_tokens + args.val_tokens
    token_count = 0
    doc_count = 0
    t0 = time.time()

    tmp_path = args.output + '.tmp'
    with open(tmp_path, 'wb') as f:
        buffer = []
        for item in ds:
            text = item.get('text', '')
            if not text.strip():
                continue
            tokens = tokenizer.encode(text)
            buffer.extend(tokens)
            doc_count += 1

            # Flush in chunks for efficiency
            if len(buffer) >= 1_000_000:
                arr = np.array(buffer, dtype=np.uint16)
                f.write(arr.tobytes())
                token_count += len(buffer)
                buffer = []

                elapsed = time.time() - t0
                rate = token_count / elapsed
                eta = (total_target - token_count) / rate if rate > 0 else 0
                print(f'  {token_count/1e9:.2f}B tokens, {doc_count} docs, '
                      f'{rate/1e6:.1f}M tok/s, ETA {eta/60:.0f}min', flush=True)

                if token_count >= total_target:
                    break

        # Flush remaining
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            f.write(arr.tobytes())
            token_count += len(buffer)

    elapsed = time.time() - t0
    print(f'\nTotal: {token_count/1e9:.3f}B tokens from {doc_count} docs in {elapsed/60:.1f}min')

    # Split train/val
    print('Splitting train/val...')
    all_tokens = np.memmap(tmp_path, dtype=np.uint16, mode='r')
    actual_total = len(all_tokens)
    val_n = min(args.val_tokens, actual_total // 10)
    train_n = actual_total - val_n

    # Write train
    train_tokens = np.memmap(args.output, dtype=np.uint16, mode='w+', shape=(train_n,))
    train_tokens[:] = all_tokens[:train_n]
    train_tokens.flush()

    # Write val
    val_path = args.output.replace('.bin', '_val.bin')
    val_tokens = np.memmap(val_path, dtype=np.uint16, mode='w+', shape=(val_n,))
    val_tokens[:] = all_tokens[train_n:train_n + val_n]
    val_tokens.flush()

    # Clean up tmp
    os.remove(tmp_path)

    # Metadata
    meta = {
        'train_tokens': int(train_n),
        'val_tokens': int(val_n),
        'total_tokens': int(actual_total),
        'vocab_size': tokenizer.vocab_size,
        'dtype': 'uint16',
        'tokenizer': 'gpt2',
        'dataset': 'HuggingFaceFW/fineweb-edu/sample-10BT',
        'preprocess_time_min': elapsed / 60,
    }
    meta_path = args.output.replace('.bin', '_meta.json')
    json.dump(meta, open(meta_path, 'w'), indent=2)
    print(f'Saved: {args.output} ({train_n/1e9:.3f}B train), {val_path} ({val_n/1e6:.1f}M val)')
    print(f'Meta: {meta_path}')


if __name__ == '__main__':
    main()
