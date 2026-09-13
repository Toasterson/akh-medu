#!/usr/bin/env python3
"""Verify actual MLX chat masks and context lengths before any optimizer run."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from build_curriculum import digest, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--max-seq-length', type=int, default=4096)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    from mlx_lm.tuner.datasets import ChatDataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = dict(created_at=now(), max_seq_length=args.max_seq_length, mask_prompt=True,
                  packages={p:importlib.metadata.version(p) for p in ['mlx-lm', 'transformers']},
                  corpus_files={p.name:digest(p) for p in args.corpus.glob('*.json*')},
                  tokenizer_files={p.name:digest(p) for p in args.tokenizer.iterdir()
                                   if p.is_file() and ('token' in p.name or p.suffix in {'.jinja','.txt'})},
                  splits={}, failures=[], training_started=False, release_certification=False)
    for split in ['train', 'valid', 'test']:
        rows = [json.loads(line) for line in (args.corpus / f'{split}.jsonl').read_text().splitlines()]
        dataset = ChatDataset(rows, tokenizer, mask_prompt=True)
        lengths, targets = [], []
        for row in rows:
            tokens, offset = dataset.process(row)
            prompt = tokenizer.apply_chat_template(row['messages'][:-1], tools=row['tools'],
                          add_generation_prompt=True, return_dict=False)
            inference_prompt = tokenizer.apply_chat_template(row['messages'][:-1], tools=row['tools'],
                          add_generation_prompt=True, enable_thinking=False, return_dict=False)
            target = tokenizer.decode(tokens[offset:])
            checks = dict(only_prior_context=tokens[:offset] == prompt,
                          nonthinking_inference_prefix=tokens[:len(inference_prompt)] == inference_prompt,
                          target_in_loss=row['messages'][-1]['content'] in target,
                          loss_nonempty=0 < offset < len(tokens),
                          no_truncation=len(tokens) <= args.max_seq_length)
            if not all(checks.values()):
                result['failures'].append(dict(episode=row['episode_id'], decision=row['decision_id'],
                                               tokens=len(tokens), offset=offset, checks=checks))
            lengths.append(len(tokens))
            targets.append(len(tokens)-offset)
        result['splits'][split] = dict(decisions=len(rows), max_tokens=max(lengths, default=0),
            min_tokens=min(lengths, default=0), target_tokens=sum(targets),
            exceeds_default_2048=sum(n > 2048 for n in lengths))
    result['passed'] = not result['failures']
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], splits=result['splits'], failures=len(result['failures'])), indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
