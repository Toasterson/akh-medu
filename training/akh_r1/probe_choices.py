#!/usr/bin/env python3
"""Teacher-context diagnostic: generated choices and token-level loss, no tools executed."""
import argparse
import json
from pathlib import Path

from build_curriculum import digest, now
from choice_loss import choice_span
from protocol import parse_action


def select_rows(rows, episodes, all_decisions=False, limit=32):
    if not 1 <= limit <= 64:
        raise ValueError('Diagnostic limit must be between 1 and 64 decisions')
    chosen = []
    for episode in episodes:
        candidates = [row for row in rows if row['episode_id'] == episode]
        if not candidates:
            raise ValueError(f'Missing requested episode: {episode}')
        if all_decisions:
            chosen.extend(candidates)
        else:
            selected = set()
            for row in candidates:
                tool = parse_action(row['messages'][-1]['content'], 'native')['tool']
                if tool in {'report', 'remember', 'kg_query', 'terminal'} and tool not in selected:
                    selected.add(tool)
                    chosen.append(row)
            if len(selected) != 4:
                raise ValueError('Each selected episode must contain all four probe tools')
    if len(chosen) > limit:
        raise ValueError('Requested episodes exceed the decision budget; refusing silent truncation')
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', default='named_flags_tsv-71-acquire,named_flags_tsv-71-cold_graph')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='train')
    parser.add_argument('--all-decisions', action='store_true',
                        help='Probe every requested decision, including binding/action/verification; no tools execute')
    parser.add_argument('--max-decisions', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='0 selects greedy decoding; default preserves existing probes')
    args = parser.parse_args()
    if not 0 <= args.temperature <= 1:
        parser.error('Diagnostic temperature must be between 0 and 1')
    args.out.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tuner.datasets import ChatDataset
    from transformers import AutoTokenizer

    corpus_file = args.corpus / (args.split + '.jsonl')
    rows = [json.loads(line) for line in corpus_file.read_text().splitlines()]
    chosen = select_rows(rows, args.episodes.split(','), args.all_decisions, args.max_decisions)
    files = [p for p in args.model.iterdir() if p.is_file()]
    files += [corpus_file, Path(__file__), Path(__file__).with_name('choice_loss.py')]
    if args.adapter is not None:
        files += [args.adapter / 'adapters.safetensors', args.adapter / 'adapter_config.json']
    manifest = dict(started_at=now(), args={k:str(v) for k,v in vars(args).items()},
                    files={str(p):digest(p) for p in files},
                    selection='all requested decisions' if args.all_decisions else 'first four action types per requested episode',
                    corpus_split=args.split, decision_budget=args.max_decisions,
                    max_tokens=256, seed=83, temperature=args.temperature, top_p=0.8, top_k=20,
                    teacher_context=True, actual_tools_executed=False, release_certification=False)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    model, tokenizer = load(str(args.model), adapter_path=str(args.adapter) if args.adapter else None)
    raw_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    results = []
    with (args.out / 'trace.jsonl').open('w') as log:
        for row in chosen:
            tokens, offset = ChatDataset([row], tokenizer, mask_prompt=True).process(row)
            raw_tokens, raw_offset = ChatDataset([row], raw_tokenizer, mask_prompt=True).process(row)
            prompt = tokenizer.apply_chat_template(row['messages'][:-1], tools=row['tools'],
                        tokenize=True, add_generation_prompt=True, enable_thinking=False)
            mx.random.seed(83)
            generated = generate(model, tokenizer, prompt=prompt, max_tokens=256,
                                  sampler=make_sampler(temp=args.temperature, top_p=0.8, top_k=20))
            expected = parse_action(row['messages'][-1]['content'], 'native')
            try:
                actual = parse_action(generated, 'native')
            except (ValueError, TypeError) as error:
                actual = dict(parse_error=str(error))
            start, end = choice_span(tokenizer, tokens, offset, len(tokens))
            logits = model(mx.array([tokens[:-1]]))[0]
            nll = nn.losses.cross_entropy(logits, mx.array(tokens[1:]), reduction='none').astype(mx.float32)
            mx.eval(nll)
            record = dict(episode_id=row['episode_id'], decision_id=row['decision_id'], expected=expected,
                          actual=actual, raw=generated, prompt_tokens=len(prompt),
                          wrapper_raw_tokenizer_equal=tokens==raw_tokens and offset==raw_offset,
                          inference_prefix_matches_training=tokens[:len(prompt)]==prompt,
                          tool_match=actual.get('tool')==expected['tool'],
                          arguments_exact_match=actual.get('arguments')==expected['arguments'],
                          target_nll_mean=float(nll[offset-1:].mean().item()),
                          tool_name_nll_mean=float(nll[start-1:end-1].mean().item()))
            results.append(record)
            log.write(json.dumps(record, ensure_ascii=False)+'\n'); log.flush()
            print(json.dumps({k:record[k] for k in ['episode_id','decision_id','tool_match','arguments_exact_match','tool_name_nll_mean']}), flush=True)
    manifest.update(finished_at=now(), files_unchanged=all(digest(Path(p))==h for p,h in manifest['files'].items()))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    summary = dict(decisions=len(results), tool_matches=sum(r['tool_match'] for r in results),
                   argument_matches=sum(r['arguments_exact_match'] for r in results),
                   all_tokenization_checks_pass=all(r['wrapper_raw_tokenizer_equal'] and r['inference_prefix_matches_training'] for r in results),
                   certification=False, teacher_context=True, actual_tools_executed=False)
    (args.out / 'results.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
