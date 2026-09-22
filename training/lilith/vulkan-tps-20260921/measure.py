"""Counterbalanced local decode/cache experiment; run with an idle GPU, no builds."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

out = Path(__file__).resolve().parent
cmd = json.loads((out / 'command.json').read_text())
env = dict(os.environ, AKH_GPU_MATMUL='0', AKH_GPU_QUANT='0',
           VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json')
rows = []
texts = []
for label, entries in [('a0', 2048), ('b0', 8192), ('b1', 8192), ('a1', 2048)]:
    run_cmd = cmd.copy()
    run_cmd[run_cmd.index('--hidden-dump') + 1] = str(out / (label + '.f32'))
    path = out / (label + '.log')
    with path.open('x') as log:
        subprocess.run(['/usr/bin/time', '-f', '%M', '-o', str(out / (label + '.rss'))] + run_cmd, env=dict(env, AKH_VULKAN_PIPELINE_CACHE_ENTRIES=str(entries)),
                       stdout=log, stderr=subprocess.STDOUT, timeout=240, check=True)
    events = [json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')]
    generations = [e for e in events if e['phase'] == 'generation']
    assert len(generations) == 2 and all(e['tokens'] == 128 for e in generations)
    texts.extend(e['text'] for e in generations)
    row = dict(label=label, entries=entries,
               tps=[e['decode_steps_per_second'] for e in generations],
               peak_rss_kib=int((out / (label + '.rss')).read_text().strip()),
               device_memory=[e for e in events if e['phase'] == 'memory_after_generation'],
               cache=[e for e in events if e['phase'] == 'vulkan_pipeline_cache'],
               hidden_sha256=hashlib.sha256((out / (label + '.f32')).read_bytes()).hexdigest())
    rows.append(row)
    print(json.dumps(row), flush=True)
    (out / 'measurements.json').write_text(json.dumps(rows, indent=2) + '\n')
a = statistics.mean(r['tps'][1] for r in rows if r['entries'] == 2048)
b = statistics.mean(r['tps'][1] for r in rows if r['entries'] == 8192)
result = dict(reference_warm_tps=a, candidate_warm_tps=b, speedup=b/a,
              exact_text=len(set(texts)) == 1,
              exact_hidden=len({r['hidden_sha256'] for r in rows}) == 1,
              target_relative_improvement=0.10)
result['passed'] = result['exact_text'] and result['exact_hidden'] and b >= a * 1.10
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['passed'] else 1)
