#!/usr/bin/env python3
"""Process-local MLX hook recording initial weights and actual training cost."""
import hashlib
import json
import resource
import sys
import time
from pathlib import Path


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import lora
    config_path=Path(sys.argv[sys.argv.index('--config')+1]);config=json.loads(config_path.read_text());root=config_path.parent
    original_train=lora.train
    def audited_train(*args,**kwargs):
        model=kwargs['model'];initial=root/'initial-adapter.safetensors'
        tensors=dict(sorted(tree_flatten(model.trainable_parameters())));mx.eval(tensors)
        mx.save_safetensors(str(initial),tensors)
        checksum=hashlib.sha256(initial.read_bytes()).hexdigest()
        receipt=dict(path=str(initial),sha256=checksum,bytes=initial.stat().st_size,seed=config['seed'],
                     trainable_parameters=sum(t.size for t in tensors.values()),device=mx.device_info())
        (root/'initialization.json').write_text(json.dumps(receipt,indent=2)+'\n')
        expected=config.get('expected_initialization_sha256')
        if expected and checksum!=expected:raise ValueError('Initial adapter differs from frozen comparison')
        if config.get('paired_row_order'):
            from paired_batches import make_paired_batches
            kwargs['iterate_batches']=make_paired_batches(config['seed'],root/'training-row-order.jsonl')
        started=time.monotonic();completed=False
        try:
            value=original_train(*args,**kwargs);completed=True;return value
        finally:
            result=dict(completed=completed,elapsed_seconds=time.monotonic()-started,peak_mlx_bytes=mx.get_peak_memory(),
                        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        initialization_sha256=checksum,iterations=config['iters'])
            (root/'training-cost.json').write_text(json.dumps(result,indent=2)+'\n')
    lora.train=audited_train
    lora.main()


if __name__=='__main__':main()
