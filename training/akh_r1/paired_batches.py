"""Local batch-one iterator with format-independent order and no truncation."""
import json
from pathlib import Path


def make_paired_batches(seed,order_path):
    import mlx.core as mx
    import numpy as np
    def iterate(dataset,batch_size,max_seq_length,loop=False,seed=None,comm_group=None):
        if batch_size!=1 or (comm_group is not None and comm_group.size()!=1):raise ValueError('Paired iterator requires one local batch-one worker')
        rng=np.random.RandomState(default_seed if seed is None else seed);epoch=0
        while True:
            for index in rng.permutation(len(dataset)):
                row=dataset[int(index)]
                tokens,offset=row if len(row)==2 else (row,0)
                length=len(tokens)
                if length>max_seq_length:raise ValueError('Refusing to truncate a paired teaching row')
                padded=min(1+32*((length+31)//32),max_seq_length)
                batch=np.zeros((1,padded),dtype=np.int32);batch[0,:length]=tokens
                if loop:
                    with Path(order_path).open('a') as log:log.write(json.dumps(dict(epoch=epoch,index=int(index),tokens=length,offset=int(offset)))+'\n')
                yield mx.array(batch),mx.array([[offset,length]])
            if not loop:break
            epoch+=1
    default_seed=seed
    return iterate
