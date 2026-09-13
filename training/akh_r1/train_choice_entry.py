#!/usr/bin/env python3
"""Local entrypoint using MLX trainer's loss/iterator hooks; no installed edits."""
import json
import sys
from pathlib import Path

from choice_loss import balanced_loss, make_choice_batches


def main():
    from mlx_lm import lora
    from mlx_lm.tuner.trainer import iterate_batches
    from transformers import AutoTokenizer

    config_path = Path(sys.argv[sys.argv.index('--config')+1])
    config = json.loads(config_path.read_text())
    if not config.get('mask_prompt'):
        raise ValueError('Choice-weighted pilot requires prompt masking')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    original_train = lora.train
    batches = make_choice_batches(tokenizer, iterate_batches)

    def train_with_choice_loss(*args, **kwargs):
        return original_train(*args, **kwargs, loss=balanced_loss, iterate_batches=batches)

    # Process-local wiring to the trainer's supplied hooks; package files stay unchanged.
    lora.train = train_with_choice_loss
    print('Objective: 0.5 mean function-name loss + 0.5 mean remaining target loss.', flush=True)
    lora.main()


if __name__ == '__main__':
    main()
