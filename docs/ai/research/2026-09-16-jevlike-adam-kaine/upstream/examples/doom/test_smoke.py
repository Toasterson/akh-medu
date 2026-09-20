"""Small checks for the shared visual scorer and seven-button controller."""

import numpy as np
import torch

from environment import DOOM_ACTIONS
from jevlike.vision import (
    CHESS_OPTION_IDS,
    TOTAL_OPTIONS,
    DoomScorerV2,
    observation,
    observation_tensor,
)


def test_visual_scorer_preserves_each_option_axis() -> None:
    model = DoomScorerV2(actions=TOTAL_OPTIONS)
    current = np.full((120, 160, 3), 90, dtype=np.uint8)
    previous = np.full((120, 160, 3), 80, dtype=np.uint8)
    item = observation(current, previous)
    logits, value, trace = model.forward_trace(
        observation_tensor([item], torch.device("cpu")), torch.arange(7)
    )
    assert item.shape == (120, 160, 4)
    assert logits.shape == (1, len(DOOM_ACTIONS))
    assert value.shape == (1,)
    assert trace["attention_map"].shape == (1, len(DOOM_ACTIONS), 80)
    assert torch.allclose(trace["probabilities"].sum(-1), torch.ones(1))


def test_one_table_selects_the_five_chess_keys() -> None:
    model = DoomScorerV2(actions=TOTAL_OPTIONS)
    item = np.zeros((120, 160, 4), dtype=np.uint8)
    item[..., 3] = 128
    logits, value = model(
        observation_tensor([item], torch.device("cpu")),
        torch.tensor(CHESS_OPTION_IDS),
    )
    assert model.options.num_embeddings == 12
    assert logits.shape == (1, 5)
    assert value.shape == (1,)
