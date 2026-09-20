"""Experiment: a conv stem whose receptive field spans the board, same 8x10 token layout.

Three residual 3x3 convolutions with dilation 2, 4 and 8 on the stride-4 feature map
widen each token's view from about 21 px to about 133 px (the board is 120 px), so a
single attention read can see relations across the board. `widen(model)` swaps it
into a DoomScorerV2 in place; checkpoints record it as payload["stem"] = "wide".
"""

from __future__ import annotations

from torch import nn


class DilatedContext(nn.Module):
    def __init__(self, width: int, dilations=(2, 4, 8)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d), nn.GroupNorm(4, width), nn.SiLU())
            for d in dilations)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


def widen(model):
    model.stem = nn.Sequential(*model.stem, DilatedContext(model.width))
    return model
