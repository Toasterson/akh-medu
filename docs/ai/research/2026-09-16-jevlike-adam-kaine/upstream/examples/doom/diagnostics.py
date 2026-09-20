"""Small, model-independent helpers for Doom state-dependence checks."""

from __future__ import annotations

import math

import numpy as np
import torch


ENEMY_NAMES = {
    "MarineChainsawVzd", "Zombieman", "ShotgunGuy", "ChaingunGuy",
    "DoomImp", "Demon", "Spectre", "Cacodemon", "LostSoul", "BaronOfHell",
    "HellKnight", "Revenant", "Mancubus", "Arachnotron", "PainElemental",
    "Archvile", "Cyberdemon", "SpiderMastermind",
}


def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(p.dtype).eps
    return (p * ((p + epsilon).log() - (q + epsilon).log())).sum(-1)


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def enemy_observation(state) -> dict[str, float] | None:
    enemies = [label for label in state.labels if label.object_name in ENEMY_NAMES]
    if not enemies:
        return None
    enemy = min(
        enemies,
        key=lambda label: math.hypot(label.object_position_x, label.object_position_y),
    )
    height, width = state.screen_buffer.shape[:2]
    centre = enemy.x + enemy.width / 2
    return {
        "x": float(2 * centre / width - 1),
        "area": float(enemy.width * enemy.height / (width * height)),
        "distance": float(math.hypot(enemy.object_position_x, enemy.object_position_y)),
    }
