"""Smoke test: renderer, controller, key paths, batches, and one model forward. Run from chess/."""

import random
import chess
import numpy as np
import torch

from jevlike.vision import TOTAL_OPTIONS, DoomScorerV2, observation_tensor  # noqa: E402

import data  # noqa: E402
from keys import Controller  # noqa: E402
from render import observation  # noqa: E402


def main() -> None:
    # Every teacher walk, replayed through the real controller from a random cursor, plays a good move.
    rng = random.Random(0)
    try:
        rows = data.load_positions("heldout")
    except FileNotFoundError:  # no generated positions yet: use the starting position
        rows = [(chess.STARTING_FEN, None, None, ["e2e4", "g1f3"], False)] * 300
    for fen, _, _, good, _ in rng.sample(rows, 300):
        board = chess.Board(fen)
        controller = Controller(board, cursor=(rng.randrange(8), rng.randrange(8)))
        states = data.walk(chess.Board(fen), good, controller.cursor)
        events = [controller.press(key) for _, _, key in states]
        assert events[-1][0] == "put" and events[-1][1].uci()[:4] in [m[:4] for m in good], (fen, good, events)
        assert all(e[0] in ("moved", "lift") for e in events[:-1]), (fen, good, events)

    frames = np.stack([observation(chess.Board(), (r, c)) for r in range(4) for c in range(4)])
    ids = data.CHESS_OPTION_IDS
    frame = observation(chess.Board(), (7, 4))
    assert frame.shape == (120, 160, 4) and frame[:, :20, :3].max() == 0 and (frame[..., 3] == 128).all()

    model = DoomScorerV2(actions=TOTAL_OPTIONS)
    logits, _ = model(observation_tensor(frames, torch.device("cpu")), ids)
    assert logits.shape == (16, 5) and torch.isfinite(logits).all()
    doom_logits, _ = model(torch.rand(2, 4, 120, 160), torch.arange(7))
    assert doom_logits.shape == (2, 7)
    print("ok")


if __name__ == "__main__":
    main()
