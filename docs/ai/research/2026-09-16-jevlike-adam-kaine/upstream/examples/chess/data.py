"""Supervised chess key batches from Stockfish-labelled positions.

The teacher is a key policy. For each position it knows the near-best Stockfish
moves (an optional `good` list per row; otherwise the single best move). With
nothing held, it walks the cursor (rows first, then columns) to the good move
whose piece is nearest, and lifts that piece. While holding, it walks to the
piece's nearest good destination and puts it down, or drops the piece back if
it has no good move. Walking toward the nearest target keeps that target the
nearest, so labels stay stable along the path.

Training frames come from two places, labelled the same way:
- every screen on a teacher walk from a random cursor square, with the teacher's next key
- every recovery state the model reached itself in play (runs/dagger)
"""

from __future__ import annotations

import bisect
import glob
import itertools
import json
import multiprocessing
import os
import random

import chess
import numpy as np
import torch

from keys import DOWN, LEFT, RIGHT, TOGGLE, UP, path_keys
from render import observation, screen_rc, square_at

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS = os.path.join(HERE, "runs", "positions*")
CHESS_OPTION_IDS = torch.arange(7, 12)
LATE_WEIGHT = float(os.environ.get("CHESS_LATE_WEIGHT", "1"))  # oversampling of late-game rows


def _is_late(fen: str, cp) -> bool:
    """Endgame-like (14 or fewer pieces) or a forced mate on the board."""
    return sum(ch.isalpha() for ch in fen.split()[0]) <= 14 or (cp is not None and abs(cp) >= 10000)


def _read(pattern: str):
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # a partially written last line
                    continue


def load_positions(split: str = "train") -> list[tuple]:
    """Rows (fen, cursor=None, holding=None, good moves, is_late)."""
    name = "heldout.jsonl" if split == "heldout" else "train-*.jsonl"
    rows = []
    for row in _read(os.path.join(POSITIONS, name)):
        good = [m for m in (row.get("good") or [row["move"]])
                if chess.Move.from_uci(m).promotion in (None, chess.QUEEN)]
        if good:
            rows.append((row["fen"], None, None, good, _is_late(row["fen"], row.get("cp"))))
    if not rows:
        raise FileNotFoundError(f"no {split} positions under {POSITIONS}")
    return rows


def load_dagger() -> list[tuple]:
    """Rows (fen, cursor, holding, good moves, is_late) the model itself reached in play."""
    rows = []
    for row in _read(os.path.join(HERE, "runs", "dagger", "*.jsonl")):
        good = row["good"] if "good" in row else ([row["move"]] if row["move"] else [])
        if good or row["holding"] is not None:
            rows.append((row["fen"], tuple(row["cursor"]), row["holding"], good, _is_late(row["fen"], None)))
    return rows


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def teacher_key(board: chess.Board, good: list[str], cursor: tuple[int, int], holding: int | None) -> int:
    flip = board.turn == chess.BLACK
    moves = [chess.Move.from_uci(m) for m in good]
    if holding is None:  # min keeps the first of equals, i.e. the better-ranked move
        target = min(moves, key=lambda m: _distance(cursor, screen_rc(m.from_square, flip))).from_square
    else:
        mine = [m for m in moves if m.from_square == holding]
        target = (min(mine, key=lambda m: _distance(cursor, screen_rc(m.to_square, flip))).to_square
                  if mine else holding)
    keys = path_keys(cursor, screen_rc(target, flip))
    return keys[0] if keys else TOGGLE


def walk(board: chess.Board, good: list[str], cursor: tuple[int, int]) -> list[tuple]:
    """(cursor, holding, key) for each state of the teacher's walk until the move is put down."""
    states, holding = [], None
    while len(states) < 40:
        key = teacher_key(board, good, cursor, holding)
        states.append((cursor, holding, key))
        if key == TOGGLE:
            if holding is not None:
                return states
            holding = square_at(*cursor, board.turn == chess.BLACK)
        else:
            cursor = (cursor[0] + (key == DOWN) - (key == UP), cursor[1] + (key == RIGHT) - (key == LEFT))
    raise AssertionError(f"teacher walk did not finish: {board.fen()} {good}")


def _cumulative(rows) -> list[float]:
    return list(itertools.accumulate(LATE_WEIGHT if row[4] else 1.0 for row in rows))


def sample(rows, cumulative, rng: random.Random):
    """One (observation, label, is_first_key, cursor, lifted piece's screen square) sample.

    A position row starts a teacher walk from a random cursor and takes a random state
    on it; a recovery row is the reached state itself.
    """
    fen, cursor, holding, good, _ = rows[bisect.bisect(cumulative, rng.random() * cumulative[-1])]
    board = chess.Board(fen)
    if cursor is not None:
        return observation(board, cursor, holding), teacher_key(board, good, cursor, holding), False, cursor, (0, 0)
    states = walk(board, good, (rng.randrange(8), rng.randrange(8)))
    step = rng.randrange(len(states))
    at, holding, key = states[step]
    lifted = next(h for _, h, _ in states if h is not None)
    return observation(board, at, holding), key, step == 0, at, screen_rc(lifted, board.turn == chess.BLACK)


_ROWS: dict[str, tuple] = {}


def _batch(task: tuple[str, int, int, int, float]):
    split, batch_size, seed, index, dagger = task
    if split not in _ROWS:
        rows = load_positions(split)
        recovery = load_dagger() if dagger else []
        _ROWS[split], _ROWS["dagger"] = (rows, _cumulative(rows)), (recovery, _cumulative(recovery))
    rng = random.Random(seed * 1_000_003 + index)
    recovered = round(batch_size * dagger) if _ROWS["dagger"][0] else 0
    items = [sample(*_ROWS["dagger"], rng) for _ in range(recovered)]
    items += [sample(*_ROWS[split], rng) for _ in range(batch_size - recovered)]
    return (np.stack([item[0] for item in items]), np.array([item[1] for item in items]),
            np.array([item[3][0] * 8 + item[3][1] for item in items]),
            np.array([item[4][0] * 8 + item[4][1] for item in items]))


def batches(batch_size: int, seed: int = 0, split: str = "train", workers: int = 6,
            extras: bool = False, dagger: float = 0.5):
    """Infinite (frames uint8 Bx120x160x4, option_ids [7..11], labels B) batches.

    Rendering is ~2 ms per sample in Python, so batches are drawn in worker processes;
    the sequence is deterministic in (seed, batch index). `dagger` is the fraction of
    each batch drawn from recovery states when runs/dagger has any.
    """
    load_positions(split)  # fail fast in the caller if there is no data
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for start in itertools.count(0, 4 * workers):
            tasks = [(split, batch_size, seed, i, dagger) for i in range(start, start + 4 * workers)]
            for frames, labels, cursor, origin in pool.imap(_batch, tasks):
                if extras:  # screen squares (row * 8 + col) of the cursor and the lifted piece
                    yield frames, CHESS_OPTION_IDS, torch.from_numpy(labels), cursor, origin
                else:
                    yield frames, CHESS_OPTION_IDS, torch.from_numpy(labels)


def fixed_set(n: int, seed: int = 1, split: str = "heldout", first_only: bool = False):
    """A fixed evaluation set (uniform over rows): frames, labels, is_first, cursor, lifted square."""
    rows = load_positions(split)
    uniform = list(range(1, len(rows) + 1))
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        item = sample(rows, uniform, rng)
        if item[2] or not first_only:
            items.append(item)
    return (np.stack([i[0] for i in items]), np.array([i[1] for i in items]),
            np.array([i[2] for i in items]), np.array([i[3][0] * 8 + i[3][1] for i in items]),
            np.array([i[4][0] * 8 + i[4][1] for i in items]))


if __name__ == "__main__":
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    keys = [k for _, _, k in walk(board, ["e7e5"], (0, 0))]
    assert keys == [DOWN] * 6 + [RIGHT] * 3 + [TOGGLE, UP, UP, TOGGLE], keys
    # black to move, board flipped: g8 is screen (7, 1), b8 is (7, 6); from (7, 0) the nearer knight is g8
    keys = [k for _, _, k in walk(board, ["b8c6", "g8f6"], (7, 0))]
    assert keys[:2] == [RIGHT, TOGGLE], keys
    assert teacher_key(board, ["b8c6"], (7, 6), chess.B8) == UP  # holding b8: walk up toward c6
    frames, ids, labels = next(batches(8, split=os.environ.get("SPLIT", "train"), workers=2))
    assert frames.shape == (8, 120, 160, 4) and frames.dtype == np.uint8 and ids.tolist() == [7, 8, 9, 10, 11]
    print("ok", labels.tolist())
