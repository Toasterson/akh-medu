"""DAgger data: play the model, record the key states it visits, label them with Stockfish.

python dagger.py CHECKPOINT --games 200 --out runs/dagger/NAME.jsonl [--sample-prob 0.5]
    [--device cpu] [--engines 8] [--depth 8] [--max-states 60000]

Games are played as in eval.evaluate (half vs random, half vs sf0, model alternating colours).
Per game per model turn the keys are greedy with prob 1-sample_prob, else sampled from the softmax.
Every pre-press state (fen, cursor, holding) is recorded once and labelled with the teacher move:
the best move, or when holding a piece the best move from that square (null = drop it back).
One JSON line per state: {"fen", "cursor": [r, c], "holding": int|null, "move": uci|null}.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine

STOCKFISH = shutil.which("stockfish") or "stockfish"
_engine = None
_limit = None


def _init(depth: int) -> None:
    global _engine, _limit
    _engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    _engine.configure({"Threads": 1, "Hash": 16})
    _limit = chess.engine.Limit(depth=depth)


def label(state: tuple[str, list[int], int | None]) -> dict | None:
    fen, cursor, holding = state
    board = chess.Board(fen)
    if board.is_game_over():
        return None
    if holding is None:
        move = _engine.play(board, _limit).move
    else:
        candidates = [m for m in board.legal_moves
                      if m.from_square == holding and m.promotion in (None, chess.QUEEN)]
        if not candidates:
            move = None
        else:
            pv = _engine.analyse(board, _limit, root_moves=candidates).get("pv")
            move = pv[0] if pv else candidates[0]
    return {"fen": fen, "cursor": cursor, "holding": holding, "move": move.uci() if move else None}


def play(checkpoint: Path, games: int, sample_prob: float, device: str, max_states: int, seed: int = 0) -> list:
    import numpy as np
    import torch

    import eval as ev

    torch.set_num_threads(3)
    device = torch.device(device)
    model = ev.load(checkpoint, device).eval()
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    opponents = ("random", "sf0")
    pools = {"sf0": ev.Pool(ev.SKILL["sf0"])}
    live = [ev.Game(name, index % 2 == 0) for name in opponents for index in range(games // 2)]
    seen, states = set(), []
    done, steps, started = 0, 0, time.perf_counter()
    try:
        with torch.inference_mode():
            while live and len(states) < max_states:
                waiting = [g for g in live if g.board.turn != g.model_colour]
                batch = [g for g in waiting if g.opponent == "sf0"]
                if batch:
                    moves = pools["sf0"].run(lambda e, b: e.play(b, ev.OPPONENT_LIMIT).move, [g.board for g in batch])
                    for g, move in zip(batch, moves):
                        g.board.push(move)
                for g in waiting:
                    if g.opponent == "random":
                        g.board.push(rng.choice(list(g.board.legal_moves)))
                    g.check()
                movers = [g for g in live if not g.done and g.board.turn == g.model_colour]
                if movers:
                    for g in movers:
                        if g.turn_keys == 0:  # new turn: pick greedy or sampled for the whole turn
                            g.sampling = rng.random() < sample_prob
                        c, h = g.controller, g.controller.holding
                        key = (g.board.board_fen() + (" w" if g.board.turn else " b"), c.cursor, h)
                        if key not in seen:
                            seen.add(key)
                            states.append((g.board.fen(), list(c.cursor), h))
                    logits, _ = model(ev.observation_tensor(ev.observations(movers), device),
                                      torch.arange(7, 12, device=device))
                    greedy = logits.argmax(-1).tolist()
                    probs = logits.float().softmax(-1).cpu().numpy().astype(np.float64)
                    for g, best, p in zip(movers, greedy, probs):
                        k = int(np_rng.choice(5, p=p / p.sum())) if g.sampling else best
                        if g.press(k, rng)[1] is not None:
                            g.check()
                done += sum(g.done for g in live)
                live = [g for g in live if not g.done]
                steps += 1
                if steps % 200 == 0:
                    print(f"play step {steps}: {done} games done, {len(live)} live, {len(states)} states, "
                          f"{time.perf_counter() - started:.0f}s", flush=True)
    finally:
        for pool in pools.values():
            pool.close()
    print(f"played {done} games ({len(live)} cut off), {len(states)} states, {time.perf_counter() - started:.0f}s",
          flush=True)
    return states[:max_states]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-prob", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--engines", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--max-states", type=int, default=60000)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    states = play(args.checkpoint, args.games, args.sample_prob, args.device, args.max_states)
    written = 0
    with args.out.open("a") as handle, multiprocessing.Pool(args.engines, _init, (args.depth,)) as pool:
        for row in pool.imap_unordered(label, states, chunksize=8):
            if row is None:
                continue
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            written += 1
            if written % 1000 == 0:
                print(f"labelled {written}/{len(states)}, {time.perf_counter() - started:.0f}s", flush=True)
    print(f"wrote {written} states to {args.out} in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
