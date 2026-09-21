"""Play full chess games through the key interface and score them.

python eval.py CHECKPOINT [--games 50] [--device mps] [--quick] [--sample] [--out runs/eval-NAME.json]

The model presses keys (up/down/left/right/toggle = option ids 7..11) on the rendered
screen until a move is played or KEY_BUDGET presses are spent; then a uniformly random
legal move is played and a failure counted. All live games share one forward pass per step.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

import keys  # noqa: E402
import render  # noqa: E402
from jevlike.vision import CHESS_OPTION_IDS, DoomScorerV2, observation_tensor  # noqa: E402

STOCKFISH = shutil.which("stockfish") or "stockfish"
KEY_BUDGET = 40
MAX_PLIES = 200
ENGINES = 4
SKILL = {"sf0": 0, "sf3": 3}
# Skill Level picks its move at depth 1 + level, so a shallow fixed depth is
# as weak as a short time limit and much cheaper; raise it for a stronger opponent.
OPPONENT_LIMIT = chess.engine.Limit(depth=5)
CPL_LIMIT = chess.engine.Limit(depth=10)
PLURAL = {"win": "wins", "draw": "draws", "loss": "losses"}


class Pool:
    """A few Stockfish processes shared by worker threads."""

    def __init__(self, skill: int | None = None, size: int = ENGINES) -> None:
        self.engines = [chess.engine.SimpleEngine.popen_uci(STOCKFISH) for _ in range(size)]
        self.free: queue.Queue = queue.Queue()
        for engine in self.engines:
            engine.configure({"Threads": 1, "Hash": 16, **({"Skill Level": skill} if skill is not None else {})})
            self.free.put(engine)

    def run(self, fn, items: list) -> list:
        def call(item):
            engine = self.free.get()
            try:
                return fn(engine, item)
            finally:
                self.free.put(engine)
        with ThreadPoolExecutor(len(self.engines)) as executor:
            return list(executor.map(call, items))

    def close(self) -> None:
        for engine in self.engines:
            engine.quit()


def pick_keys(model, frames: np.ndarray, device, sample: bool, rng: np.random.Generator) -> list[int]:
    logits, _ = model(observation_tensor(frames, device), torch.arange(7, 12, device=device))
    if not sample:
        return logits.argmax(-1).tolist()
    # sample=True is temperature 1; a float is the softmax temperature
    probs = (logits.float() / float(sample)).softmax(-1).cpu().numpy().astype(np.float64)
    return [int(rng.choice(5, p=p / p.sum())) for p in probs]


def observations(games: list) -> np.ndarray:
    """render.observation for many games, written straight into one batch array (2-3x faster)."""
    left = (render.FRAME_WIDTH - render.BOARD) // 2
    batch = np.zeros((len(games), render.FRAME_HEIGHT, render.FRAME_WIDTH, 4), np.uint8)
    batch[..., 3] = 128
    for item, g in zip(batch, games):
        if g.board_image is None:  # wasted presses leave the screen unchanged
            g.board_image = render.render_board(
                g.board, g.controller.cursor, g.board.turn == chess.BLACK, g.controller.holding, render.BOARD)
        item[:, left:left + render.BOARD, :3] = g.board_image
    return batch


class Game:
    def __init__(self, opponent: str, model_colour: bool) -> None:
        self.board = chess.Board()
        self.controller = keys.Controller(self.board, (7, 4))
        self.opponent, self.model_colour = opponent, model_colour
        self.turn_keys = 0
        self.presses = self.wasted = self.completed_keys = self.completed = self.failures = 0
        self.model_moves: list[tuple[str, str]] = []  # (fen before, uci) for moves the model played itself
        self.adjudicated = self.done = False
        self.board_image: np.ndarray | None = None

    def check(self) -> None:
        """Call after every move; game-over tests are too slow to run per key press."""
        self.board_image = None
        self.adjudicated = not self.board.is_game_over() and self.board.ply() >= MAX_PLIES
        self.done = self.adjudicated or self.board.is_game_over()

    def press(self, key: int, rng: random.Random) -> tuple[str, chess.Move | None]:
        """One model key; returns the event and the move played (by the model or the budget)."""
        fen = self.board.fen() if key == keys.TOGGLE and self.controller.holding is not None else None
        event, move = self.controller.press(key)
        self.presses += 1
        self.turn_keys += 1
        if event in keys.WASTED:
            self.wasted += 1
        else:
            self.board_image = None
        if event == "put":
            self.completed += 1
            self.completed_keys += self.turn_keys
            self.model_moves.append((fen, move.uci()))
        elif self.turn_keys >= KEY_BUDGET:
            move = rng.choice(list(self.board.legal_moves))
            self.board.push(move)
            self.controller.holding = None
            self.failures += 1
            event = "budget"
        if move is not None:
            self.turn_keys = 0
        return event, move

    def result(self) -> str:
        if self.adjudicated or not self.board.is_checkmate():
            return "draw"
        return "loss" if self.board.turn == self.model_colour else "win"


def opponent_move(game: Game, pools: dict, rng: random.Random) -> None:
    if game.opponent == "random":
        game.board.push(rng.choice(list(game.board.legal_moves)))
    else:
        game.board.push(pools[game.opponent].run(
            lambda engine, board: engine.play(board, OPPONENT_LIMIT).move, [game.board])[0])


def cpl(engine, item) -> int:
    fen, uci = item
    board = chess.Board(fen)
    colour = board.turn
    best = engine.analyse(board, CPL_LIMIT)["score"].pov(colour).score(mate_score=10000)
    board.push_uci(uci)
    played = engine.analyse(board, CPL_LIMIT)["score"].pov(colour).score(mate_score=10000)
    return min(1000, max(0, best - played))


def material(board: chess.Board, colour: bool) -> int:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    return sum(v * (len(board.pieces(t, colour)) - len(board.pieces(t, not colour))) for t, v in values.items())


def evaluate(model, games: int = 50, device="mps", quick: bool = False, sample: bool = False,
             seed: int = 0, log=None) -> dict:
    device = torch.device(device)
    opponents = ("random", "sf0") if quick else ("random", "sf0", "sf3")
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    was_training = model.training
    model.eval()
    pools = {name: Pool(SKILL[name]) for name in opponents if name in SKILL}
    live = [Game(name, index % 2 == 0) for name in opponents for index in range(games)]
    finished: list[Game] = []
    forward_seconds, forward_keys, steps = 0.0, 0, 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            while live:
                waiting = [g for g in live if g.board.turn != g.model_colour]
                if waiting:  # one engine call per game, spread over each pool's engines
                    for name in opponents:
                        batch = [g for g in waiting if g.opponent == name]
                        if name in pools and batch:
                            moves = pools[name].run(lambda e, b: e.play(b, OPPONENT_LIMIT).move,
                                                    [g.board for g in batch])
                            for g, move in zip(batch, moves):
                                g.board.push(move)
                        else:
                            for g in batch:
                                opponent_move(g, pools, rng)
                for g in waiting:
                    g.check()
                movers = [g for g in live if not g.done and g.board.turn == g.model_colour]
                if movers:
                    frames = observations(movers)
                    tick = time.perf_counter()
                    chosen = pick_keys(model, frames, device, sample, np_rng)
                    if device.type == "mps":
                        torch.mps.synchronize()
                    forward_seconds += time.perf_counter() - tick
                    forward_keys += len(movers)
                    for g, key in zip(movers, chosen):
                        if g.press(key, rng)[1] is not None:
                            g.check()
                finished += [g for g in live if g.done]
                live = [g for g in live if not g.done]
                steps += 1
                if steps % 500 == 0:
                    print(f"eval step {steps}: {len(finished)} games done, {len(live)} live, "
                          f"{time.perf_counter() - started:.0f}s", file=sys.stderr, flush=True)
        result = {"mode": "quick" if quick else "full", "sample": sample,
                  "ms_per_key": 1000 * forward_seconds / max(1, forward_keys)}
        analysis = None if quick else Pool()
        try:
            for name in opponents:
                played = [g for g in finished if g.opponent == name]
                outcomes = [g.result() for g in played]
                moves = sum(g.completed + g.failures for g in played)
                metrics = {
                    "games": len(played),
                    **{p: outcomes.count(k) for k, p in PLURAL.items()},
                    "adjudicated": sum(g.adjudicated for g in played),
                    # material lead (P=1 N=B=3 R=5 Q=9) of 3+ for the model when a game hit the ply cap
                    "adjudicated_model_ahead": sum(g.adjudicated and material(g.board, g.model_colour) >= 3 for g in played),
                    "adjudicated_model_behind": sum(g.adjudicated and material(g.board, g.model_colour) <= -3 for g in played),
                    "wasted_press_rate": sum(g.wasted for g in played) / max(1, sum(g.presses for g in played)),
                    "keys_per_move": sum(g.completed_keys for g in played) / max(1, sum(g.completed for g in played)),
                    "failure_rate": sum(g.failures for g in played) / max(1, moves),
                    "model_moves": moves,
                    "by_colour": {
                        colour: {p: sum(g.result() == k for g in played if g.model_colour == (colour == "white"))
                                 for k, p in PLURAL.items()}
                        for colour in ("white", "black")},
                }
                if analysis:
                    losses = analysis.run(cpl, [item for g in played for item in g.model_moves])
                    # CPL over moves the model completed itself; budget-fallback random moves are excluded.
                    metrics["cpl"] = float(np.mean(losses)) if losses else None
                    metrics["cpl_moves"] = len(losses)
                result[name] = metrics
        finally:
            if analysis:
                analysis.close()
    finally:
        for pool in pools.values():
            pool.close()
        model.train(was_training)
    result["seconds"] = time.perf_counter() - started
    if log:
        with open(log, "a") as handle:
            handle.write(json.dumps(result) + "\n")
    return result


def load(checkpoint, device) -> DoomScorerV2:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = DoomScorerV2(actions=12, width=payload.get("width", 32), rank=payload.get("rank", 32))
    model.load_state_dict(payload["model"])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sample", nargs="?", type=float, const=1.0, default=0.0,
                        help="sample keys at this softmax temperature (bare flag: 1.0); omit for greedy")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(load(args.checkpoint, args.device), args.games, args.device, args.quick,
                      args.sample, log=args.out)
    print(f"{'opponent':9} {'W':>3} {'D':>3} {'L':>3} {'wasted':>7} {'keys/mv':>7} {'fail':>6} {'cpl':>6}")
    for name in ("random", "sf0", "sf3"):
        if name in result:
            m = result[name]
            cpl_text = "-" if m.get("cpl") is None else f"{m['cpl']:.0f}"
            print(f"{name:9} {m['wins']:3} {m['draws']:3} {m['losses']:3} {m['wasted_press_rate']:7.2f} "
                  f"{m['keys_per_move']:7.1f} {m['failure_rate']:6.2f} {cpl_text:>6}")
    print(f"{result['ms_per_key']:.3f} ms/key, {result['seconds']:.1f} s, mode {result['mode']}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        game = Game("random", True)
        for uci in ("e2e4", "e7e5", "g1f3"):
            game.board.push_uci(uci)
        game.controller.cursor, game.controller.holding = (1, 3), chess.E5
        assert (observations([game])[0] == render.observation(game.board, (1, 3), chess.E5)).all()
        print("ok")
    else:
        main()
