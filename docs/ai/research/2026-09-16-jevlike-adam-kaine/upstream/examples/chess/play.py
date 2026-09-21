"""Paced chess play trace in the same format as the Doom example's play trace.

python play.py CHECKPOINT --out runs/chess-film-trace.json --frames-dir runs/film-frames
    --decisions 150 [--opponent sf0] [--sample] [--device mps]

One decision = one key press. Frames are the 480x480 board the model saw before pressing,
JPEG in the trace (as Doom) and PNG in --frames-dir; decisions are paced like Doom (4/35 s).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import time
from pathlib import Path

import chess
import numpy as np
import torch
from PIL import Image

import render
from eval import SKILL, Game, Pool, load, opponent_move
from keys import KEYS
from jevlike.vision import benchmark, observation_tensor

DECISION_SECONDS = 4 / 35  # Doom's TICS_PER_ACTION / TICS_PER_SECOND


def clean_windows(decisions: list[dict], seconds: float = 5.0, count: int = 3) -> list[dict]:
    """The cleanest film windows: most moves played, no wasted or budget presses, within one game.

    Returned as build-film.mjs marks them, "episode@seconds into that episode".
    """
    length = round(seconds / DECISION_SECONDS)
    starts = {}
    for index, decision in enumerate(decisions):
        starts.setdefault(decision["episode"], index)
    scored = []
    for begin in range(len(decisions) - length + 1):
        window = decisions[begin:begin + length]
        if window[0]["episode"] != window[-1]["episode"]:
            continue
        events = [d["event"] for d in window]
        if any(e in ("edge", "empty", "illegal", "budget", "drop") for e in events):
            continue
        # moves count most; a window that opens mid-walk and ends on a put reads well
        scored.append((events.count("put") * 10 + events.count("lift") * 3, begin))
    chosen = []
    for score, begin in sorted(scored, reverse=True):
        if score and all(abs(begin - other) >= length for _, other in chosen):
            chosen.append((score, begin))
        if len(chosen) == count:
            break
    return [{"mark": f"{decisions[b]['episode']}@{(b - starts[decisions[b]['episode']]) * DECISION_SECONDS:.4f}",
             "moves": sum(d["event"] == "put" for d in decisions[b:b + length]),
             "lifts": sum(d["event"] == "lift" for d in decisions[b:b + length])} for _, b in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/chess-film-trace.json"))
    parser.add_argument("--frames-dir", type=Path, default=Path("runs/film-frames"))
    parser.add_argument("--decisions", type=int, default=150, help="total cap when --games is not given")
    parser.add_argument("--games", type=int, help="record this many games instead")
    parser.add_argument("--game-decisions", type=int, default=240,
                        help="with --games: decisions recorded per game (keeps the trace to a few hundred MB)")
    parser.add_argument("--opponent", choices=("random", *SKILL), default="sf0")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    model = load(args.checkpoint, device).eval()
    option_ids = torch.arange(7, 12, device=device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    pools = {args.opponent: Pool(SKILL[args.opponent], size=1)} if args.opponent in SKILL else {}
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    decisions, latencies = [], []
    episode, game, first_item, game_decisions = 0, None, None, 0
    started = time.perf_counter()
    try:
        while len(decisions) < args.decisions if args.games is None else True:
            if game is None or game.done or (args.games and game_decisions >= args.game_decisions):
                if args.games and episode == args.games:
                    break
                episode += 1
                game_decisions = 0
                game = Game(args.opponent, model_colour=episode % 2 == 1)
            if game.board.turn != game.model_colour:
                opponent_move(game, pools, rng)
                game.check()
                continue
            board, cursor, holding = game.board, game.controller.cursor, game.controller.holding
            flip = board.turn == chess.BLACK
            item = render.observation(board, cursor, holding)
            first_item = item if first_item is None else first_item
            picture = render.render_board(board, cursor, flip, holding, size=480)
            visible = len(board.piece_map())
            tick = time.perf_counter()
            with torch.inference_mode():
                logits, _, tensors = model.forward_trace(observation_tensor([item], device), option_ids)
                if device.type == "mps":
                    torch.mps.synchronize()
                if args.sample:
                    key = int(torch.multinomial(logits.softmax(-1)[0], 1).item())
                else:
                    key = int(logits.argmax(-1).item())
            latency = (time.perf_counter() - tick) * 1000
            event, move = game.press(key, rng)
            if move is not None:
                game.check()
            latencies.append(latency)
            index = len(decisions)
            game_decisions += 1
            Image.fromarray(picture).save(args.frames_dir / f"frame_{index:04d}.png")
            buffer = io.BytesIO()
            Image.fromarray(picture).save(buffer, format="JPEG", quality=88)
            all_attention = tensors["attention_map"][0].reshape(
                len(KEYS), model.grid_rows, model.grid_columns).cpu().tolist()
            decisions.append({
                "frame": base64.b64encode(buffer.getvalue()).decode(),
                "timestamp": time.perf_counter() - started,
                "episode": episode, "action": KEYS[key], "action_index": key,
                "probabilities": tensors["probabilities"][0].cpu().tolist(),
                "attention": all_attention[key],
                "activations": {
                    "query": tensors["query_matrix"][0].cpu().tolist(),
                    "key": tensors["key_matrix"][0, ::2, ::2].T.cpu().tolist(),
                    "value": tensors["value_matrix"][0, ::2, ::2].T.cpu().tolist(),
                    "attention": all_attention,
                    "scores": [tensors["logits_matrix"][0].cpu().tolist()],
                    "probabilities": [tensors["probabilities"][0].cpu().tolist()],
                },
                "latency_ms": latency, "event": event,
                "move": move.uci() if move is not None else None,
                "reward": 0.0, "running_reward": 0.0, "visible_objects": visible,
            })
            time.sleep(max(0.0, DECISION_SECONDS - (time.perf_counter() - tick)))
    finally:
        for pool in pools.values():
            pool.close()
    result = {
        "episodes": episode, "mean_reward": 0.0,
        "online_inference_ms": float(np.mean(latencies)),
        f"{device.type}_ms": benchmark(model, first_item, device, option_ids=option_ids.cpu()),
        "video": None, "opponent": args.opponent,
    }
    result["windows"] = clean_windows(decisions)
    args.out.write_text(json.dumps({
        **result, "trained_episodes": payload.get("episodes", 0),
        "training_rewards": payload.get("rewards", []),
        "actions": KEYS, "grid_rows": model.grid_rows, "grid_columns": model.grid_columns,
        "decisions": decisions,
    }))
    print(json.dumps({**result, "decisions": len(decisions), "trace": str(args.out),
                      "frames": str(args.frames_dir)}))


if __name__ == "__main__":
    main()
