"""Deadline path: alternating Doom and chess DAgger CE in one 12-option model."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from jevlike.vision import DOOM_OPTION_IDS, TOTAL_OPTIONS, observation, observation_tensor
from train_imitation import augment, expert_action
from environment import DEFEND_ACTIONS, TICS_PER_ACTION, action_vector_for_game, make_game
from train_joint_ppo import JOINT_ACTIONS, chess_frames, load_expanded


def save(path: Path, model, source: Path, chess_source: Path, seed: int,
         updates: int, episodes: int, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 4, "model": model.state_dict(), "episodes": episodes,
        "seed": seed, "actions": JOINT_ACTIONS, "doom_option_ids": DOOM_OPTION_IDS,
        "chess_option_ids": tuple(range(7, 12)), "width": model.width,
        "rank": model.rank, "reads": model.reads, "scenario": "deadly_corridor",
        "source_doom_checkpoint": str(source),
        "source_chess_checkpoint": str(chess_source), "history": history,
        "training_config": {"method": "alternating on-policy DAgger CE; no PPO",
                            "updates_per_domain": updates},
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doom-checkpoint", type=Path, required=True)
    parser.add_argument("--chess-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chess-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--chess-learning-rate", type=float, default=3e-4)
    parser.add_argument("--chess-steps", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--heldout-size", type=int, default=1000)
    parser.add_argument("--student-probability", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=163)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    model, _ = load_expanded(args.doom_checkpoint, device, args.chess_checkpoint)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    chess_directory = str(Path(__file__).resolve().parents[1] / "chess")
    if chess_directory not in sys.path:
        sys.path.append(chess_directory)
    data = importlib.import_module("data")
    chess_batches = iter(data.batches(args.chess_batch_size, seed=args.seed))
    heldout_frames, heldout_keys, heldout_first, _, _ = data.fixed_set(
        args.heldout_size, seed=args.seed + 1, split="heldout"
    )
    games = [make_game(args.seed + index, "deadly_corridor") for index in range(args.envs)]
    for game in games:
        game.new_episode()
    previous = [None] * len(games)
    pending_x: list[np.ndarray] = []
    pending_y: list[int] = []
    episodes = updates = 0
    kills: list[float] = []
    goals: list[bool] = []
    history: list[dict] = []
    started = time.perf_counter()
    try:
        while updates < args.updates:
            frames, items, labels, indices = [], [], [], []
            for index, game in enumerate(games):
                if game.is_episode_finished():
                    episodes += 1
                    kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
                    goals.append(bool(game.get_game_variable(vzd.GameVariable.ARMOR) > 0))
                    game.new_episode()
                    previous[index] = None
                state = game.get_state()
                if state is None:
                    continue
                frame = np.ascontiguousarray(state.screen_buffer)
                frames.append(frame)
                items.append(observation(frame, previous[index]))
                labels.append(expert_action(state, 0.20))
                indices.append(index)
            model.eval()
            with torch.inference_mode():
                logits, _ = model(observation_tensor(items, device),
                                  torch.tensor(DOOM_OPTION_IDS, device=device))
                student = logits.argmax(-1).cpu().numpy()
            actions = np.where(rng.random(len(items)) < args.student_probability,
                               student, np.asarray(labels))
            pending_x.extend(items)
            pending_y.extend(labels)
            for offset, index in enumerate(indices):
                games[index].make_action(
                    action_vector_for_game(int(actions[offset]), DEFEND_ACTIONS, games[index]),
                    TICS_PER_ACTION,
                )
                previous[index] = frames[offset]
            if len(pending_x) < args.batch_size:
                continue

            x_array = np.stack(pending_x[:args.batch_size])
            y_array = np.asarray(pending_y[:args.batch_size], dtype=np.int64)
            del pending_x[:args.batch_size]
            del pending_y[:args.batch_size]
            x_array, y_array = augment(x_array, y_array, rng)
            model.train()
            doom_logits, _ = model(observation_tensor(x_array, device),
                                   torch.tensor(DOOM_OPTION_IDS, device=device))
            doom_y = torch.from_numpy(y_array).to(device)
            doom_loss = F.cross_entropy(doom_logits, doom_y)
            optimiser.zero_grad()
            doom_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()

            chess_losses, chess_accuracies = [], []
            for group in optimiser.param_groups:
                group["lr"] = args.chess_learning_rate
            for _ in range(args.chess_steps):
                chess_x, chess_ids, chess_y = next(chess_batches)
                chess_logits, _ = model(chess_frames(chess_x, device),
                                        torch.as_tensor(chess_ids, device=device))
                chess_y = torch.as_tensor(chess_y, dtype=torch.long, device=device)
                chess_loss = F.cross_entropy(chess_logits, chess_y)
                optimiser.zero_grad()
                chess_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimiser.step()
                chess_losses.append(float(chess_loss.detach()))
                chess_accuracies.append(float((chess_logits.argmax(-1) == chess_y).float().mean()))
            for group in optimiser.param_groups:
                group["lr"] = args.learning_rate
            updates += 1
            row = {"update": updates, "doom_episodes": episodes,
                   "doom_loss": float(doom_loss.detach()),
                   "doom_accuracy": float((doom_logits.argmax(-1) == doom_y).float().mean()),
                   "chess_loss": float(np.mean(chess_losses)),
                   "chess_accuracy": float(np.mean(chess_accuracies)),
                   "mean_kills": float(np.mean(kills)) if kills else None,
                   "goal_rate": float(np.mean(goals)) if goals else None,
                   "wall_seconds": time.perf_counter() - started}
            if updates % args.eval_every == 0 or updates == args.updates:
                model.eval()
                predictions = []
                with torch.inference_mode():
                    for start in range(0, len(heldout_frames), 256):
                        logits, _ = model(
                            observation_tensor(heldout_frames[start:start + 256], device),
                            torch.arange(7, 12, device=device),
                        )
                        predictions.append(logits.argmax(-1).cpu().numpy())
                predicted = np.concatenate(predictions)
                row["chess_heldout_key_accuracy"] = float(
                    np.mean(predicted == heldout_keys)
                )
                row["chess_heldout_first_key_accuracy"] = float(
                    np.mean(predicted[heldout_first] == heldout_keys[heldout_first])
                )
                row["doom_mean_kills_50"] = float(np.mean(kills[-50:])) if kills else None
                row["doom_goal_rate_50"] = float(np.mean(goals[-50:])) if goals else None
                snapshot = args.output.with_name(f"{args.output.stem}-{updates}.pt")
                save(snapshot, model, args.doom_checkpoint, args.chess_checkpoint,
                     args.seed, updates, episodes, history + [row])
                print(json.dumps({"event": "milestone", **row,
                                  "checkpoint": str(snapshot)}), flush=True)
            history.append(row)
            if updates % 10 == 0:
                print(json.dumps(row), flush=True)
            save(args.output, model, args.doom_checkpoint, args.chess_checkpoint,
                 args.seed, updates, episodes, history)
    finally:
        for game in games:
            game.close()
    print(json.dumps({"event": "complete", "updates": updates,
                      "doom_episodes": episodes, "checkpoint": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
