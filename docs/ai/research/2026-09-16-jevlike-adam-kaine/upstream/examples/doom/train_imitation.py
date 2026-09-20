"""Pixel-only imitation curriculum from a ViZDoom labels-buffer expert.

The labels buffer chooses training targets only.  The scorer receives the same
RGB-plus-motion tensor used by PPO and never receives labels at inference.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from diagnostics import enemy_observation
from jevlike.vision import DoomScorerV2, observation, observation_tensor
from environment import DEFEND_ACTIONS, TICS_PER_ACTION, action_vector_for_game, make_game


def expert_action(state, attack_threshold: float = 0.12) -> int:
    """Advance on a clear path; face and fire at the nearest visible enemy."""
    enemy = enemy_observation(state)
    if enemy is None:
        return DEFEND_ACTIONS.index("move forward")
    if abs(enemy["x"]) <= attack_threshold:
        return DEFEND_ACTIONS.index("attack")
    return DEFEND_ACTIONS.index("turn left" if enemy["x"] < 0 else "turn right")


def augment(items: np.ndarray, labels: np.ndarray, generator: np.random.Generator):
    """Mirror half the batch and exchange left/right controller targets."""
    items = items.copy()
    labels = labels.copy()
    mirrored = generator.random(len(items)) < 0.5
    items[mirrored] = items[mirrored, :, ::-1]
    swaps = {0: 1, 1: 0, 4: 5, 5: 4}
    labels[mirrored] = np.array([swaps.get(int(value), int(value)) for value in labels[mirrored]])
    return items, labels


def save(path: Path, model: DoomScorerV2, episodes: int, seed: int,
         kills: list[float], rewards: list[float], goals: list[bool], source: Path | None,
         updates: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 2, "model": model.state_dict(), "episodes": episodes,
        "rewards": rewards, "shaped_rewards": [], "kills": kills,
        "armour_reached": goals, "actions": DEFEND_ACTIONS,
        "width": model.width, "rank": model.rank, "seed": seed,
        "scenario": "deadly_corridor", "random_mean_kills": 0.4,
        "training_config": {"method": "labels-buffer expert imitation then PPO",
                            "source_checkpoint": str(source) if source else None,
                            "updates": updates},
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialise-from", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/doom-imitation.pt"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--envs", type=int, choices=range(8, 17), default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs-per-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--class-balance", action="store_true",
                        help="inverse-frequency CE; natural action frequencies work better here")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    source = (torch.load(args.initialise_from, map_location="cpu", weights_only=True)
              if args.initialise_from else None)
    model = DoomScorerV2(actions=7, width=source.get("width", 32) if source else 32,
                         rank=source.get("rank", 32) if source else 32).to(device)
    if source:
        model.load_state_dict(source["model"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    games = [make_game(args.seed + index, "deadly_corridor") for index in range(args.envs)]
    for game in games:
        game.new_episode()
    previous: list[np.ndarray | None] = [None] * len(games)
    pending_items: list[np.ndarray] = []
    pending_labels: list[int] = []
    kills: list[float] = []
    rewards: list[float] = []
    goals: list[bool] = []
    updates = 0
    last_loss = last_accuracy = float("nan")
    started = time.perf_counter()
    try:
        while len(kills) < args.episodes:
            for index, game in enumerate(games):
                if len(kills) >= args.episodes:
                    break
                if game.is_episode_finished():
                    kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
                    rewards.append(float(game.get_total_reward()))
                    armour = game.get_game_variable(vzd.GameVariable.ARMOR)
                    health = game.get_game_variable(vzd.GameVariable.HEALTH)
                    goals.append(bool(armour > 0 or (health > 0 and game.get_episode_time() < 2100)))
                    if len(kills) % 50 == 0:
                        print(json.dumps({
                            "episode": len(kills), "mean_kills_50": float(np.mean(kills[-50:])),
                            "goal_rate_50": float(np.mean(goals[-50:])),
                            "mean_reward_50": float(np.mean(rewards[-50:])),
                            "updates": updates, "loss": last_loss, "accuracy": last_accuracy,
                            "seconds": time.perf_counter() - started,
                        }), flush=True)
                        save(args.output.with_name(f"{args.output.stem}-{len(kills)}.pt"), model,
                             len(kills), args.seed, kills, rewards, goals,
                             args.initialise_from, updates)
                    if len(kills) >= args.episodes:
                        break
                    game.new_episode()
                    previous[index] = None
                state = game.get_state()
                if state is None:
                    continue
                frame = np.ascontiguousarray(state.screen_buffer)
                action = expert_action(state)
                pending_items.append(observation(frame, previous[index]))
                pending_labels.append(action)
                game.make_action(action_vector_for_game(action, DEFEND_ACTIONS, game),
                                 TICS_PER_ACTION)
                previous[index] = frame

            while len(pending_items) >= args.batch_size:
                items = np.stack(pending_items[:args.batch_size])
                labels = np.asarray(pending_labels[:args.batch_size], dtype=np.int64)
                del pending_items[:args.batch_size]
                del pending_labels[:args.batch_size]
                if not args.no_mirror:
                    items, labels = augment(items, labels, generator)
                x = observation_tensor(items, device)
                y = torch.from_numpy(labels).to(device)
                class_weight = None
                if args.class_balance:
                    counts = torch.bincount(y, minlength=len(DEFEND_ACTIONS)).float()
                    present = counts > 0
                    class_weight = torch.zeros_like(counts)
                    class_weight[present] = len(y) / (present.sum() * counts[present])
                model.train()
                for _ in range(args.epochs_per_batch):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits, y, weight=class_weight)
                    optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimiser.step()
                updates += 1
                last_loss = float(loss.detach())
                last_accuracy = float((logits.argmax(-1) == y).float().mean())
    finally:
        for game in games:
            game.close()
    save(args.output, model, len(kills), args.seed, kills, rewards, goals,
         args.initialise_from, updates)
    print(json.dumps({"event": "complete", "episodes": len(kills),
                      "mean_kills": float(np.mean(kills)),
                      "goal_rate": float(np.mean(goals)), "updates": updates,
                      "checkpoint": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
