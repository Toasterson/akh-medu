"""DAgger curriculum: label states visited by the pixel policy with the expert."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from jevlike.vision import DoomScorerV2, observation, observation_tensor
from train_imitation import augment, expert_action
from environment import DEFEND_ACTIONS, TICS_PER_ACTION, action_vector_for_game, make_game


def save(path: Path, model: DoomScorerV2, episodes: int, seed: int,
         kills: list[float], rewards: list[float], goals: list[bool], source: Path,
         updates: int, student_probability: float, read_init_noise: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 2, "model": model.state_dict(), "episodes": episodes,
        "rewards": rewards, "shaped_rewards": [], "kills": kills,
        "armour_reached": goals, "actions": DEFEND_ACTIONS,
        "width": model.width, "rank": model.rank, "seed": seed,
        "reads": model.reads,
        "scenario": "deadly_corridor", "random_mean_kills": 0.4,
        "training_config": {"method": "DAgger with labels-buffer expert",
                            "source_checkpoint": str(source), "updates": updates,
                            "student_rollout_probability": student_probability,
                            "read_init_noise": read_init_noise},
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialise-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/doom-dagger.pt"))
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--envs", type=int, choices=range(8, 17), default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs-per-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--student-rollout-probability", type=float, default=0.9)
    parser.add_argument("--attack-threshold", type=float, default=0.12)
    parser.add_argument("--attack-weight", type=float, default=1.0)
    parser.add_argument("--no-class-balance", action="store_true",
                        help="use the expert's natural action frequencies")
    parser.add_argument("--reads", type=int, choices=range(1, 5),
                        help="option-conditioned visual reads; defaults to source checkpoint")
    parser.add_argument("--read-init-noise", type=float, default=0.01,
                        help="symmetry-breaking noise for newly added cloned reads")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()
    if not 0 <= args.student_rollout_probability <= 1:
        parser.error("--student-rollout-probability must be in [0, 1]")
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    source = torch.load(args.initialise_from, map_location="cpu", weights_only=True)
    source_reads = source.get("reads", 1)
    reads = args.reads if args.reads is not None else source_reads
    model = DoomScorerV2(actions=7, width=source.get("width", 32),
                         rank=source.get("rank", 32), reads=reads).to(device)
    missing, unexpected = model.load_state_dict(source["model"], strict=False)
    allowed = {name for name in missing if name.startswith("extra_heads.")}
    if unexpected or set(missing) != allowed:
        raise ValueError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    for index in range(max(0, source_reads - 1), reads - 1):
        model.extra_heads[index].load_state_dict(model.head.state_dict())
        with torch.no_grad():
            for parameter in model.extra_heads[index].parameters():
                parameter.add_(torch.randn_like(parameter) * args.read_init_noise)
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
    updates = last_report = 0
    last_loss = last_accuracy = float("nan")
    started = time.perf_counter()
    try:
        while len(kills) < args.episodes:
            states, frames, items, labels, indices = [], [], [], [], []
            for index, game in enumerate(games):
                if game.is_episode_finished():
                    kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
                    rewards.append(float(game.get_total_reward()))
                    armour = game.get_game_variable(vzd.GameVariable.ARMOR)
                    health = game.get_game_variable(vzd.GameVariable.HEALTH)
                    goals.append(bool(armour > 0 or (health > 0 and game.get_episode_time() < 2100)))
                    if len(kills) >= args.episodes:
                        continue
                    game.new_episode()
                    previous[index] = None
                state = game.get_state()
                if state is None:
                    continue
                frame = np.ascontiguousarray(state.screen_buffer)
                states.append(state)
                frames.append(frame)
                items.append(observation(frame, previous[index]))
                labels.append(expert_action(state, args.attack_threshold))
                indices.append(index)
            if not items:
                continue
            model.eval()
            with torch.inference_mode():
                logits, _ = model(observation_tensor(items, device))
                student_actions = logits.argmax(-1).cpu().numpy()
            use_student = generator.random(len(items)) < args.student_rollout_probability
            rollout_actions = np.where(use_student, student_actions, np.asarray(labels))
            pending_items.extend(items)
            pending_labels.extend(labels)
            for offset, index in enumerate(indices):
                games[index].make_action(
                    action_vector_for_game(int(rollout_actions[offset]), DEFEND_ACTIONS, games[index]),
                    TICS_PER_ACTION,
                )
                previous[index] = frames[offset]

            while len(pending_items) >= args.batch_size:
                x_array = np.stack(pending_items[:args.batch_size])
                y_array = np.asarray(pending_labels[:args.batch_size], dtype=np.int64)
                del pending_items[:args.batch_size]
                del pending_labels[:args.batch_size]
                x_array, y_array = augment(x_array, y_array, generator)
                x = observation_tensor(x_array, device)
                y = torch.from_numpy(y_array).to(device)
                class_weight = None
                if not args.no_class_balance:
                    counts = torch.bincount(y, minlength=len(DEFEND_ACTIONS)).float()
                    present = counts > 0
                    class_weight = torch.zeros_like(counts)
                    class_weight[present] = len(y) / (present.sum() * counts[present])
                    class_weight[DEFEND_ACTIONS.index("attack")] *= args.attack_weight
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
            report = len(kills) // 50 * 50
            if report > last_report:
                last_report = report
                print(json.dumps({
                    "episode": len(kills), "mean_kills_50": float(np.mean(kills[-50:])),
                    "goal_rate_50": float(np.mean(goals[-50:])),
                    "mean_reward_50": float(np.mean(rewards[-50:])),
                    "updates": updates, "loss": last_loss, "accuracy": last_accuracy,
                    "seconds": time.perf_counter() - started,
                }), flush=True)
                save(args.output.with_name(f"{args.output.stem}-{len(kills)}.pt"), model,
                     len(kills), args.seed, kills, rewards, goals, args.initialise_from,
                     updates, args.student_rollout_probability, args.read_init_noise)
    finally:
        for game in games:
            game.close()
    save(args.output, model, len(kills), args.seed, kills, rewards, goals,
         args.initialise_from, updates, args.student_rollout_probability,
         args.read_init_noise)
    print(json.dumps({"event": "complete", "episodes": len(kills),
                      "mean_kills": float(np.mean(kills)), "goal_rate": float(np.mean(goals)),
                      "updates": updates, "checkpoint": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
