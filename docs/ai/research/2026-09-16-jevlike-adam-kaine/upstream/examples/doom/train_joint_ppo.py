"""Jointly train one 12-option scorer on Doom PPO and chess key SFT."""

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

from diagnostics import enemy_observation
from jevlike.vision import (DOOM_OPTION_IDS, TOTAL_OPTIONS, DoomScorerV2,
                      attention_entropy_by_action, observation, observation_tensor)
from environment import DEFEND_ACTIONS, TICS_PER_ACTION, action_vector_for_game, make_game
from train_ppo import discounted_returns, shaped_reward


JOINT_ACTIONS = (*DEFEND_ACTIONS, "move up", "move down", "move left", "move right",
                 "pick up / put down")


def load_expanded(path: Path, device: torch.device,
                  chess_checkpoint: Path | None = None) -> tuple[DoomScorerV2, dict]:
    """Load a 7- or 12-option checkpoint into the one 12-option model."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = DoomScorerV2(actions=TOTAL_OPTIONS, width=payload.get("width", 32),
                         rank=payload.get("rank", 32), reads=payload.get("reads", 1))
    target = model.state_dict()
    for name, value in payload["model"].items():
        if name == "options.weight":
            rows = min(len(value), len(target[name]))
            target[name][:rows] = value[:rows]
        elif name in target and target[name].shape == value.shape:
            target[name] = value
    model.load_state_dict(target)
    if chess_checkpoint:
        chess_payload = torch.load(chess_checkpoint, map_location="cpu", weights_only=True)
        chess_options = chess_payload["model"]["options.weight"]
        if len(chess_options) < TOTAL_OPTIONS:
            raise ValueError("chess checkpoint does not contain the joint 12-option table")
        with torch.no_grad():
            model.options.weight[7:12].copy_(chess_options[7:12])
    return model.to(device), payload


def chess_frames(frames, device: torch.device) -> torch.Tensor:
    if isinstance(frames, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(frames))
    else:
        tensor = frames
    tensor = tensor.to(device)
    if tensor.ndim != 4:
        raise ValueError(f"chess frames must be rank four, got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 4:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.shape[1] != 4 or tensor.shape[2:] != (120, 160):
        raise ValueError(f"chess frames must be Bx4x120x160, got {tuple(tensor.shape)}")
    tensor = tensor.float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor


def reached_corridor_goal(game: vzd.DoomGame) -> bool:
    armour = game.get_game_variable(vzd.GameVariable.ARMOR)
    health = game.get_game_variable(vzd.GameVariable.HEALTH)
    return bool(armour > 0 or (health > 0 and game.get_episode_time() < 2100))


def doom_update(model: DoomScorerV2, optimiser: torch.optim.Optimizer,
                games: list[vzd.DoomGame], device: torch.device, args,
                entropy_coefficient: float) -> dict:
    for group in optimiser.param_groups:
        group["lr"] = args.doom_learning_rate
    model.train()
    trajectories = []
    doom_ids = torch.tensor(DOOM_OPTION_IDS, device=device)
    while len(trajectories) < args.doom_episodes_per_update:
        count = min(len(games), args.doom_episodes_per_update - len(trajectories))
        wave = [{"observations": [], "actions": [], "log_probs": [], "values": [],
                 "rewards": [], "expert_actions": [], "shaping": 0.0} for _ in range(count)]
        active = [True] * count
        previous: list[np.ndarray | None] = [None] * count
        for game in games[:count]:
            game.new_episode()
        while any(active):
            indices = [index for index, running in enumerate(active) if running]
            states = [games[index].get_state() for index in indices]
            frames = [np.ascontiguousarray(state.screen_buffer) for state in states]
            items = [observation(frame, previous[index]) for frame, index in zip(frames, indices)]
            with torch.inference_mode():
                logits, values = model(observation_tensor(items, device), doom_ids)
                distribution = torch.distributions.Categorical(logits=logits)
                actions = distribution.sample()
                log_probs = distribution.log_prob(actions)
                actions_cpu = actions.cpu().numpy()
                log_probs_cpu = log_probs.cpu().numpy()
                values_cpu = values.cpu().numpy()
            for offset, index in enumerate(indices):
                game = games[index]
                action = int(actions_cpu[offset])
                before = enemy_observation(states[offset])
                kills_before = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                hits_before = game.get_game_variable(vzd.GameVariable.HITCOUNT)
                raw = game.make_action(action_vector_for_game(action, DEFEND_ACTIONS, game),
                                       TICS_PER_ACTION)
                next_state = None if game.is_episode_finished() else game.get_state()
                after = enemy_observation(next_state) if next_state is not None else None
                bonus = shaped_reward(before, after, DEFEND_ACTIONS[action])
                if DEFEND_ACTIONS[action] == "move forward":
                    bonus += args.forward_bonus
                bonus += args.kill_bonus * max(
                    0.0, game.get_game_variable(vzd.GameVariable.KILLCOUNT) - kills_before)
                bonus += args.hit_bonus * max(
                    0.0, game.get_game_variable(vzd.GameVariable.HITCOUNT) - hits_before)
                trajectory = wave[index]
                trajectory["observations"].append(items[offset])
                trajectory["actions"].append(action)
                if before is None:
                    expert = DEFEND_ACTIONS.index("move forward")
                elif abs(before["x"]) <= 0.12:
                    expert = DEFEND_ACTIONS.index("attack")
                else:
                    expert = DEFEND_ACTIONS.index(
                        "turn left" if before["x"] < 0 else "turn right"
                    )
                trajectory["expert_actions"].append(expert)
                trajectory["log_probs"].append(float(log_probs_cpu[offset]))
                trajectory["values"].append(float(values_cpu[offset]))
                trajectory["rewards"].append(float(raw) * args.reward_scale + bonus)
                trajectory["shaping"] += bonus
                previous[index] = frames[offset]
                if game.is_episode_finished():
                    active[index] = False
        for index, trajectory in enumerate(wave):
            game = games[index]
            trajectory["raw_reward"] = float(game.get_total_reward())
            trajectory["kills"] = float(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
            trajectory["goal"] = reached_corridor_goal(game)
        trajectories.extend(wave)

    states, actions, expert_actions, old_log_probs, old_values, returns = [], [], [], [], [], []
    for trajectory in trajectories:
        states.extend(trajectory["observations"])
        actions.extend(trajectory["actions"])
        expert_actions.extend(trajectory["expert_actions"])
        old_log_probs.extend(trajectory["log_probs"])
        old_values.extend(trajectory["values"])
        returns.extend(discounted_returns(trajectory["rewards"], 0.99))
    state_array = np.stack(states)
    action_tensor = torch.tensor(actions, device=device)
    expert_action_tensor = torch.tensor(expert_actions, device=device)
    old_log_prob_tensor = torch.tensor(old_log_probs, device=device)
    return_tensor = torch.tensor(np.asarray(returns), device=device)
    advantage = return_tensor - torch.tensor(old_values, device=device)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-5)
    samples = len(state_array)
    expert_weight = None
    if args.expert_class_balance:
        counts = torch.bincount(expert_action_tensor, minlength=len(DEFEND_ACTIONS)).float()
        present = counts > 0
        expert_weight = torch.zeros_like(counts)
        expert_weight[present] = samples / (present.sum() * counts[present])
    for _ in range(args.ppo_epochs):
        for batch in torch.randperm(samples, device=device).split(args.minibatch):
            logits, predicted = model(
                observation_tensor(state_array[batch.cpu().numpy()], device), doom_ids
            )
            distribution = torch.distributions.Categorical(logits=logits)
            current = distribution.log_prob(action_tensor[batch])
            ratio = (current - old_log_prob_tensor[batch]).exp()
            policy_loss = -torch.minimum(
                ratio * advantage[batch], ratio.clamp(0.8, 1.2) * advantage[batch]
            ).mean()
            value_loss = F.smooth_l1_loss(predicted, return_tensor[batch])
            expert_loss = F.cross_entropy(
                logits, expert_action_tensor[batch], weight=expert_weight
            )
            entropy = distribution.entropy().mean()
            loss = (policy_loss + 0.5 * value_loss - entropy_coefficient * entropy
                    + args.expert_coefficient * expert_loss)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
    attention_entropy = attention_entropy_by_action(model, state_array, device, doom_ids)
    return {
        "episodes": len(trajectories), "transitions": samples,
        "mean_kills": float(np.mean([row["kills"] for row in trajectories])),
        "mean_raw_reward": float(np.mean([row["raw_reward"] for row in trajectories])),
        "goal_rate": float(np.mean([row["goal"] for row in trajectories])),
        "mean_shaping": float(np.mean([row["shaping"] for row in trajectories])),
        "policy_entropy": float(entropy.detach()), "ppo_loss": float(loss.detach()),
        "expert_loss": float(expert_loss.detach()),
        "attention_entropy_by_action": dict(zip(DEFEND_ACTIONS, attention_entropy)),
        "return_target_mean": float(return_tensor.mean()),
        "return_target_std": float(return_tensor.std(unbiased=False)),
        "value_prediction_mean": float(predicted.detach().mean()),
        "value_prediction_std": float(predicted.detach().std(unbiased=False)),
    }


def chess_step(model: DoomScorerV2, optimiser: torch.optim.Optimizer, batch,
               device: torch.device) -> tuple[float, float]:
    model.train()
    frames, option_ids, label_index = batch
    frames = chess_frames(frames, device)
    option_ids = torch.as_tensor(option_ids, dtype=torch.long, device=device)
    labels = torch.as_tensor(label_index, dtype=torch.long, device=device)
    logits, _ = model(frames, option_ids)
    loss = F.cross_entropy(logits, labels)
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimiser.step()
    accuracy = (logits.argmax(-1) == labels).float().mean()
    return float(loss.detach()), float(accuracy)


def save(path: Path, model: DoomScorerV2, cycles: int, doom_episodes: int, seed: int,
         source: Path, chess_source: Path | None, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 3, "model": model.state_dict(), "episodes": doom_episodes,
        "seed": seed,
        "joint_cycles": cycles, "actions": JOINT_ACTIONS, "doom_option_ids": DOOM_OPTION_IDS,
        "chess_option_ids": tuple(range(7, 12)), "width": model.width, "rank": model.rank,
        "reads": model.reads,
        "scenario": "deadly_corridor", "source_doom_checkpoint": str(source),
        "source_chess_checkpoint": str(chess_source) if chess_source else None,
        "history": history,
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doom-checkpoint", type=Path, required=True)
    parser.add_argument("--chess-checkpoint", type=Path,
                        help="optionally seed only chess option rows 7..11")
    parser.add_argument("--output", type=Path, default=Path("runs/joint-ppo.pt"))
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--doom-envs", type=int, choices=range(8, 17), default=16)
    parser.add_argument("--doom-episodes-per-update", type=int, default=32)
    parser.add_argument("--chess-steps", type=int, default=32)
    parser.add_argument("--chess-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--learning-rate", type=float, dest="doom_learning_rate", default=1e-4,
                        help="learning rate for Doom PPO updates")
    parser.add_argument("--chess-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--entropy", type=float, default=0.002)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--forward-bonus", type=float, default=0.01)
    parser.add_argument("--kill-bonus", type=float, default=1.0)
    parser.add_argument("--hit-bonus", type=float, default=0.05)
    parser.add_argument("--expert-coefficient", type=float, default=0.0,
                        help="labels-buffer imitation term alongside Doom PPO")
    parser.add_argument("--expert-class-balance", action="store_true",
                        help="inverse-frequency weight expert CE (natural frequency is default)")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=4,
                        help="games per opponent in each quick chess evaluation")
    parser.add_argument("--no-eval", action="store_true",
                        help="skip chess game evaluation (useful for interface smoke tests)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model, source_payload = load_expanded(args.doom_checkpoint, device, args.chess_checkpoint)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.doom_learning_rate,
                                  weight_decay=1e-4)
    chess_directory = str(Path(__file__).resolve().parents[1] / "chess")
    if chess_directory not in sys.path:
        sys.path.append(chess_directory)
    data_module = importlib.import_module("data")
    eval_module = importlib.import_module("eval")
    batch_source = data_module.batches(args.chess_batch_size, seed=args.seed)
    batch_iterator = iter(batch_source)
    games = [make_game(args.seed + index, "deadly_corridor") for index in range(args.doom_envs)]
    history: list[dict] = []
    doom_episodes = 0
    started = time.perf_counter()
    try:
        for cycle in range(1, args.cycles + 1):
            doom = doom_update(model, optimiser, games, device, args, args.entropy)
            doom_episodes += doom["episodes"]
            chess_losses, chess_accuracies = [], []
            for group in optimiser.param_groups:
                group["lr"] = args.chess_learning_rate
            for _ in range(args.chess_steps):
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    batch_iterator = iter(data_module.batches(args.chess_batch_size, seed=args.seed))
                    batch = next(batch_iterator)
                loss, accuracy = chess_step(model, optimiser, batch, device)
                chess_losses.append(loss)
                chess_accuracies.append(accuracy)
            row = {
                "cycle": cycle, "doom_episodes": doom_episodes, **doom,
                "chess_loss": float(np.mean(chess_losses)),
                "chess_key_accuracy": float(np.mean(chess_accuracies)),
                "wall_seconds": time.perf_counter() - started,
                "popart_scale_check": {
                    "doom_return_mean": doom["return_target_mean"],
                    "doom_return_std": doom["return_target_std"],
                    "doom_value_prediction_mean": doom["value_prediction_mean"],
                    "doom_value_prediction_std": doom["value_prediction_std"],
                    "chess_cross_entropy": float(np.mean(chess_losses)),
                    "return_std_to_chess_ce": doom["return_target_std"] /
                    max(float(np.mean(chess_losses)), 1e-8),
                },
            }
            if not args.no_eval and (cycle % args.eval_every == 0 or cycle == args.cycles):
                row["chess_eval"] = eval_module.evaluate(
                    model, games=args.eval_games, device=args.device, quick=True
                )
            history.append(row)
            print(json.dumps(row), flush=True)
            save(args.output, model, cycle, doom_episodes, args.seed, args.doom_checkpoint,
                 args.chess_checkpoint, history)
    finally:
        for game in games:
            game.close()
    print(json.dumps({"event": "complete", "cycles": args.cycles,
                      "doom_episodes": doom_episodes, "checkpoint": str(args.output),
                      "source_episodes": source_payload.get("episodes")}), flush=True)


if __name__ == "__main__":
    main()
