"""Unpaced PPO for the screen-dependent Doom scorer."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from diagnostics import enemy_observation
from jevlike.vision import DoomScorerV2, attention_entropy_by_action, observation, observation_tensor
from environment import DEFEND_ACTIONS, TICS_PER_ACTION, action_vector_for_game, draw_curve, make_game


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    result = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        result[index] = running
    return result


def shaped_reward(before: dict[str, float] | None, after: dict[str, float] | None,
                  action: str) -> float:
    """Dense aiming signal; labels shape reward but never enter the policy."""
    if before is None:
        return 0.0
    x = before["x"]
    reward = 0.002 * (1.0 - min(1.0, abs(x)))
    if abs(x) <= 0.15:
        reward += 0.04 if action == "attack" else -0.002
    elif x < 0:
        reward += 0.03 if action == "turn left" else -0.01 if action == "turn right" else 0.0
        if action == "attack":
            reward -= 0.005
    else:
        reward += 0.03 if action == "turn right" else -0.01 if action == "turn left" else 0.0
        if action == "attack":
            reward -= 0.005
    if after is not None:
        reward += 0.02 * (abs(x) - abs(after["x"]))
    return reward


def gradient_norms(model: DoomScorerV2) -> dict[str, float]:
    result = {}
    for group in ("stem", "patch", "options", "head", "extra_heads", "value_head"):
        squared = 0.0
        for name, parameter in model.named_parameters():
            if (name == group or name.startswith(group + ".")) and parameter.grad is not None:
                squared += float(parameter.grad.detach().float().square().sum())
        result[group] = math.sqrt(squared)
    return result


def save(path: Path, model: DoomScorerV2, episode: int, rewards: list[float],
         shaped: list[float], kills: list[float], seed: int, scenario: str,
         baseline: float, armour_reached: list[bool], training_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 2, "model": model.state_dict(), "episodes": episode,
        "rewards": rewards, "shaped_rewards": shaped, "kills": kills,
        "armour_reached": armour_reached, "actions": DEFEND_ACTIONS,
        "width": model.width, "rank": model.rank, "seed": seed,
        "reads": model.reads,
        "scenario": scenario, "random_mean_kills": baseline,
        "training_config": training_config,
    }, path)
    draw_curve(rewards, path.with_name(path.stem + "-reward.png"))


def random_baseline(games: list[vzd.DoomGame], episodes: int, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    kills, armour = [], []
    for episode in range(episodes):
        game = games[episode % len(games)]
        game.new_episode()
        while not game.is_episode_finished():
            action = int(generator.integers(len(DEFEND_ACTIONS)))
            game.make_action(action_vector_for_game(action, DEFEND_ACTIONS, game), TICS_PER_ACTION)
        kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
        try:
            armour.append(game.get_game_variable(vzd.GameVariable.ARMOR) > 0)
        except Exception:
            armour.append(False)
    return float(np.mean(kills)), float(np.mean(armour))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--envs", type=int, choices=range(8, 17), default=16)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--scenario", choices=("defend_the_center", "deadly_corridor"),
                        default="defend_the_center")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--entropy-start", type=float, default=0.01)
    parser.add_argument("--entropy-end", type=float, default=0.001)
    parser.add_argument("--entropy-decay-episodes", type=int, default=4000)
    parser.add_argument("--shaping-scale", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="scale environment rewards for PPO; reporting remains raw")
    parser.add_argument("--forward-bonus", type=float, default=0.002,
                        help="corridor-only bonus for selecting move forward")
    parser.add_argument("--clear-path-forward-bonus", type=float, default=0.0,
                        help="extra corridor forward bonus when no enemy label is visible")
    parser.add_argument("--clear-path-idle-penalty", type=float, default=0.0,
                        help="penalty for non-forward actions when no enemy label is visible")
    parser.add_argument("--kill-bonus", type=float, default=0.0)
    parser.add_argument("--hit-bonus", type=float, default=0.0)
    parser.add_argument("--disable-motion", action="store_true")
    parser.add_argument("--random-baseline", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialise-from", type=Path,
                        help="load weights only, for a new scenario curriculum stage")
    parser.add_argument("--output", type=Path, default=Path("runs/doom-ppo.pt"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    games = [make_game(args.seed + index, args.scenario) for index in range(args.envs)]
    source_payload = None
    if args.initialise_from:
        source_payload = torch.load(args.initialise_from, map_location="cpu", weights_only=True)
    if args.resume:
        source_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
    model = DoomScorerV2(
        actions=len(DEFEND_ACTIONS),
        width=source_payload.get("width", 32) if source_payload else 32,
        rank=source_payload.get("rank", 32) if source_payload else 32,
        reads=source_payload.get("reads", 1) if source_payload else 1,
    ).to(device)
    rewards: list[float] = []
    shaped_totals: list[float] = []
    kills: list[float] = []
    armour_reached: list[bool] = []
    completed = 0
    if args.resume and args.initialise_from:
        parser.error("use only one of --resume and --initialise-from")
    if args.initialise_from:
        model.load_state_dict(source_payload["model"])
    if args.resume:
        payload = source_payload
        model.load_state_dict(payload["model"])
        rewards = list(payload.get("rewards", []))
        shaped_totals = list(payload.get("shaped_rewards", []))
        kills = list(payload.get("kills", []))
        armour_reached = list(payload.get("armour_reached", []))
        completed = int(payload.get("episodes", 0))
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    if args.random_baseline is None:
        baseline, baseline_armour = random_baseline(games, 100, args.seed + 9000)
        source = "measured"
    else:
        baseline, baseline_armour, source = args.random_baseline, 0.0, "supplied"
    print(json.dumps({"event": "random_baseline", "episodes": 100,
                      "mean_kills": baseline, "armour_rate": baseline_armour,
                      "source": source, "shaping_scale": args.shaping_scale,
                      "reward_scale": args.reward_scale,
                      "motion_channel": not args.disable_motion}), flush=True)
    training_config = {
        "entropy_start": args.entropy_start, "entropy_end": args.entropy_end,
        "entropy_decay_episodes": args.entropy_decay_episodes,
        "shaping_scale": args.shaping_scale, "motion_channel": not args.disable_motion,
        "reward_scale": args.reward_scale, "forward_bonus": args.forward_bonus,
        "clear_path_forward_bonus": args.clear_path_forward_bonus,
        "clear_path_idle_penalty": args.clear_path_idle_penalty,
        "kill_bonus": args.kill_bonus, "hit_bonus": args.hit_bonus,
        "learning_rate": args.learning_rate,
    }
    start = time.perf_counter()
    start_completed = completed
    target = completed + args.episodes
    try:
        while completed < target:
            update_count = min(args.episodes_per_update, target - completed)
            trajectories = []
            while len(trajectories) < update_count:
                count = min(args.envs, update_count - len(trajectories))
                wave = [{"observations": [], "actions": [], "log_probs": [], "values": [],
                         "rewards": [], "shaping": 0.0} for _ in range(count)]
                active = [True] * count
                previous: list[np.ndarray | None] = [None] * count
                for game in games[:count]:
                    game.new_episode()
                while any(active):
                    indices = [index for index, running in enumerate(active) if running]
                    states = [games[index].get_state() for index in indices]
                    frames = [np.ascontiguousarray(state.screen_buffer) for state in states]
                    items = [observation(frame, frame if args.disable_motion else previous[index])
                             for frame, index in zip(frames, indices)]
                    with torch.inference_mode():
                        logits, values = model(observation_tensor(items, device))
                        distribution = torch.distributions.Categorical(logits=logits)
                        actions = distribution.sample()
                        log_probs = distribution.log_prob(actions)
                        actions_cpu = actions.cpu().numpy()
                        log_probs_cpu = log_probs.cpu().numpy()
                        values_cpu = values.cpu().numpy()
                    for offset, index in enumerate(indices):
                        game = games[index]
                        action_index = int(actions_cpu[offset])
                        before = enemy_observation(states[offset])
                        kills_before = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                        hits_before = game.get_game_variable(vzd.GameVariable.HITCOUNT)
                        raw = game.make_action(
                            action_vector_for_game(action_index, DEFEND_ACTIONS, game),
                            TICS_PER_ACTION,
                        )
                        next_state = None if game.is_episode_finished() else game.get_state()
                        after = enemy_observation(next_state) if next_state is not None else None
                        bonus = shaped_reward(before, after, DEFEND_ACTIONS[action_index])
                        kills_after = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
                        hits_after = game.get_game_variable(vzd.GameVariable.HITCOUNT)
                        bonus += args.kill_bonus * max(0.0, kills_after - kills_before)
                        bonus += args.hit_bonus * max(0.0, hits_after - hits_before)
                        if args.scenario == "deadly_corridor" and DEFEND_ACTIONS[action_index] == "move forward":
                            bonus += args.forward_bonus
                        if args.scenario == "deadly_corridor" and before is None:
                            if DEFEND_ACTIONS[action_index] == "move forward":
                                bonus += args.clear_path_forward_bonus
                            else:
                                bonus -= args.clear_path_idle_penalty
                        bonus *= args.shaping_scale
                        trajectory = wave[index]
                        trajectory["observations"].append(items[offset])
                        trajectory["actions"].append(action_index)
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
                    try:
                        armour = game.get_game_variable(vzd.GameVariable.ARMOR)
                        health = game.get_game_variable(vzd.GameVariable.HEALTH)
                        trajectory["armour"] = bool(armour > 0 or (
                            args.scenario == "deadly_corridor" and health > 0
                            and game.get_episode_time() < 2100
                        ))
                    except Exception:
                        trajectory["armour"] = False
                trajectories.extend(wave)

            flat_observations, flat_actions, flat_log_probs, flat_values, flat_returns = [], [], [], [], []
            for trajectory in trajectories:
                flat_observations.extend(trajectory["observations"])
                flat_actions.extend(trajectory["actions"])
                flat_log_probs.extend(trajectory["log_probs"])
                flat_values.extend(trajectory["values"])
                flat_returns.extend(discounted_returns(trajectory["rewards"], 0.99))
                rewards.append(trajectory["raw_reward"])
                shaped_totals.append(trajectory["shaping"])
                kills.append(trajectory["kills"])
                armour_reached.append(trajectory["armour"])
            states = np.stack(flat_observations)
            actions_tensor = torch.tensor(flat_actions, device=device)
            old_log_probs = torch.tensor(flat_log_probs, device=device)
            return_tensor = torch.tensor(np.asarray(flat_returns), device=device)
            advantage = return_tensor - torch.tensor(flat_values, device=device)
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-5)
            entropy = args.entropy_end + (args.entropy_start - args.entropy_end) * max(
                0.0, 1.0 - completed / args.entropy_decay_episodes
            )
            samples = len(states)
            last_gradients = {}
            for _ in range(args.ppo_epochs):
                for batch in torch.randperm(samples, device=device).split(args.minibatch):
                    logits, predicted = model(observation_tensor(states[batch.cpu().numpy()], device))
                    distribution = torch.distributions.Categorical(logits=logits)
                    current_log_probs = distribution.log_prob(actions_tensor[batch])
                    ratio = (current_log_probs - old_log_probs[batch]).exp()
                    policy_loss = -torch.minimum(
                        ratio * advantage[batch], ratio.clamp(0.8, 1.2) * advantage[batch]
                    ).mean()
                    value_loss = F.smooth_l1_loss(predicted, return_tensor[batch])
                    entropy_value = distribution.entropy().mean()
                    loss = policy_loss + 0.5 * value_loss - entropy * entropy_value
                    optimiser.zero_grad()
                    loss.backward()
                    last_gradients = gradient_norms(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimiser.step()
            completed += update_count
            attention_entropy = attention_entropy_by_action(model, states, device)
            row = {
                "episode": completed, "mean_kills_100": float(np.mean(kills[-100:])),
                "mean_reward_100": float(np.mean(rewards[-100:])),
                "mean_shaping_100": float(np.mean(shaped_totals[-100:])),
                "armour_rate_100": float(np.mean(armour_reached[-100:])),
                "random_mean_kills": baseline,
                "episodes_per_second": (completed - start_completed) / (time.perf_counter() - start),
                "transitions": samples, "entropy_coefficient": entropy,
                "policy_entropy": float(entropy_value.detach()), "loss": float(loss.detach()),
                "attention_entropy_by_action": dict(zip(DEFEND_ACTIONS, attention_entropy)),
                "return_target_mean": float(return_tensor.mean()),
                "return_target_std": float(return_tensor.std(unbiased=False)),
                "gradient_norms": last_gradients,
            }
            print(json.dumps(row), flush=True)
            if completed % 500 < update_count or completed == target:
                milestone = args.output.with_name(f"{args.output.stem}-{completed}.pt")
                save(milestone, model, completed, rewards, shaped_totals, kills, args.seed,
                     args.scenario, baseline, armour_reached, training_config)
    finally:
        for game in games:
            game.close()
    save(args.output, model, completed, rewards, shaped_totals, kills, args.seed,
         args.scenario, baseline, armour_reached, training_config)
    print(json.dumps({
        "event": "complete", "episodes": completed,
        "mean_kills_100": float(np.mean(kills[-100:])),
        "mean_reward_100": float(np.mean(rewards[-100:])),
        "armour_rate_100": float(np.mean(armour_reached[-100:])),
        "random_mean_kills": baseline, "wall_seconds": time.perf_counter() - start,
        "checkpoint": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
