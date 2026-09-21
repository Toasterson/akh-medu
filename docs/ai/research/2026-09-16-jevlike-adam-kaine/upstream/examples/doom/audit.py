"""Screen-dependence and enemy-response audit for v2 Doom checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from diagnostics import correlation, enemy_observation, kl
from jevlike.vision import DoomScorerV2, PlainConvPolicy, benchmark, observation, observation_tensor
from environment import DEFEND_ACTIONS, TICS_PER_ACTION, action_vector_for_game, make_game


def patch_shuffle(items: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    result = items.copy().reshape(-1, 8, 15, 10, 16, 4)
    patches = result.transpose(0, 1, 3, 2, 4, 5).reshape(-1, 80, 15, 16, 4)
    shuffled = np.stack([row[generator.permutation(80)] for row in patches])
    return shuffled.reshape(-1, 8, 10, 15, 16, 4).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 120, 160, 4)


def collect(model: DoomScorerV2, seed: int, count: int, device: torch.device,
            scenario: str, action_names: tuple[str, ...], option_ids: torch.Tensor,
            greedy: bool = False):
    game = make_game(seed, scenario)
    frames, items, enemies, kills, armour = [], [], [], [], []
    previous = None
    torch.manual_seed(seed)
    try:
        while len(items) < count:
            game.new_episode()
            previous = None
            while not game.is_episode_finished() and len(items) < count:
                state = game.get_state()
                if state is None:
                    break
                frame = np.ascontiguousarray(state.screen_buffer)
                item = observation(frame, previous)
                frames.append(frame)
                items.append(item)
                enemies.append(enemy_observation(state))
                with torch.inference_mode():
                    logits, _ = model(observation_tensor([item], device), option_ids)
                    action = int(logits.argmax(-1).item() if greedy else
                                 torch.multinomial(logits.softmax(-1)[0], 1).item())
                game.make_action(action_vector_for_game(action, action_names, game), TICS_PER_ACTION)
                previous = frame
            if game.is_episode_finished():
                kills.append(float(game.get_game_variable(__import__("vizdoom").GameVariable.KILLCOUNT)))
                try:
                    armour_value = game.get_game_variable(__import__("vizdoom").GameVariable.ARMOR)
                    health = game.get_game_variable(__import__("vizdoom").GameVariable.HEALTH)
                    reached_goal = (armour_value > 0 or (
                        scenario == "deadly_corridor" and health > 0
                        and game.get_episode_time() < 2100
                    ))
                    armour.append(bool(reached_goal))
                except Exception:
                    armour.append(False)
    finally:
        game.close()
    return np.stack(frames), np.stack(items), enemies, kills, armour


def analyse(checkpoint: Path, count: int, seed: int, batch_size: int,
            device: torch.device, greedy: bool = False) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("model_type") == "plain_conv":
        option_rows = len(payload["actions"])
        model = PlainConvPolicy(actions=option_rows, width=payload.get("width", 32)).to(device)
    else:
        option_rows = payload["model"]["options.weight"].shape[0]
        model = DoomScorerV2(actions=option_rows, width=payload.get("width", 32),
                             rank=payload.get("rank", 32),
                             reads=payload.get("reads", 1)).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    all_action_names = tuple(payload["actions"])
    doom_option_ids = tuple(payload.get("doom_option_ids", range(len(all_action_names))))
    action_names = tuple(all_action_names[index] for index in doom_option_ids)
    option_ids = torch.tensor(doom_option_ids, device=device)
    frames, items, enemies, eval_kills, eval_armour = collect(
        model, seed, count, device, payload["scenario"], action_names, option_ids, greedy
    )
    blank = np.zeros_like(items)
    blank[..., 3] = 128
    average = np.rint(items.mean(0, keepdims=True)).astype(np.uint8)
    shuffled = patch_shuffle(items, seed + 1)
    groups = {"real": items, "blank": blank, "shuffled": shuffled,
              "average": np.repeat(average, len(items), 0)}
    probabilities: dict[str, list[torch.Tensor]] = {name: [] for name in groups}
    attentions = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            for name, values in groups.items():
                tensor = observation_tensor(values[start:start + batch_size], device)
                if name == "real":
                    logits, _, trace = model.forward_trace(tensor, option_ids)
                    if "attention_map" in trace:
                        attentions.append(trace["attention_map"].cpu())
                else:
                    logits, _ = model(tensor, option_ids)
                probabilities[name].append(logits.softmax(-1).cpu())
    probs = {name: torch.cat(parts) for name, parts in probabilities.items()}
    real = probs["real"]
    mean_policy = real.mean(0, keepdim=True).expand_as(real)
    attention = torch.cat(attentions) if attentions else None
    visible = np.array([index for index, enemy in enumerate(enemies) if enemy is not None])
    enemy_x = np.array([enemies[index]["x"] for index in visible]) if len(visible) else np.array([])
    near = np.array([index for index in visible if abs(enemies[index]["x"]) <= 0.15])
    off = np.array([index for index in visible if abs(enemies[index]["x"]) >= 0.35])
    left = np.array([index for index in visible if enemies[index]["x"] <= -0.15])
    right = np.array([index for index in visible if enemies[index]["x"] >= 0.15])
    action_names = list(action_names)
    attack = action_names.index("attack")
    turn_left = action_names.index("turn left")
    turn_right = action_names.index("turn right")
    p = real.numpy()

    def conditional(index: int, subset: np.ndarray) -> float | None:
        return float(p[subset, index].mean()) if len(subset) else None

    def margin(positive: float | None, negative: float | None) -> float | None:
        return None if positive is None or negative is None else positive - negative

    attack_near, attack_off = conditional(attack, near), conditional(attack, off)
    left_left, left_right = conditional(turn_left, left), conditional(turn_left, right)
    right_right, right_left = conditional(turn_right, right), conditional(turn_right, left)
    ranges = {name: {"min": float(p[:, index].min()), "max": float(p[:, index].max()),
                     "range": float(np.ptp(p[:, index])), "std": float(p[:, index].std()),
                     "mean": float(p[:, index].mean())}
              for index, name in enumerate(action_names)}
    if attention is not None:
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1) / math.log(80)
        x_axis = torch.linspace(-0.9, 0.9, 10).repeat(8)
        attention_x = (attention * x_axis).sum(-1).numpy()
        attention_report = {
            "normalised_entropy_mean": float(entropy.mean()),
            "normalised_entropy_by_button": dict(zip(action_names, entropy.mean(0).tolist())),
            "peak_mean": float(attention.max(-1).values.mean()),
            "x_enemy_correlation_by_button": {
                name: correlation(attention_x[visible, index], enemy_x)
                for index, name in enumerate(action_names)},
        }
    else:
        attention_report = None
    return {
        "checkpoint": str(checkpoint), "episodes_trained": int(payload["episodes"]),
        "action_selection": "greedy" if greedy else "sampled",
        "frames": count, "eval_episodes": len(eval_kills),
        "eval_mean_kills": float(np.mean(eval_kills)),
        "eval_armour_rate": float(np.mean(eval_armour)) if eval_armour else 0.0,
        "training_mean_kills_100": (
            float(np.mean(payload["kills"][-100:])) if payload.get("kills") else
            float(payload.get("history", [{}])[-1].get("mean_kills", float("nan")))
        ),
        "kl_nats": {"real_vs_blank": float(kl(real, probs["blank"]).mean()),
                    "real_vs_shuffled_patches": float(kl(real, probs["shuffled"]).mean()),
                    "real_vs_average_frame": float(kl(real, probs["average"]).mean()),
                    "real_vs_mean_policy": float(kl(real, mean_policy).mean())},
        "probability_ranges": ranges,
        "attention": attention_report,
        "enemy_probe": {
            "visible": int(len(visible)), "centred": int(len(near)), "off_centre": int(len(off)),
            "left": int(len(left)), "right": int(len(right)),
            "p_attack_centred": attack_near, "p_attack_off_centre": attack_off,
            "attack_margin": margin(attack_near, attack_off),
            "p_turn_left_enemy_left": left_left, "p_turn_left_enemy_right": left_right,
            "turn_left_margin": margin(left_left, left_right),
            "p_turn_right_enemy_right": right_right, "p_turn_right_enemy_left": right_left,
            "turn_right_margin": margin(right_right, right_left),
        },
        "latency_ms": {
            "cpu": benchmark(model.cpu(), items[0], torch.device("cpu"),
                             option_ids=torch.tensor(doom_option_ids)),
            "mps": benchmark(model.to("mps"), items[0], torch.device("mps"),
                             option_ids=torch.tensor(doom_option_ids)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1702)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    result = analyse(args.checkpoint, args.frames, args.seed, args.batch_size,
                     torch.device(args.device), args.greedy)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
