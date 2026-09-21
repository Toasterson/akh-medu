"""Paced evaluation, timing, trace and raw footage for Doom scorer v2."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import vizdoom as vzd
from PIL import Image

from jevlike.vision import DoomScorerV2, benchmark, observation, observation_tensor
from environment import TICS_PER_ACTION, TICS_PER_SECOND, action_vector_for_game, make_game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--game-seconds", type=float, default=60.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--capture-resolution", choices=("160x120", "640x480"),
                        default="640x480")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--training-wall-seconds", type=float, default=0.0)
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    all_action_names = tuple(payload["actions"])
    doom_option_ids = tuple(payload.get("doom_option_ids", range(len(all_action_names))))
    action_names = tuple(all_action_names[index] for index in doom_option_ids)
    option_rows = payload["model"]["options.weight"].shape[0]
    model = DoomScorerV2(actions=option_rows, width=payload.get("width", 32),
                         rank=payload.get("rank", 32))
    model.load_state_dict(payload["model"])
    device = torch.device(args.device)
    seed = payload.get("seed", 59)
    torch.manual_seed(seed + 1000)
    model.to(device).eval()
    option_ids = torch.tensor(doom_option_ids, device=device)
    game = make_game(seed + 1000, payload["scenario"], args.capture_resolution)
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.output, fps=30, codec="libx264", quality=8)
    rewards, kills, ratios, latencies = [], [], [], []
    decisions = []
    video_frames = video_tics = total_steps = 0
    first_item = None
    limit = round(args.game_seconds * TICS_PER_SECOND / TICS_PER_ACTION)
    try:
        for episode in range(1, args.episodes + 1):
            game.new_episode()
            previous = None
            episode_start = time.perf_counter()
            episode_reward = 0.0
            steps = 0
            while not game.is_episode_finished() and total_steps < limit:
                state = game.get_state()
                if state is None:
                    break
                frame = np.ascontiguousarray(state.screen_buffer)
                policy_frame = np.asarray(
                    Image.fromarray(frame).resize((160, 120), Image.Resampling.BILINEAR)
                )
                item = observation(policy_frame, previous)
                first_item = item if first_item is None else first_item
                tick = time.perf_counter()
                with torch.inference_mode():
                    logits, _, tensors = model.forward_trace(
                        observation_tensor([item], device), option_ids
                    )
                    if device.type == "mps":
                        torch.mps.synchronize()
                    if args.random_policy:
                        action = int(torch.randint(len(action_names), (), device=device).item())
                    elif args.greedy:
                        action = int(logits.argmax(-1).item())
                    else:
                        action = int(torch.multinomial(logits.softmax(-1)[0], 1).item())
                latency = (time.perf_counter() - tick) * 1000
                reward = game.make_action(
                    action_vector_for_game(action, action_names, game), TICS_PER_ACTION
                )
                episode_reward += float(reward)
                latencies.append(latency)
                if args.trace:
                    buffer = io.BytesIO()
                    Image.fromarray(frame).save(buffer, format="JPEG", quality=88)
                    all_attention = tensors["attention_map"][0].reshape(
                        len(action_names), model.grid_rows, model.grid_columns).cpu().tolist()
                    decisions.append({
                        "frame": base64.b64encode(buffer.getvalue()).decode(),
                        "episode": episode, "action": action_names[action],
                        "action_index": action,
                        "probabilities": tensors["probabilities"][0].cpu().tolist(),
                        "attention": tensors["attention_map"][0, action].reshape(
                            model.grid_rows, model.grid_columns).cpu().tolist(),
                        "activations": {
                            "query": tensors["query_matrix"][0].cpu().tolist(),
                            "key": tensors["key_matrix"][0, ::2, ::2].T.cpu().tolist(),
                            "value": tensors["value_matrix"][0, ::2, ::2].T.cpu().tolist(),
                            "attention": all_attention,
                            "scores": [tensors["logits_matrix"][0].cpu().tolist()],
                            "probabilities": [tensors["probabilities"][0].cpu().tolist()],
                        },
                        "latency_ms": latency, "reward": float(reward),
                        "running_reward": episode_reward,
                        "visible_objects": len(state.labels or []),
                    })
                video_tics += TICS_PER_ACTION
                target_frames = round(video_tics * 30 / TICS_PER_SECOND)
                if writer:
                    while video_frames < target_frames:
                        writer.append_data(frame)
                        video_frames += 1
                elapsed = time.perf_counter() - tick
                time.sleep(max(0.0, TICS_PER_ACTION / TICS_PER_SECOND - elapsed))
                previous = policy_frame
                steps += 1
                total_steps += 1
            rewards.append(float(game.get_total_reward()))
            kills.append(float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)))
            ratios.append((time.perf_counter() - episode_start)
                          / max(1e-9, steps * TICS_PER_ACTION / TICS_PER_SECOND))
            if total_steps >= limit:
                break
    finally:
        if writer:
            writer.close()
        game.close()
    timings = {"cpu_ms": benchmark(
        model.cpu(), first_item, torch.device("cpu"),
        option_ids=torch.tensor(doom_option_ids)
    )}
    if torch.backends.mps.is_available():
        timings["mps_ms"] = benchmark(
            model.to("mps"), first_item, torch.device("mps"),
            option_ids=torch.tensor(doom_option_ids)
        )
    result = {
        "episodes": len(rewards), "mean_reward": float(np.mean(rewards)),
        "mean_kills": float(np.mean(kills)),
        "mean_wall_game_ratio": float(np.mean(ratios)),
        "online_inference_ms": float(np.mean(latencies)), **timings,
        "video": str(args.output) if args.output else None,
    }
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        args.trace.write_text(json.dumps({
            **result, "trained_episodes": payload["episodes"],
            "training_rewards": payload.get("rewards", []),
            "training_wall_seconds": args.training_wall_seconds,
            "actions": action_names, "grid_rows": model.grid_rows,
            "grid_columns": model.grid_columns, "decisions": decisions,
        }))
        result["trace"] = str(args.trace)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
