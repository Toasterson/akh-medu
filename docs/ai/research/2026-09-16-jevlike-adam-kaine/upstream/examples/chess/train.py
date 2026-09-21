"""Supervised chess pretraining of the shared 12-option network (chess rows 7-11 only).

The same DoomScorerV2 class and option table the joint trainer uses; this run
only moves the chess keys so the joint run can start from a model that already
reads boards.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jevlike.vision import TOTAL_OPTIONS, DoomScorerV2, observation_tensor  # noqa: E402

import data  # noqa: E402


def heldout_accuracy(model, held, device) -> dict:
    frames, keys, first = held
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        predictions = torch.cat([
            model(observation_tensor(frames[i:i + 500], device), data.CHESS_OPTION_IDS)[0]
            .argmax(-1).cpu() for i in range(0, len(frames), 500)]).numpy()
        # per-key attention entropy over the 80 patches (nats; uniform is ln 80 = 4.38)
        attention = model.forward_trace(observation_tensor(frames[:500], device), data.CHESS_OPTION_IDS)[2]["attention_map"]
        entropy = (-(attention * attention.clamp_min(1e-9).log()).sum(-1)).mean(0).cpu().tolist()
    model.train(was_training)
    per_key = {str(k): float((predictions[keys == k] == k).mean()) for k in range(5) if (keys == k).any()}
    return {"key_acc": float((predictions == keys).mean()),
            "first_key_acc": float((predictions[first] == keys[first]).mean()),
            "per_key_recall": per_key, "attention_entropy": [round(e, 3) for e in entropy]}


def save(path: Path, model: DoomScorerV2, step: int, history: list) -> None:
    torch.save({"version": 3, "model": model.state_dict(), "width": model.width, "rank": model.rank,
                "chess_option_ids": tuple(range(7, 12)), "steps": step, "history": history}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", type=Path, help="start from a 12-option checkpoint")
    parser.add_argument("--name", default="chess-pretrain")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dagger", type=float, default=0.5,
                        help="fraction of each batch drawn from runs/dagger states")
    parser.add_argument("--width", type=int, default=32, help="32 matches the Doom checkpoints")
    args = parser.parse_args()
    device = torch.device("mps")
    torch.manual_seed(0)
    model = DoomScorerV2(actions=TOTAL_OPTIONS, width=args.width, rank=args.width)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True)["model"])
    model.to(device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, args.lr, total_steps=args.steps,
                                                   pct_start=0.05)
    frames, keys, first, _, _ = data.fixed_set(4000, split="heldout")
    held = (frames, keys, first)
    first_frames, first_keys, first_first, _, _ = data.fixed_set(2000, seed=2, split="heldout",
                                                                 first_only=True)
    runs = Path(__file__).resolve().parent / "runs"
    log = runs / f"{args.name}.jsonl"
    stream = data.batches(args.batch, seed=11, workers=args.workers, dagger=args.dagger)
    history, losses, correct = [], [], []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, ids, labels, = next(stream)
        labels = labels.to(device)
        logits, _ = model(observation_tensor(x, device), ids)
        loss = F.cross_entropy(logits, labels)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimiser.step()
        schedule.step()
        losses.append(loss.detach())
        correct.append((logits.argmax(-1) == labels).float().mean().detach())
        if step % args.eval_every == 0 or step == args.steps:
            row = {"step": step, "seconds": round(time.perf_counter() - started, 1),
                   "train_loss": float(torch.stack(losses).mean()),
                   "train_acc": float(torch.stack(correct).mean()),
                   "heldout": heldout_accuracy(model, held, device),
                   "heldout_fresh_move_first_key_acc":
                       heldout_accuracy(model, (first_frames, first_keys, first_first), device)["key_acc"]}
            losses, correct = [], []
            history.append(row)
            print(json.dumps(row), flush=True)
            with log.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            save(runs / f"{args.name}.pt", model, step, history)
    stream.close()  # shuts the render worker pool so the process exits


if __name__ == "__main__":
    main()
