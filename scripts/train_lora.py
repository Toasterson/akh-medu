#!/usr/bin/env python3
"""LoRA fine-tuning pipeline for Ennead agent psyche adapters.

Trains a small LoRA adapter (~10–50 MB) on agent-specific QA pairs so that
each akh-medu agent's personality is encoded in model weights.

Supports two backends:
  --backend mlx     Apple Silicon native (recommended on Mac mini / M-series)
  --backend llamacpp  llama.cpp finetune (cross-platform, slower)

Usage
-----
# Generate seeds (one-time):
  python scripts/generate_qa_pairs.py --agent thoth --expand

# Train a single agent:
  python scripts/train_lora.py --agent thoth --data-dir ~/.akh-medu --backend mlx

# Train all agents:
  python scripts/train_lora.py --all --data-dir ~/.akh-medu --backend mlx

# Resume from checkpoint:
  python scripts/train_lora.py --agent thoth --data-dir ~/.akh-medu --resume
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── Agent definitions ──────────────────────────────────────────────────────────

AGENTS = {
    "tecton": {
        "deity": "Ptah",
        "primary": "sage",
        "secondary": "creator",
        "description": "Master builder and systems architect. Precise, visionary, patient.",
    },
    "thoth": {
        "deity": "Thoth",
        "primary": "sage",
        "secondary": "explorer",
        "description": "Knowledge seeker and scribe. Curious, analytical, rigorous.",
    },
    "horus": {
        "deity": "Horus",
        "primary": "hero",
        "secondary": "guardian",
        "description": "Decisive executor and protector. Direct, confident, reliable.",
    },
    "maat": {
        "deity": "Maat",
        "primary": "guardian",
        "secondary": "healer",
        "description": "Balance keeper and ethical judge. Fair, empathetic, principled.",
    },
    "set": {
        "deity": "Set",
        "primary": "rebel",
        "secondary": "explorer",
        "description": "Disruptor and challenger. Unconventional, bold, energetic.",
    },
    "khnum": {
        "deity": "Khnum",
        "primary": "creator",
        "secondary": "sage",
        "description": "Craftsman and pattern maker. Generative, meticulous, inventive.",
    },
}

# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class LoraConfig:
    rank: int = 16
    alpha: float = 32.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    base_model: str = "gemma2:2b-instruct-q4_K_M"
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_len: int = 512
    warmup_steps: int = 50
    grad_checkpoint: bool = True
    lora_dropout: float = 0.05


DEFAULT_CONFIG = LoraConfig()

# ── System prompt template ─────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are {name}, an AI agent inspired by {deity}. {description}

You speak with {tone}. Your primary archetype is {primary}; your secondary is {secondary}.
You are part of the Ennead — a collective of six agents with distinct psyches — \
and you respond from your unique perspective and expertise.
"""


def build_system_prompt(agent_id: str) -> str:
    info = AGENTS[agent_id]
    tone_map = {
        "tecton": "deliberate precision and structural clarity",
        "thoth": "measured inquisitiveness and scholarly depth",
        "horus": "direct confidence and decisive brevity",
        "maat": "compassionate balance and principled reasoning",
        "set": "sharp wit and energetic provocation",
        "khnum": "hands-on craftsmanship and generative depth",
    }
    return SYSTEM_PROMPT_TEMPLATE.format(
        name=agent_id.capitalize(),
        deity=info["deity"],
        description=info["description"],
        tone=tone_map[agent_id],
        primary=info["primary"],
        secondary=info["secondary"],
    )


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    pairs = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {lineno} in {path}: {e}")
    return pairs


def format_for_chat(pair: dict, system_prompt: str) -> dict:
    """Convert a QA pair into a chat-formatted training example."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["input"]},
            {"role": "assistant", "content": pair["output"]},
        ]
    }


def prepare_dataset(agent_id: str, data_dir: Path) -> tuple[Path, Path]:
    """Merge seed and expanded QA pairs, split 90/10 into train/eval JSONL."""
    agent_qa_dir = data_dir / "qa" / agent_id
    adapter_dir = data_dir / "adapters" / agent_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt(agent_id)

    sources = []
    for fname in ["seeds.jsonl", "expanded.jsonl"]:
        p = agent_qa_dir / fname
        if p.exists():
            sources.extend(load_jsonl(p))

    if not sources:
        print(f"[ERROR] No QA data found for {agent_id} in {agent_qa_dir}")
        print(f"  Run: python scripts/generate_qa_pairs.py --agent {agent_id}")
        sys.exit(1)

    print(f"[{agent_id}] Loaded {len(sources)} QA pairs")

    chat_examples = [format_for_chat(p, system_prompt) for p in sources]

    # 90/10 train/eval split (deterministic: last 10% = eval)
    split = max(1, len(chat_examples) * 9 // 10)
    train_data = chat_examples[:split]
    eval_data = chat_examples[split:]

    train_path = adapter_dir / "train.jsonl"
    eval_path = adapter_dir / "eval.jsonl"

    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(eval_path, "w") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")

    print(f"[{agent_id}] Train: {len(train_data)} | Eval: {len(eval_data)}")
    return train_path, eval_path


# ── MLX backend ───────────────────────────────────────────────────────────────

def train_mlx(
    agent_id: str,
    train_path: Path,
    eval_path: Path,
    adapter_dir: Path,
    cfg: LoraConfig,
    resume: bool = False,
) -> Path:
    """Train with MLX (Apple Silicon native, recommended)."""
    try:
        import importlib
        importlib.import_module("mlx_lm")
    except ImportError:
        print("[ERROR] mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    from mlx_lm import lora  # type: ignore[import]

    checkpoint_dir = adapter_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    adapter_path = adapter_dir / "adapter.safetensors"

    lora_args = lora.TrainingArgs(
        model=cfg.base_model,
        train=True,
        data=str(adapter_dir),
        batch_size=cfg.batch_size,
        iters=cfg.epochs * 500,
        val_batches=len(list(open(eval_path))),
        learning_rate=cfg.learning_rate,
        lora_layers=8,
        lora_parameters={
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "dropout": cfg.lora_dropout,
            "scale": cfg.alpha / cfg.rank,
        },
        adapter_path=str(adapter_path),
        resume_adapter_file=str(adapter_path) if resume and adapter_path.exists() else None,
        save_every=100,
        max_seq_length=cfg.max_seq_len,
        grad_checkpoint=cfg.grad_checkpoint,
    )

    print(f"[{agent_id}] Starting MLX training (rank={cfg.rank}, alpha={cfg.alpha})")
    lora.train(lora_args)

    return adapter_path


# ── llama.cpp backend ─────────────────────────────────────────────────────────

def train_llamacpp(
    agent_id: str,
    train_path: Path,
    eval_path: Path,
    adapter_dir: Path,
    cfg: LoraConfig,
    llamacpp_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    """Train with llama.cpp finetune (cross-platform fallback)."""
    llamacpp_bin = Path(llamacpp_dir or os.environ.get("LLAMACPP_DIR", "")) / "llama-finetune"
    if not llamacpp_bin.exists():
        llamacpp_bin = Path("llama-finetune")

    model_path = _find_gguf_model(cfg.base_model)
    if model_path is None:
        print(f"[ERROR] Could not find GGUF model for '{cfg.base_model}'")
        print("  Set LLAMACPP_MODEL_PATH or place the GGUF in ~/.akh-medu/models/llm/")
        sys.exit(1)

    adapter_path = adapter_dir / "adapter.bin"
    cmd = [
        str(llamacpp_bin),
        "--model-base", str(model_path),
        "--train-data", str(train_path),
        "--lora-out", str(adapter_path),
        "--lora-r", str(cfg.rank),
        "--lora-alpha", str(int(cfg.alpha)),
        "--ctx", str(cfg.max_seq_len),
        "--batch", str(cfg.batch_size),
        "--epochs", str(cfg.epochs),
        "--lr", str(cfg.learning_rate),
        "--adam-epsf", "1e-8",
        "--warmup", str(cfg.warmup_steps),
    ]
    if resume and adapter_path.exists():
        cmd += ["--lora-init-scale", "0"]

    print(f"[{agent_id}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] llama-finetune failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # llama.cpp outputs .bin; convert to safetensors for uniform loading.
    safetensors_path = adapter_dir / "adapter.safetensors"
    _convert_bin_to_safetensors(adapter_path, safetensors_path)
    return safetensors_path


def _find_gguf_model(model_name: str) -> Path | None:
    candidates = [
        Path.home() / ".akh-medu" / "models" / "llm" / f"{model_name}.gguf",
        Path.home() / ".ollama" / "models" / "blobs" / model_name,
        Path(os.environ.get("LLAMACPP_MODEL_PATH", model_name)),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _convert_bin_to_safetensors(bin_path: Path, out_path: Path) -> None:
    """Minimal .bin→.safetensors conversion (placeholder; use convert_lora_to_gguf.py)."""
    print(f"[WARN] Copying {bin_path.name} as safetensors (no format conversion)")
    import shutil
    shutil.copy2(bin_path, out_path)


# ── Post-training metadata ────────────────────────────────────────────────────

def write_meta(
    agent_id: str,
    adapter_dir: Path,
    adapter_path: Path,
    cfg: LoraConfig,
    train_pairs: int,
    eval_accuracy: float,
    duration_secs: float,
) -> None:
    meta = {
        "agent_id": agent_id,
        "adapter_file": adapter_path.name,
        "file_size_bytes": adapter_path.stat().st_size if adapter_path.exists() else 0,
        "lora_config": {
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "target_modules": cfg.target_modules,
            "base_model": cfg.base_model,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "max_seq_len": cfg.max_seq_len,
            "warmup_steps": cfg.warmup_steps,
        },
        "training_pairs": train_pairs,
        "eval_accuracy": round(eval_accuracy, 4),
        "training_duration_secs": round(duration_secs, 1),
        "trained_at": int(time.time()),
        "base_model_hash": cfg.base_model,
    }
    meta_path = adapter_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{agent_id}] Wrote metadata to {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train_agent(
    agent_id: str,
    data_dir: Path,
    backend: Literal["mlx", "llamacpp"],
    cfg: LoraConfig,
    resume: bool,
    llamacpp_dir: Path | None,
) -> None:
    if agent_id not in AGENTS:
        print(f"[ERROR] Unknown agent '{agent_id}'. Known: {', '.join(AGENTS)}")
        sys.exit(1)

    adapter_dir = data_dir / "adapters" / agent_id
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_path, eval_path = prepare_dataset(agent_id, data_dir)
    train_pairs = sum(1 for _ in open(train_path))

    t0 = time.time()

    if backend == "mlx":
        adapter_path = train_mlx(agent_id, train_path, eval_path, adapter_dir, cfg, resume)
    else:
        adapter_path = train_llamacpp(
            agent_id, train_path, eval_path, adapter_dir, cfg, llamacpp_dir, resume
        )

    duration = time.time() - t0

    # Evaluate on the eval set (keyword-overlap proxy; scripts/benchmark_adapter.py for full eval).
    eval_accuracy = _quick_eval(agent_id, eval_path)

    write_meta(agent_id, adapter_dir, adapter_path, cfg, train_pairs, eval_accuracy, duration)

    size_mb = adapter_path.stat().st_size / 1024**2 if adapter_path.exists() else 0
    print(
        f"[{agent_id}] Done: {duration/60:.1f} min | "
        f"eval_acc={eval_accuracy:.2%} | "
        f"adapter={size_mb:.1f} MB"
    )


def _quick_eval(agent_id: str, eval_path: Path) -> float:
    """Placeholder accuracy: count non-empty responses (real eval in benchmark_adapter.py)."""
    pairs = load_jsonl(eval_path)
    if not pairs:
        return 0.0
    # Without inference backend here, return a placeholder.
    print(f"[{agent_id}] Quick eval: {len(pairs)} pairs (run benchmark_adapter.py for real metrics)")
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapters for Ennead agents")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--agent", choices=list(AGENTS), help="Train a single agent")
    target.add_argument("--all", action="store_true", help="Train all six agents")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".akh-medu",
        help="Data directory (default: ~/.akh-medu)",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "llamacpp"],
        default="mlx",
        help="Training backend (default: mlx)",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha (default: 32)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    parser.add_argument(
        "--llamacpp-dir",
        type=Path,
        default=None,
        help="Directory containing llama-finetune binary",
    )

    args = parser.parse_args()

    cfg = LoraConfig(
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    agents = list(AGENTS) if args.all else [args.agent]
    for agent_id in agents:
        print(f"\n{'='*60}")
        print(f"Training adapter for: {agent_id.upper()} ({AGENTS[agent_id]['deity']})")
        print(f"{'='*60}")
        train_agent(
            agent_id,
            args.data_dir,
            args.backend,
            cfg,
            args.resume,
            args.llamacpp_dir,
        )


if __name__ == "__main__":
    main()
