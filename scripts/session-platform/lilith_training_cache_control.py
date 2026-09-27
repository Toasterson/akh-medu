#!/usr/bin/env python3
"""Mechanical resident-versus-offloaded KV cache control for pinned Lilith Gemma."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence


SCHEMA = "lilith/training-cache-control@1"
TOLERANCE = 1e-4
PINNED_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_invariants(cache: Any) -> dict[str, Any]:
    from lilith_training_engine import assert_detached_kv

    layers = []
    for index, layer in enumerate(cache.layers):
        stored = 0 if not layer.is_initialized else int(layer.keys.shape[-2])
        layers.append({
            "index": index,
            "type": type(layer).__name__,
            "is_sliding": bool(getattr(layer, "is_sliding", False)),
            "sequence_length": int(layer.get_seq_length()),
            "stored_length": stored,
            "device": None if not layer.is_initialized else str(layer.keys.device),
        })
    return {
        "layer_count": len(cache.layers),
        "detached_tensors_checked": assert_detached_kv(cache),
        "layers": layers,
    }


def _seed_nonzero_adapter(engine: Any, seed: int = 73) -> list[str]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = []
    with torch.no_grad():
        for name, parameter in engine.model.named_parameters():
            if parameter.requires_grad:
                value = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
                value.uniform_(-1e-4, 1e-4, generator=generator)
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                names.append(name)
    if len(names) != 2 or not all(any(kind in name for name in names) for kind in ("lora_A", "lora_B")):
        raise RuntimeError("control requires exactly one nonzero LoRA A/B pair")
    return names


def run_control(engine: Any, output: Path | str,
                positions: Sequence[int] = (1023, 1024, 1025)) -> dict[str, Any]:
    """Run the cache control without changing adapter weights or taking an optimizer step."""
    import torch
    import transformers
    import lilith_training_engine as engine_module

    output = Path(output)
    positions = tuple(int(position) for position in positions)
    if not positions or tuple(sorted(set(positions))) != positions or positions[0] < 1:
        raise ValueError("positions must be unique positive integers in ascending order")
    if max(positions) > engine.context_tokens:
        raise ValueError("control positions exceed the engine context bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    initial_policy = engine.policy_identity()
    first_parameter = next(engine.model.parameters())
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "phase": "initializing",
        "positions": list(positions),
        "tolerance": TOLERANCE,
        "reference": "GPU-resident tokenwise cache",
        "full_bf16_is_diagnostic_only": True,
        "optimizer_steps": 0,
        "engine_sha256": _sha256(Path(engine_module.__file__)),
        "control_sha256": _sha256(Path(__file__)),
        "versions": {"python": sys.version, "torch": torch.__version__,
                     "transformers": transformers.__version__, "hip": torch.version.hip},
        "runtime": {
            "device": str(first_parameter.device),
            "device_name": (torch.cuda.get_device_name(first_parameter.device)
                            if first_parameter.device.type == "cuda" else None),
            "dtype": str(first_parameter.dtype),
            "attention_backend": engine.model.config.get_text_config()._attn_implementation,
        },
        "initial_policy_identity": {**asdict(initial_policy), "digest": initial_policy.digest},
        "contract": engine.contract(),
        "variants": {},
    }

    # Exclusive creation makes every result an immutable, uniquely named evidence target.
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")

    def persist() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n")

    snapshot = engine.snapshot_adapter()
    original_offloading = engine.cache_offloading
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model = engine.model
    tokens = [2] + [100 + index % 40 for index in range(max(positions) - 1)]
    selected_token = 101
    logits: dict[str, dict[int, Any]] = {}
    gradients: dict[str, dict[str, Any]] = {}

    try:
        report["trainable_names"] = _seed_nonzero_adapter(engine)
        report["phase"] = "tokenwise_variants"
        persist()
        for variant, offloading in (("resident", False), ("synchronous_offload", True)):
            engine.cache_offloading = offloading
            cache = engine._cache()
            model.zero_grad(set_to_none=True)
            variant_logits: dict[int, Any] = {}
            loss = None
            for index, token in enumerate(tokens):
                position = index + 1
                selected = position in positions
                current = engine._forward([token], cache, index, grad=selected)
                if selected:
                    variant_logits[position] = current.detach().float().cpu()
                    term = -torch.log_softmax(current, dim=-1)[selected_token] / len(positions)
                    loss = term if loss is None else loss + term
            if loss is None:
                raise RuntimeError("no boundary losses were selected")
            mean_loss = float(loss.detach().cpu())
            loss.backward()
            variant_gradients = {
                name: parameter.grad.detach().float().cpu().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            if set(variant_gradients) != set(report["trainable_names"]):
                raise RuntimeError("control did not produce every trainable gradient")
            gradient_norms = {name: float(value.norm()) for name, value in variant_gradients.items()}
            if not all(value > 0 for value in gradient_norms.values()):
                raise RuntimeError("control requires nonzero LoRA A and B gradients")
            logits[variant] = variant_logits
            gradients[variant] = variant_gradients
            report["variants"][variant] = {
                "mean_selected_token_loss": mean_loss,
                "gradient_norms": gradient_norms,
                "cache": _cache_invariants(cache),
            }
            cache_report = report["variants"][variant]["cache"]
            text_config = model.config.get_text_config()
            expected_layer_count = text_config.num_hidden_layers - text_config.num_kv_shared_layers
            if cache_report["layer_count"] != expected_layer_count:
                raise RuntimeError("cache layer count differs from the model configuration")
            for layer in cache_report["layers"]:
                expected_stored = (
                    min(max(positions), text_config.sliding_window - 1)
                    if layer["is_sliding"] else max(positions)
                )
                if layer["sequence_length"] != max(positions) or layer["stored_length"] != expected_stored:
                    raise RuntimeError("cache sequence or stored length invariant failed")
                expected_device = "cpu" if offloading else str(model.device)
                if layer["device"] != expected_device:
                    raise RuntimeError("cache storage device invariant failed")
            persist()
            del cache, current, loss, term
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        report["resident_vs_synchronous"] = {
            "logit_max_abs": {
                str(position): float((logits["resident"][position] -
                                      logits["synchronous_offload"][position]).abs().max())
                for position in positions
            },
            "loss_abs": abs(
                report["variants"]["resident"]["mean_selected_token_loss"] -
                report["variants"]["synchronous_offload"]["mean_selected_token_loss"]
            ),
            "gradient_max_abs": {
                name: float((gradients["resident"][name] -
                             gradients["synchronous_offload"][name]).abs().max())
                for name in gradients["resident"]
            },
        }

        report["phase"] = "full_bf16_diagnostic"
        report["full_bf16_diagnostic"] = {}
        persist()
        device = model.device
        for position in positions:
            ids = torch.tensor([tokens[:position]], dtype=torch.long, device=device)
            position_ids = torch.arange(position, device=device)[None, :]
            with torch.no_grad():
                full = model(input_ids=ids, use_cache=False, position_ids=position_ids,
                             logits_to_keep=1).logits.float()[0, -1].cpu()
            difference = full - logits["resident"][position]
            report["full_bf16_diagnostic"][str(position)] = {
                "max_abs": float(difference.abs().max()),
                "mean_abs": float(difference.abs().mean()),
            }
            persist()

        report["phase"] = "generate_replay"
        engine.cache_offloading = True
        model.zero_grad(set_to_none=True)
        trajectory = engine.generate(
            prompt_ids=[2] + list(range(100, 116)), max_new_tokens=4,
            mc_samples=2, seed=73,
        )
        receipt = engine.accumulate_trajectory(
            trajectory,
            lambda action_logp, auxiliary_logps, old_action, old_aux, reward: -action_logp,
        )
        replay_gradients = {
            name: float(parameter.grad.detach().float().norm())
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        report["generate_replay"] = {
            "trajectory_id": trajectory.id,
            "completion_status": trajectory.completion_status,
            "generated_tokens": len(trajectory.token_ids) - len(trajectory.prompt_token_ids),
            "policy_sha256": trajectory.policy_sha256,
            "receipt_policy_sha256": receipt.policy_sha256,
            "tokens_scored": receipt.tokens_scored,
            "max_sampler_logp_drift": receipt.max_sampler_logp_drift,
            "cache_tensors_checked": receipt.cache_tensors_checked,
            "gradient_norms": replay_gradients,
        }
        comparison = report["resident_vs_synchronous"]
        hard_values = (
            list(comparison["logit_max_abs"].values()) +
            [comparison["loss_abs"]] +
            list(comparison["gradient_max_abs"].values())
        )
        passed = (
            max(hard_values) <= TOLERANCE and
            trajectory.policy_sha256 == receipt.policy_sha256 and
            receipt.max_sampler_logp_drift <= TOLERANCE and
            receipt.tokens_scored > 0 and
            all(value > 0 for value in replay_gradients.values())
        )
        report.update(
            status="passed" if passed else "failed_equivalence",
            phase="finished",
            peak_gpu_bytes=(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0),
        )
        persist()
        return report
    except BaseException as error:
        report.update(status="error", phase="failed",
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        persist()
        raise
    finally:
        engine.restore_adapter(snapshot)
        engine.cache_offloading = original_offloading
        model.zero_grad(set_to_none=True)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        restored_policy = engine.policy_identity()
        identity_restored = restored_policy == initial_policy
        report["restored_policy_identity"] = {
            **asdict(restored_policy), "digest": restored_policy.digest,
        }
        report["initial_policy_identity_restored"] = identity_restored
        if not identity_restored:
            report.update(status="error", phase="restore_failed",
                          error="PolicyIdentity differs after adapter restoration")
        persist()


def main() -> None:
    from lilith_adapter_mechanics import PARENT_CONFIG_SHA256, PARENT_SHA256, TARGET
    from lilith_training_engine import GemmaTrainingEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=[1023, 1024, 1025])
    args = parser.parse_args()
    engine = GemmaTrainingEngine.load(
        args.parent,
        expected_parent_sha256=PARENT_SHA256,
        expected_config_sha256=PARENT_CONFIG_SHA256,
        expected_tokenizer_sha256=PINNED_TOKENIZER_SHA256,
        target_module=TARGET,
    )
    result = run_control(engine, args.output, args.positions)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
