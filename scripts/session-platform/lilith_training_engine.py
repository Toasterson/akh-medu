#!/usr/bin/env python3
"""Importable Gemma training engine for an akhrtd-managed Lilith worker.

This module owns model/optimizer mechanics only. It does not start daemons,
create sessions, choose tools, or provide a standalone experiment launcher.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_CONTEXT_TOKENS = 32_768
DEFAULT_MAX_NEW_TOKENS = 16_384
ENGINE_SCHEMA = "lilith/akhrtd-training-engine@1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _torch():
    import torch
    return torch


def _tensor_members(value: Any, seen: set[int] | None = None):
    """Yield tensor-like cache members without retaining new references to graphs."""
    seen = seen or set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if hasattr(value, "requires_grad") and hasattr(value, "grad_fn") and hasattr(value, "shape"):
        yield value
        return
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    elif hasattr(value, "__dict__"):
        children = vars(value).values()
    else:
        return
    for child in children:
        yield from _tensor_members(child, seen)


def assert_detached_kv(cache: Any) -> int:
    tensors = list(_tensor_members(cache))
    for tensor in tensors:
        if tensor.requires_grad or tensor.grad_fn is not None:
            raise RuntimeError("KV cache retained an autograd graph")
    return len(tensors)


@dataclass(frozen=True)
class PolicyIdentity:
    parent_sha256: str
    tokenizer_sha256: str
    adapter_sha256: str
    engine_sha256: str

    @property
    def digest(self) -> str:
        return sha256(canonical(asdict(self)))


@dataclass
class Trajectory:
    id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    policy_mask: list[bool]
    sampler_logps: list[float | None]
    mc_token_ids: list[list[int]]
    completion_status: str
    policy_sha256: str
    raw_text: str
    response_text: str = ""
    mc_sampler_logps: list[list[float]] = field(default_factory=list)
    token_roles: list[str] = field(default_factory=list)
    terminator_token_id: int | None = None
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prompt_length"] = len(self.prompt_token_ids)
        value["generated_text"] = self.raw_text
        value["parser_text"] = self.response_text
        return value

    def validate(self, context_limit: int) -> None:
        size = len(self.token_ids)
        if not (len(self.policy_mask) == len(self.sampler_logps) == len(self.mc_token_ids) == size):
            raise ValueError("trajectory arrays differ in length")
        if self.token_roles and len(self.token_roles) != size:
            raise ValueError("token roles differ in length")
        if self.mc_sampler_logps and len(self.mc_sampler_logps) != size:
            raise ValueError("MC sampler log probabilities differ in length")
        if self.mc_sampler_logps and any(
            len(token_ids) != len(token_logps)
            for token_ids, token_logps in zip(self.mc_token_ids, self.mc_sampler_logps)
        ):
            raise ValueError("MC token IDs and sampler log probabilities differ in length")
        if self.token_ids[:len(self.prompt_token_ids)] != self.prompt_token_ids:
            raise ValueError("prompt is not an exact token prefix")
        if size > context_limit:
            raise ValueError("trajectory exceeds engine context")
        first = len(self.prompt_token_ids)
        if any(self.policy_mask[:first]) or not all(self.policy_mask[first:]):
            raise ValueError("policy mask must select exactly generated tokens")
        if any(value is not None for value in self.sampler_logps[:first]):
            raise ValueError("prompt tokens cannot carry action log probabilities")


@dataclass(frozen=True)
class ReplayReceipt:
    trajectory_id: str
    tokens_scored: int
    sum_loss: float
    cache_tensors_checked: int
    policy_sha256: str
    max_sampler_logp_drift: float


@dataclass(frozen=True)
class UpdateReceipt:
    kind: str
    optimizer_steps: int
    records: int
    tokens: int
    sum_loss: float
    policy_before: str
    policy_after: str


class GemmaTrainingEngine:
    """One pinned Gemma parent and last-layer q-projection rank-2 LoRA."""

    def __init__(self, model: Any, tokenizer: Any, parent_sha256: str,
                 tokenizer_sha256: str, target_module: str,
                 context_tokens: int = DEFAULT_CONTEXT_TOKENS,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 prefill_chunk_tokens: int = 1,
                 cache_offloading: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.parent_sha256 = parent_sha256
        self.tokenizer_sha256 = tokenizer_sha256
        self.target_module = target_module
        self.context_tokens = context_tokens
        self.max_new_tokens = max_new_tokens
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.cache_offloading = cache_offloading
        if prefill_chunk_tokens <= 0:
            raise ValueError("prefill chunk must be positive")
        if max_new_tokens <= 0 or context_tokens <= max_new_tokens:
            raise ValueError("context must leave room for a prompt")
        self._assert_trainable_surface()
        # Checkpointing/input-gradient hooks would retain graphs across cached steps.
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model, "disable_input_require_grads"):
            model.disable_input_require_grads()
        model.eval()

    @classmethod
    def load(cls, parent: Path, *, device: str = "cuda:0",
             context_tokens: int = DEFAULT_CONTEXT_TOKENS,
             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
             expected_parent_sha256: str, expected_config_sha256: str,
             expected_tokenizer_sha256: str, target_module: str):
        torch = _torch()
        from transformers import AutoTokenizer, Gemma4UnifiedForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        def file_sha(name: str) -> str:
            digest = hashlib.sha256()
            with (Path(parent) / name).open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        if file_sha("model.safetensors") != expected_parent_sha256:
            raise RuntimeError("parent weight hash mismatch")
        if file_sha("config.json") != expected_config_sha256:
            raise RuntimeError("parent config hash mismatch")
        if file_sha("tokenizer.json") != expected_tokenizer_sha256:
            raise RuntimeError("tokenizer hash mismatch")
        tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True)
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            parent, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": device}, low_cpu_mem_usage=True,
            attn_implementation="sdpa")
        model.config.use_cache = True
        model.config.text_config.use_cache = True
        model = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=2, lora_dropout=0.0,
            target_modules=[target_module], bias="none"))
        return cls(model, tokenizer, expected_parent_sha256,
                   expected_tokenizer_sha256, target_module,
                   context_tokens, max_new_tokens)

    def snapshot_adapter(self) -> dict[str, Any]:
        return {name: parameter.detach().cpu().clone() for name, parameter in self.model.named_parameters()
                if parameter.requires_grad}

    def restore_adapter(self, state: dict[str, Any]) -> None:
        torch = _torch()
        expected = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            raise ValueError("adapter snapshot keys differ")
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    if state[name].shape != parameter.shape or state[name].dtype != parameter.dtype:
                        raise ValueError("adapter snapshot tensor differs: " + name)
                    parameter.copy_(state[name].to(parameter.device))

    @staticmethod
    def blend_adapter_states(left: dict[str, Any], right: dict[str, Any], alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0 or set(left) != set(right):
            raise ValueError("incompatible adapter blend")
        result = {}
        for name in sorted(left):
            if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
                raise ValueError("adapter blend tensor differs: " + name)
            result[name] = left[name].mul(alpha).add(right[name], alpha=1.0-alpha)
        return result

    def save_adapter(self, path: Path) -> str:
        self.model.save_pretrained(path, safe_serialization=True)
        artifact = Path(path) / "adapter_model.safetensors"
        digest = hashlib.sha256()
        with artifact.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def contract(self) -> dict[str, Any]:
        return {"schema": ENGINE_SCHEMA, "context_tokens": self.context_tokens,
                "max_new_tokens": self.max_new_tokens, "thinking": True,
                "prefill_chunk_tokens": self.prefill_chunk_tokens,
                "cache_offloading": self.cache_offloading,
                "cache_transfer": "synchronous_caller_stream" if self.cache_offloading else "resident",
                "cache": "explicit hybrid layers, detached KV",
                "update": "one optimizer step after full eligible bank",
                "target_module": self.target_module, "rank": 2}

    def _assert_trainable_surface(self) -> None:
        names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not names or any("lora_" not in name or self.target_module not in name for name in names):
            raise RuntimeError("trainable surface is not exclusively the selected LoRA projection")

    def _adapter_state_hash(self) -> str:
        torch = _torch()
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            if parameter.requires_grad:
                value = parameter.detach().cpu().contiguous()
                digest.update(name.encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def policy_identity(self) -> PolicyIdentity:
        return PolicyIdentity(self.parent_sha256, self.tokenizer_sha256,
                              self._adapter_state_hash(),
                              hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def render_messages(self, messages: Sequence[dict[str, Any]], *,
                        tools: Sequence[dict[str, Any]] | None = None,
                        add_generation_prompt: bool = True) -> tuple[str, list[int]]:
        kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt,
                  "enable_thinking": True}
        if tools is not None:
            kwargs["tools"] = list(tools)
        rendered = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        return rendered, self.tokenizer.encode(rendered, add_special_tokens=False)

    def render_prompt(self, text: str) -> list[int]:
        return self.render_messages([{"role": "user", "content": text}])[1]

    def _generation_tokens(self) -> tuple[set[int], set[int]]:
        config = getattr(self.model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None) or self.model.config.eos_token_id
        suppress = getattr(config, "suppress_tokens", None) or []
        return set(eos if isinstance(eos, (list, tuple, set)) else [eos]), set(suppress)

    def _cache(self):
        from transformers import Cache
        from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
        config = self.model.config.get_text_config()
        layer_types = list(config.layer_types)
        shared = getattr(config, "num_kv_shared_layers", 0)
        if shared:
            layer_types = layer_types[:-shared]
        # Transformers 5.10 slices [:-0] for zero shared layers, silently
        # replacing the hybrid layout with lazily allocated full-attention KV.
        # Construct the public cache layers explicitly, preserving the model config.
        layers = []
        for kind in layer_types:
            if kind == "sliding_attention":
                layers.append(DynamicSlidingWindowLayer(config=config))
            elif kind == "full_attention":
                layers.append(DynamicLayer())
            else:
                raise ValueError("unsupported training cache layer: " + kind)
        if not self.cache_offloading:
            return Cache(layers=layers)

        class SynchronousOffloadedCache(Cache):
            """Keep cache transfers on the caller stream, with host copies ready.

            The installed asynchronous prefetch path races cross-stream tensor
            lifetimes on ROCm: repeated identical forwards produce different
            logits. Blocking transfers also prevent reading host KV before its
            preceding device-to-host copy has completed.
            """

            def update(self, key_states, value_states, layer_idx, *args, **kwargs):
                layer = self.layers[layer_idx]
                if layer.is_initialized:
                    layer.keys = layer.keys.to(key_states.device, non_blocking=False)
                    layer.values = layer.values.to(value_states.device, non_blocking=False)
                keys, values = layer.update(key_states, value_states, *args, **kwargs)
                layer.keys = layer.keys.to("cpu", non_blocking=False)
                layer.values = layer.values.to("cpu", non_blocking=False)
                return keys, values

        return SynchronousOffloadedCache(layers=layers)

    def _forward(self, token_ids: Sequence[int], cache: Any, start: int, grad: bool):
        torch = _torch()
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.model.device)
        positions = torch.arange(start, start + len(token_ids), device=self.model.device)[None, :]
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            output = self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                                position_ids=positions, cache_position=positions[0],
                                logits_to_keep=1)
        # Gemma rebuilds shared_kv_states inside each forward: current non-sharing
        # layers publish their cache-inclusive K/V for later sharing layers.
        assert_detached_kv(cache)
        return output.logits.float()[0, -1]

    def _prefill_no_grad(self, token_ids: Sequence[int], cache: Any, start: int = 0):
        """Bound activation memory while building an exact causal KV cache."""
        logits = None
        for offset in range(0, len(token_ids), self.prefill_chunk_tokens):
            chunk = token_ids[offset:offset + self.prefill_chunk_tokens]
            logits = self._forward(chunk, cache, start + offset, grad=False)
        if logits is None:
            raise ValueError("cannot prefill an empty sequence")
        return logits


    def generate(self, *, prompt: str | None = None,
                 prompt_ids: Sequence[int] | None = None,
                 messages: Sequence[dict[str, Any]] | None = None,
                 tools: Sequence[dict[str, Any]] | None = None,
                 max_new_tokens: int | None = None,
                 token_observer: Callable[[dict[str, Any]], None] | None = None,
                 mc_samples: int = 8, seed: int = 73) -> Trajectory:
        torch = _torch()
        from lilith_rl import sample_policy_tokens
        supplied = sum(value is not None for value in (prompt, prompt_ids, messages))
        if supplied != 1:
            raise ValueError("provide exactly one of prompt, prompt_ids, or messages")
        if messages is not None:
            prompt_tokens = self.render_messages(messages, tools=tools)[1]
        else:
            prompt_tokens = self.render_prompt(prompt) if prompt is not None else list(prompt_ids or [])
        budget = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if budget > self.max_new_tokens or len(prompt_tokens) + budget > self.context_tokens:
            raise ValueError("requested generation exceeds frozen engine bounds")
        if len(prompt_tokens) < 1:
            raise ValueError("prompt must contain a token")
        self.model.eval()
        cache = self._cache()
        logits = self._prefill_no_grad(prompt_tokens, cache, 0)
        ids = list(prompt_tokens)
        logps: list[float | None] = [None] * len(ids)
        draws: list[list[int]] = [[] for _ in ids]
        draw_logps: list[list[float]] = [[] for _ in ids]
        action_rng = torch.Generator(device="cpu").manual_seed(seed)
        aux_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
        eos_ids, official_suppress_tokens = self._generation_tokens()
        status = "max_new_tokens"
        terminator = None
        for step in range(budget):
            draw = sample_policy_tokens(logits.cpu(), action_rng, aux_rng, mc_samples, 1.0)
            token = int(draw["token_id"])
            ids.append(token); logps.append(float(draw["sampler_logp"])); draws.append(list(draw["mc_token_ids"])); draw_logps.append([float(value) for value in draw["mc_sampler_logps"]])
            event = {"step": step, "token_id": token, "sampler_logp": logps[-1],
                     "absolute_position": len(ids) - 1}
            if token_observer is not None:
                token_observer(event)
            if token in eos_ids:
                terminator = token
                status = "tool_handoff" if token == 50 else "completed"
                break
            if len(ids) >= self.context_tokens:
                status = "context_limit"
                break
            logits = self._forward([token], cache, len(ids) - 1, grad=False)
        policy = self.policy_identity().digest
        generated = ids[len(prompt_tokens):]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        response_ids = generated[:-1] if terminator is not None else generated
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        result = Trajectory(
            id=sha256(canonical({"policy": policy, "prompt": prompt_tokens, "tokens": generated})),
            prompt_token_ids=list(prompt_tokens), token_ids=ids,
            policy_mask=[False] * len(prompt_tokens) + [True] * len(generated),
            sampler_logps=logps, mc_token_ids=draws,
            completion_status=status, policy_sha256=policy, raw_text=raw,
            response_text=response_text, mc_sampler_logps=draw_logps,
            token_roles=["prompt"] * len(prompt_tokens) + ["assistant"] * len(generated),
            terminator_token_id=terminator,
            metadata={"seed": seed, "mc_samples": mc_samples, "thinking": True,
                      "sampler_policy": "full_support_temperature_1",
                      "official_suppress_tokens_recorded_not_applied": sorted(official_suppress_tokens),
                      "eos_token_ids": sorted(eos_ids), "tools_rendered": tools is not None})
        result.validate(self.context_tokens)
        return result

    def _prefill_replay(self, token_ids: Sequence[int], first_action: int):
        cache = self._cache()
        # Feed through first_action-2. The next grad forward feeds token i-1 and
        # scores action i, avoiding the easy one-token shift error.
        prefix_end = first_action - 1
        if prefix_end > 0:
            self._prefill_no_grad(token_ids[:prefix_end], cache, 0)
        return cache, prefix_end

    def accumulate_trajectory(self, trajectory: Trajectory,
                              token_loss: Callable[[Any, Any, Any, Any, float], Any]) -> ReplayReceipt:
        torch = _torch()
        trajectory.validate(self.context_tokens)
        current_policy = self.policy_identity().digest
        if trajectory.policy_sha256 != current_policy:
            raise RuntimeError("trajectory policy binding differs from current adapter")
        first = next((i for i, flag in enumerate(trajectory.policy_mask) if flag), None)
        if first is None or first < 1:
            raise ValueError("trajectory has no scoreable action")
        cache, absolute = self._prefill_replay(trajectory.token_ids, first)
        total, checked, count, max_drift = 0.0, 0, 0, 0.0
        for action_index in range(first, len(trajectory.token_ids)):
            if absolute != action_index - 1:
                raise RuntimeError("cached replay position drift")
            logits = self._forward([trajectory.token_ids[action_index - 1]], cache, absolute, grad=True)
            checked += assert_detached_kv(cache)
            logp = torch.log_softmax(logits, dim=-1)
            action = torch.tensor(trajectory.token_ids[action_index], device=logp.device)
            auxiliary = torch.tensor(trajectory.mc_token_ids[action_index], dtype=torch.long, device=logp.device)
            old_action = trajectory.sampler_logps[action_index]
            old_aux = trajectory.mc_sampler_logps[action_index] if trajectory.mc_sampler_logps else []
            if old_action is not None:
                max_drift = max(max_drift, abs(float(logp[action].detach().cpu()) - float(old_action)))
            if old_aux:
                replay_aux = logp[auxiliary].detach().cpu().tolist()
                max_drift = max(max_drift, max(
                    abs(float(current) - float(previous))
                    for current, previous in zip(replay_aux, old_aux)
                ))
            reward = 0.0 if trajectory.reward is None else float(trajectory.reward)
            loss = token_loss(logp[action], logp[auxiliary], old_action, old_aux, reward)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite replay loss")
            total += float(loss.detach().cpu())
            loss.backward()
            absolute += 1
            count += 1
        self._assert_trainable_surface()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                raise RuntimeError("missing or nonfinite LoRA gradient: " + name)
        return ReplayReceipt(trajectory.id, count, total, checked, current_policy, max_drift)

    def score_trajectory(self, trajectory: Trajectory,
                         token_loss: Callable[[Any, Any, Any, Any, float], Any]) -> ReplayReceipt:
        """Accumulate exact cached-replay gradients without taking an optimizer step."""
        return self.accumulate_trajectory(trajectory, token_loss)

    def online_update(self, trajectories: Iterable[Trajectory], *, learning_rate: float,
                      token_loss: Callable[[Any, Any, Any, Any, float], Any],
                      max_logp_drift: float = 1e-4) -> UpdateReceipt:
        torch = _torch()
        records = list(trajectories)
        if not records or any(row.reward is None for row in records):
            raise ValueError("online update requires externally rewarded trajectories")
        if any(row.completion_status not in {"completed", "tool_handoff"} for row in records):
            raise ValueError("online update rejects incomplete trajectories")
        before = self.policy_identity().digest
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=learning_rate, weight_decay=0)
        optimizer.zero_grad(set_to_none=True)
        receipts = [self.accumulate_trajectory(row, token_loss) for row in records]
        if any(receipt.max_sampler_logp_drift > max_logp_drift for receipt in receipts):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("sampler/replay log-probability drift exceeds tolerance")
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        after = self.policy_identity().digest
        if before == after:
            raise RuntimeError("optimizer did not change adapter")
        return UpdateReceipt("online_rl", 1, len(records), sum(r.tokens_scored for r in receipts),
                             sum(r.sum_loss for r in receipts), before, after)

    def sft_step(self, examples: Iterable[tuple[Sequence[int], Sequence[int]]], *,
                 learning_rate: float) -> UpdateReceipt:
        torch = _torch()
        rows = [(list(prompt), list(answer)) for prompt, answer in examples]
        if not rows:
            raise ValueError("SFT requires examples")
        before = self.policy_identity().digest
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                      lr=learning_rate, weight_decay=0)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0; tokens = 0
        for index, (prompt, answer) in enumerate(rows):
            if not answer or len(prompt) + len(answer) > self.context_tokens:
                raise ValueError("SFT example exceeds bounds or has no target")
            fake = Trajectory(id=str(index), prompt_token_ids=prompt, token_ids=prompt + answer,
                policy_mask=[False] * len(prompt) + [True] * len(answer),
                sampler_logps=[None] * (len(prompt) + len(answer)),
                mc_token_ids=[[] for _ in prompt + answer], completion_status="teacher_forced",
                policy_sha256=before, raw_text="",
                token_roles=["prompt"] * len(prompt) + ["assistant"] * len(answer))
            def nll(action_logp, _aux, _old, _old_aux, _reward):
                return -action_logp / len(rows)
            receipt = self.accumulate_trajectory(fake, nll)
            total += receipt.sum_loss; tokens += receipt.tokens_scored
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        after = self.policy_identity().digest
        return UpdateReceipt("sft", 1, len(rows), tokens, total, before, after)
