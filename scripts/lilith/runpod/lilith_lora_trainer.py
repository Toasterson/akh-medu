#!/usr/bin/env python3
"""Full-depth LoRA trainer for Lilith's pinned Gemma 4 12B parent (ADR 116 Decision 7).

Surface: LoRA (rank 16 default, rank 32 ablation) on every language-model attention
and MLP projection. Vision/audio embedders, embeddings and the tied LM head stay
frozen. Two memory traps apply on every device, the H100 included
(docs/ai/plans/2026-09-27-lilith-local-gpu-qualification.md §2):

* the 262,144-token vocabulary: full logits at 32K are 17 GB in bf16, so the loss
  is computed by ``chunked_token_nll`` which recomputes each logits chunk in the
  backward pass and never materializes the full [T, V] tensor;
* head_dim 512 in the 8 global layers exceeds FlashAttention-2, and naive scores
  for one global layer at 32K are 34 GB, so attention is ``ChunkedAttention``:
  query-chunked, causal/sliding-window aware, saving only (q, k, v, out, lse) and
  recomputing probabilities in backward (the ADR 107 recompute-backward approach).

Learning follows ADR 110 rule 7: weighted SFT on corrected siblings (weights in
[0, 1], never negative) plus a sibling-contrast preference loss (DPO-style).
No scalar-judge gradient is implemented here.

Heavy imports (torch, transformers, peft, bitsandbytes) are deferred so the
module imports on machines without them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Literal, Sequence

SESSION_PLATFORM = Path(__file__).resolve().parents[2] / 'session-platform'
if str(SESSION_PLATFORM) not in sys.path:
    sys.path.insert(0, str(SESSION_PLATFORM))

from lilith_adapter_mechanics import (  # noqa: E402  (pins shared with the historical engine)
    PARENT_CONFIG_SHA256, PARENT_REVISION, PARENT_SHA256,
)

PARENT_REPO = 'google/gemma-4-12B-it'
PARENT_TOKENIZER_SHA256 = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'
TRAINER_SCHEMA = 'lilith/lora-trainer@1'
ATTENTION_IMPLEMENTATION = 'lilith_chunked'

# Exactly the language model's attention and MLP projections. Global (K = V) layers
# have no v_proj, so they contribute q/k/o + MLP. Vision/audio embedders never match.
LANGUAGE_MODEL_TARGETS = (
    r'^model\.language_model\.layers\.\d+\.'
    r'(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$'
)
_TARGET_PATTERN = re.compile(LANGUAGE_MODEL_TARGETS)
ALLOWED_RANKS = (16, 32)


class LilithTrainerError(RuntimeError):
    """Trainer failure with a stable diagnostic code and a hint for the caller."""

    def __init__(self, code: str, message: str, help: str):
        super().__init__(f'[{code}] {message} (help: {help})')
        self.code = code
        self.help = help


# --------------------------------------------------------------------------- config

BasePrecision = Literal['bf16', 'nf4', 'fp32']


@dataclass(frozen=True)
class TrainerConfig:
    rank: int = 16
    alpha: int | None = None            # None -> alpha = rank (scale 1, as the rank-2 seam)
    base_precision: BasePrecision = 'bf16'
    gradient_checkpointing: bool = True
    offload_activations: bool = False   # save_on_cpu for 32K on 24 GB cards
    ce_chunk_tokens: int = 1024
    attention_query_chunk: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    preference_beta: float = 0.1
    seed: int = 73

    def __post_init__(self):
        if self.rank not in ALLOWED_RANKS:
            raise LilithTrainerError('lilith::trainer::rank', f'rank {self.rank} is not a declared surface',
                                     f'use one of {ALLOWED_RANKS} (ADR 116 Decision 7)')
        if self.alpha is not None and self.alpha <= 0:
            raise LilithTrainerError('lilith::trainer::alpha', 'alpha must be positive', 'omit alpha for alpha = rank')
        if self.base_precision not in ('bf16', 'nf4', 'fp32'):
            raise LilithTrainerError('lilith::trainer::precision', f'unknown base precision {self.base_precision}',
                                     'use bf16 (H100), nf4 (24 GB QLoRA) or fp32 (CPU tests)')
        if self.ce_chunk_tokens <= 0 or self.attention_query_chunk <= 0:
            raise LilithTrainerError('lilith::trainer::chunk', 'chunk sizes must be positive',
                                     'ce_chunk_tokens=1024 and attention_query_chunk=512 are the defaults')
        if self.preference_beta <= 0:
            raise LilithTrainerError('lilith::trainer::beta', 'preference beta must be positive', 'use 0.1')

    @property
    def lora_alpha(self) -> int:
        return self.rank if self.alpha is None else self.alpha

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record['lora_alpha'] = self.lora_alpha
        record['target_modules'] = LANGUAGE_MODEL_TARGETS
        return record


# ------------------------------------------------------------------------ examples

def _check_weight(weight: float, what: str) -> float:
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise LilithTrainerError('lilith::trainer::weight', f'{what} weight {weight!r} is outside [0, 1]',
                                 'ADR 110 rule 7: weights scale a sibling, they are never a penalty')
    return float(weight)


@dataclass(frozen=True)
class SftExample:
    """One teacher-forced sequence; ``target_mask[i]`` marks token i as a supervised target."""
    input_ids: tuple[int, ...]
    target_mask: tuple[bool, ...]
    weight: float = 1.0
    source: str = ''

    def __post_init__(self):
        if len(self.input_ids) != len(self.target_mask) or len(self.input_ids) < 2:
            raise LilithTrainerError('lilith::trainer::example', 'input_ids and target_mask differ or are too short',
                                     'build examples with build_sft_example()')
        if self.target_mask[0]:
            raise LilithTrainerError('lilith::trainer::example', 'the first token has no context to predict it',
                                     'mask the first token (it is always prompt)')
        if not any(self.target_mask):
            raise LilithTrainerError('lilith::trainer::example', 'example has no supervised target',
                                     'drop examples without assistant tokens')
        _check_weight(self.weight, 'SFT')


def build_sft_example(prompt_ids: Sequence[int], answer_ids: Sequence[int], *, weight: float = 1.0,
                      source: str = '') -> SftExample:
    prompt, answer = list(prompt_ids), list(answer_ids)
    if not prompt or not answer:
        raise LilithTrainerError('lilith::trainer::example', 'prompt and answer must be non-empty',
                                 'render the chat template before building examples')
    return SftExample(tuple(prompt + answer), tuple([False] * len(prompt) + [True] * len(answer)),
                      weight, source)


@dataclass(frozen=True)
class PreferencePair:
    """A corrected sibling (chosen) against the original response (rejected), same prompt."""
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    weight: float = 1.0
    source: str = ''

    def __post_init__(self):
        if not self.prompt_ids or not self.chosen_ids or not self.rejected_ids:
            raise LilithTrainerError('lilith::trainer::pair', 'prompt, chosen and rejected must be non-empty',
                                     'a sibling pair needs both responses')
        if self.chosen_ids == self.rejected_ids:
            raise LilithTrainerError('lilith::trainer::pair', 'chosen and rejected are identical',
                                     'identical siblings carry no contrast; drop the pair')
        _check_weight(self.weight, 'preference')

    def chosen(self) -> SftExample:
        return build_sft_example(self.prompt_ids, self.chosen_ids, weight=1.0, source=self.source)

    def rejected(self) -> SftExample:
        return build_sft_example(self.prompt_ids, self.rejected_ids, weight=1.0, source=self.source)


# ---------------------------------------------------------------- chunked attention

def _torch():
    import torch
    return torch


def _chunk_bounds(start: int, end: int, offset: int, key_length: int, window: int | None) -> tuple[int, int]:
    """Key range visible to queries [start, end) whose absolute index is i + offset."""
    high = min(key_length, end + offset)
    low = 0 if window is None else max(0, start + offset - window + 1)
    return low, high


def _chunk_allowed(torch, start: int, end: int, low: int, high: int, offset: int,
                   window: int | None, device) -> Any:
    query = torch.arange(start, end, device=device)[:, None] + offset
    key = torch.arange(low, high, device=device)[None, :]
    allowed = key <= query
    if window is not None:
        allowed &= (query - key) < window
    return allowed


class _ChunkedAttentionFunction:
    """Built lazily so the module imports without torch."""
    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is not None:
            return cls._cls
        torch = _torch()

        class ChunkedAttention(torch.autograd.Function):
            @staticmethod
            def forward(ctx, query, key, value, scaling: float, window, chunk: int):
                # query [B, H, Tq, D]; key/value [B, Hkv, Tk, D] (not repeated for GQA).
                batch, heads, q_len, dim = query.shape
                kv_heads, k_len = key.shape[1], key.shape[2]
                if heads % kv_heads:
                    raise LilithTrainerError('lilith::attention::gqa', 'query heads not a multiple of KV heads',
                                             'check the per-layer config')
                groups = heads // kv_heads
                offset = k_len - q_len
                if offset < 0:
                    raise LilithTrainerError('lilith::attention::shape', 'more queries than keys',
                                             'the cache must include the current tokens')
                q5 = query.view(batch, kv_heads, groups, q_len, dim)
                output = torch.empty_like(q5)
                lse = torch.empty((batch, kv_heads, groups, q_len), dtype=torch.float32, device=query.device)
                for start in range(0, q_len, chunk):
                    end = min(q_len, start + chunk)
                    low, high = _chunk_bounds(start, end, offset, k_len, window)
                    qc = q5[:, :, :, start:end]
                    kc = key[:, :, None, low:high]
                    vc = value[:, :, None, low:high]
                    scores = torch.matmul(qc, kc.transpose(-1, -2)).float() * scaling
                    allowed = _chunk_allowed(torch, start, end, low, high, offset, window, query.device)
                    scores = scores.masked_fill(~allowed, float('-inf'))
                    chunk_lse = torch.logsumexp(scores, dim=-1)
                    probs = torch.exp(scores - chunk_lse[..., None])
                    output[:, :, :, start:end] = torch.matmul(probs.to(value.dtype), vc)
                    lse[:, :, :, start:end] = chunk_lse
                ctx.save_for_backward(query, key, value, output, lse)
                ctx.scaling, ctx.window, ctx.chunk = scaling, window, chunk
                return output.view(batch, heads, q_len, dim)

            @staticmethod
            def backward(ctx, grad_output):
                query, key, value, output, lse = ctx.saved_tensors
                scaling, window, chunk = ctx.scaling, ctx.window, ctx.chunk
                batch, heads, q_len, dim = query.shape
                kv_heads, k_len = key.shape[1], key.shape[2]
                groups = heads // kv_heads
                offset = k_len - q_len
                q5 = query.view(batch, kv_heads, groups, q_len, dim)
                g5 = grad_output.contiguous().view(batch, kv_heads, groups, q_len, dim).to(query.dtype)
                dtype = query.dtype  # matmuls in the input dtype, accumulation/softmax in fp32
                grad_query = torch.zeros_like(q5, dtype=torch.float32)
                grad_key = torch.zeros(key.shape, dtype=torch.float32, device=key.device)
                grad_value = torch.zeros(value.shape, dtype=torch.float32, device=value.device)
                for start in range(0, q_len, chunk):
                    end = min(q_len, start + chunk)
                    low, high = _chunk_bounds(start, end, offset, k_len, window)
                    qc = q5[:, :, :, start:end]
                    kc = key[:, :, None, low:high]
                    vc = value[:, :, None, low:high]
                    gc = g5[:, :, :, start:end]
                    oc = output[:, :, :, start:end]
                    scores = torch.matmul(qc, kc.transpose(-1, -2)).float() * scaling
                    allowed = _chunk_allowed(torch, start, end, low, high, offset, window, query.device)
                    scores = scores.masked_fill(~allowed, float('-inf'))
                    probs = torch.exp(scores - lse[:, :, :, start:end, None])
                    # dV = P^T dO, summed over the query heads sharing each KV head.
                    grad_value[:, :, low:high] += torch.einsum(
                        'bhgcl,bhgcd->bhld', probs.to(dtype), gc).float()
                    grad_probs = torch.matmul(gc, vc.transpose(-1, -2)).float()
                    delta = (gc.float() * oc.float()).sum(dim=-1, keepdim=True)
                    grad_scores = (probs * (grad_probs - delta) * scaling).to(dtype)
                    grad_query[:, :, :, start:end] = torch.matmul(grad_scores, kc).float()
                    grad_key[:, :, low:high] += torch.einsum('bhgcl,bhgcd->bhld', grad_scores, qc).float()
                return (grad_query.view(batch, heads, q_len, dim).to(query.dtype),
                        grad_key.to(key.dtype), grad_value.to(value.dtype), None, None, None)

        cls._cls = ChunkedAttention
        return cls._cls


def chunked_attention(query, key, value, *, scaling: float, window: int | None, chunk: int):
    """Causal (optionally sliding-window) attention without materializing [Tq, Tk] for all rows."""
    return _ChunkedAttentionFunction.get().apply(query, key, value, float(scaling), window, int(chunk))


def reference_attention(query, key, value, *, scaling: float, window: int | None):
    """Naive math attention, used only as the test oracle."""
    torch = _torch()
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    q_len, k_len = query.shape[2], key.shape[2]
    scores = torch.matmul(query, key.transpose(-1, -2)).float() * scaling
    allowed = _chunk_allowed(torch, 0, q_len, 0, k_len, k_len - q_len, window, query.device)
    scores = scores.masked_fill(~allowed, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1).to(value.dtype), value)


_ATTENTION_STATE = {'query_chunk': 512}


def _hf_chunked_attention(module, query, key, value, attention_mask, dropout: float = 0.0,
                          scaling: float | None = None, sliding_window: int | None = None,
                          softcap: float | None = None, **_kwargs):
    if attention_mask is not None:
        raise LilithTrainerError('lilith::attention::mask', 'explicit attention masks are not supported',
                                 'the trainer is text-only and unpadded; causal/window masks are derived')
    if dropout:
        raise LilithTrainerError('lilith::attention::dropout', 'attention dropout is not supported',
                                 'Gemma 4 trains with attention_dropout = 0')
    if softcap is not None:
        raise LilithTrainerError('lilith::attention::softcap', 'attention softcapping is not supported',
                                 'Gemma 4 has no attention softcap; check the config')
    scale = module.head_dim ** -0.5 if scaling is None else scaling
    window = sliding_window if getattr(module, 'is_sliding', False) else None
    output = chunked_attention(query, key, value, scaling=scale, window=window,
                               chunk=_ATTENTION_STATE['query_chunk'])
    return output.transpose(1, 2).contiguous(), None


def register_chunked_attention(query_chunk: int) -> str:
    """Register the chunked implementation with Transformers; masks are derived, not built.

    The implementation name is absent from the mask registry, so Transformers passes
    ``attention_mask=None`` and never allocates a [T, T] mask at 32K.
    """
    from transformers import AttentionInterface
    _ATTENTION_STATE['query_chunk'] = int(query_chunk)
    AttentionInterface.register(ATTENTION_IMPLEMENTATION, _hf_chunked_attention)
    return ATTENTION_IMPLEMENTATION


# ------------------------------------------------------------ chunked cross-entropy

def _chunk_token_nll(hidden, weight, targets, softcap):
    torch = _torch()
    logits = torch.matmul(hidden, weight.t()).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return torch.logsumexp(logits, dim=-1) - logits.gather(-1, targets[:, None]).squeeze(-1)


def chunked_token_nll(hidden, weight, targets, *, softcap: float | None, chunk: int):
    """Per-token negative log-likelihood; each [chunk, V] logits block is recomputed in backward.

    ``hidden`` is [T, H] (final-norm hidden states aligned so row t predicts targets[t]).
    Peak logits memory is chunk * V * 4 bytes (1 GiB at chunk 1024) instead of T * V.
    """
    torch = _torch()
    from torch.utils.checkpoint import checkpoint
    if hidden.shape[0] != targets.shape[0]:
        raise LilithTrainerError('lilith::loss::shape', 'hidden rows and targets differ',
                                 'shift targets by one before calling')
    pieces = []
    for start in range(0, hidden.shape[0], chunk):
        end = min(hidden.shape[0], start + chunk)
        if hidden.requires_grad:
            pieces.append(checkpoint(_chunk_token_nll, hidden[start:end], weight, targets[start:end],
                                     softcap, use_reentrant=False))
        else:
            pieces.append(_chunk_token_nll(hidden[start:end], weight, targets[start:end], softcap))
    return torch.cat(pieces)


def full_token_nll(hidden, weight, targets, *, softcap: float | None):
    """Unchunked oracle for tests."""
    return _chunk_token_nll(hidden, weight, targets, softcap)


# ------------------------------------------------------------------- model surface

def is_lora_target(name: str) -> bool:
    return _TARGET_PATTERN.fullmatch(name) is not None


def lora_target_modules(model) -> list[str]:
    """Names of every Linear module the LoRA surface covers, from the unwrapped parent."""
    torch = _torch()
    return sorted(name for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Linear) and is_lora_target(name))


def expected_lora_parameters(text_config, rank: int) -> int:
    """Analytic LoRA parameter count for the declared surface (r * (in + out) per projection)."""
    hidden, mlp = text_config.hidden_size, text_config.intermediate_size
    heads = text_config.num_attention_heads
    total = 0
    per_layer = getattr(text_config, 'per_layer_config', None)   # Transformers >= 5.1x
    for index, kind in enumerate(text_config.layer_types):
        k_eq_v = text_config.attention_k_eq_v and kind != 'sliding_attention'
        if per_layer is not None:
            head_dim, kv_heads = per_layer[index].head_dim, per_layer[index].num_key_value_heads
        elif kind == 'sliding_attention':
            head_dim, kv_heads = text_config.head_dim, text_config.num_key_value_heads
        else:   # Transformers 5.10.x: flat global_head_dim / num_global_key_value_heads
            head_dim = text_config.global_head_dim or text_config.head_dim
            kv_heads = (text_config.num_global_key_value_heads if k_eq_v
                        else text_config.num_key_value_heads)
        total += rank * (hidden + heads * head_dim) * 2           # q_proj, o_proj
        total += rank * (hidden + kv_heads * head_dim) * (1 if k_eq_v else 2)   # k_proj (+ v_proj)
        total += rank * (hidden + mlp) * 3                         # gate, up, down
    return total


def quantization_spec(config: TrainerConfig) -> dict[str, Any] | None:
    """Plain-data QLoRA spec (NF4 blocks, bf16 compute, embeddings/head/towers unquantized)."""
    if config.base_precision != 'nf4':
        return None
    return {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_use_double_quant': True,
            'bnb_4bit_compute_dtype': 'bfloat16',
            'llm_int8_skip_modules': ['lm_head', 'embed_tokens', 'embed_vision', 'embed_audio',
                                      'vision_tower', 'audio_tower']}


def _dtype(config: TrainerConfig):
    torch = _torch()
    return torch.float32 if config.base_precision == 'fp32' else torch.bfloat16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_parent(parent: Path) -> dict[str, str]:
    """Refuse any parent whose weights, config or tokenizer differ from the pins."""
    expected = {'model.safetensors': PARENT_SHA256, 'config.json': PARENT_CONFIG_SHA256,
                'tokenizer.json': PARENT_TOKENIZER_SHA256}
    observed = {}
    for name, pin in expected.items():
        path = Path(parent) / name
        if not path.is_file():
            raise LilithTrainerError('lilith::parent::missing', f'{path} is missing',
                                     'run scripts/lilith/runpod/bootstrap.sh to download the pinned parent')
        observed[name] = sha256_file(path)
        if observed[name] != pin:
            raise LilithTrainerError('lilith::parent::pin', f'{name} sha256 {observed[name]} != pin {pin}',
                                     f'download {PARENT_REPO} at revision {PARENT_REVISION}; never train on a drifted parent')
    return observed


def prepare_lora_model(base, config: TrainerConfig):
    """Freeze the parent, attach LoRA to exactly the declared surface, verify it."""
    torch = _torch()
    from peft import LoraConfig, get_peft_model
    targets = lora_target_modules(base)
    if not targets:
        raise LilithTrainerError('lilith::lora::no_targets', 'no language-model projection matched',
                                 'the parent is not a Gemma4Unified model or module names changed')
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    # peft.prepare_model_for_kbit_training is deliberately NOT used: it upcasts every
    # non-quantized tensor (embeddings, tied LM head) to fp32, which breaks the NF4
    # memory budget (plan §2: 7.8 GB). Freezing + non-reentrant checkpointing is all
    # the NF4 base needs; assert_base_dtypes() below enforces bf16.
    torch.manual_seed(config.seed)
    model = get_peft_model(base, LoraConfig(
        r=config.rank, lora_alpha=config.lora_alpha, lora_dropout=0.0, bias='none',
        target_modules=LANGUAGE_MODEL_TARGETS))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        model.get_base_model().config.get_text_config().use_cache = False
    surface = assert_trainable_surface(model, targets)
    surface['base_dtypes'] = assert_base_dtypes(model.get_base_model(), config)
    return model, surface


def assert_base_dtypes(model, config: TrainerConfig) -> dict[str, int]:
    """Frozen floating tensors stay in the base dtype (bf16 for bf16/NF4); norms may be fp32.

    Quantized NF4 storage (uint8) is skipped; LoRA factors (fp32 masters) are skipped.
    Returns a dtype histogram for the receipt.
    """
    torch = _torch()
    expected = _dtype(config)
    histogram: dict[str, int] = {}
    offenders = []
    for name, parameter in model.named_parameters():
        if '.lora_' in name or not parameter.is_floating_point():
            continue
        histogram[str(parameter.dtype)] = histogram.get(str(parameter.dtype), 0) + parameter.numel()
        if parameter.dtype == expected:
            continue
        if parameter.dtype == torch.float32 and 'norm' in name.rsplit('.', 2)[-2]:
            continue
        offenders.append(f'{name}:{parameter.dtype}')
    if offenders:
        raise LilithTrainerError('lilith::trainer::dtype', f'{len(offenders)} base tensors left {expected}: '
                                 f'{offenders[:5]}',
                                 'something upcast the frozen base (e.g. prepare_model_for_kbit_training); '
                                 'keep embeddings/head/activations in bf16')
    return histogram


def _strip_adapter_prefix(name: str) -> str:
    return name.removeprefix('base_model.model.')


def assert_trainable_surface(model, targets: Sequence[str]) -> dict[str, int]:
    """Trainable parameters are exactly lora_A/lora_B of the declared targets."""
    expected = set(targets)
    covered: set[str] = set()
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        match = re.fullmatch(r'(.+)\.lora_[AB]\.[^.]+\.weight', _strip_adapter_prefix(name))
        if match is None or match.group(1) not in expected:
            raise LilithTrainerError('lilith::lora::surface', f'unexpected trainable parameter {name}',
                                     'only language-model attention/MLP LoRA factors may train')
        covered.add(match.group(1))
        count += parameter.numel()
    if covered != expected:
        raise LilithTrainerError('lilith::lora::surface', f'{len(expected - covered)} targets lack LoRA',
                                 'PEFT did not wrap every declared projection')
    return {'target_modules': len(expected), 'trainable_parameters': count}


def load_tokenizer(parent: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(parent, local_files_only=True)


def load_parent_base(parent: Path, config: TrainerConfig, *, device: str = 'cuda:0'):
    """The frozen parent exactly as training sees it (bf16, or NF4 blocks + bf16 rest).

    Cold reload uses this too, so an NF4-trained adapter is checked on an NF4 base.
    Pins are verified by the caller.
    """
    torch = _torch()
    import transformers
    implementation = register_chunked_attention(config.attention_query_chunk)
    kwargs: dict[str, Any] = {'local_files_only': True, 'dtype': _dtype(config),
                              'attn_implementation': implementation, 'low_cpu_mem_usage': True,
                              'device_map': {'': device}}
    spec = quantization_spec(config)
    if spec is not None:
        kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
            **{**spec, 'bnb_4bit_compute_dtype': torch.bfloat16})
    base = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(parent, **kwargs)
    assert_base_dtypes(base, config)
    return base, implementation


def load_training_model(parent: Path, config: TrainerConfig, *, device: str = 'cuda:0'):
    """Verify the pins, load the parent with chunked attention and attach the LoRA surface.

    Returns (model, receipt). The receipt records pins, surface size and trainer config.
    """
    pins = verify_parent(parent)
    base, implementation = load_parent_base(parent, config, device=device)
    model, surface = prepare_lora_model(base, config)
    return model, {'parent_pins': pins, 'surface': surface, 'config': config.to_record(),
                              'attention_implementation': implementation}


# -------------------------------------------------------------------------- losses

def _unwrap(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def token_logps(model, example: SftExample, config: TrainerConfig):
    """Log-probabilities of every next token (length T-1) through chunked attention and chunked CE."""
    torch = _torch()
    base = _unwrap(model)
    device = next(p for p in base.parameters()).device
    ids = torch.tensor([list(example.input_ids)], dtype=torch.long, device=device)
    positions = torch.arange(ids.shape[1], device=device)[None, :]
    hidden = base.model(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state[0]
    weight = base.lm_head.weight
    softcap = base.config.get_text_config().final_logit_softcapping
    targets = ids[0, 1:]
    return -chunked_token_nll(hidden[:-1], weight, targets, softcap=softcap, chunk=config.ce_chunk_tokens)


def weighted_sft_loss(model, examples: Sequence[SftExample], config: TrainerConfig):
    """sum_e w_e * sum_t nll_t / total supervised tokens; weight 0 contributes no gradient."""
    torch = _torch()
    total_tokens = sum(sum(example.target_mask[1:]) for example in examples)
    loss = None
    for example in examples:
        logp = token_logps(model, example, config)
        mask = torch.tensor(example.target_mask[1:], dtype=logp.dtype, device=logp.device)
        term = -(logp * mask).sum() * example.weight / total_tokens
        loss = term if loss is None else loss + term
    return loss, total_tokens


def sequence_logp(model, example: SftExample, config: TrainerConfig):
    torch = _torch()
    logp = token_logps(model, example, config)
    mask = torch.tensor(example.target_mask[1:], dtype=logp.dtype, device=logp.device)
    return (logp * mask).sum()


def preference_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *,
                    beta: float, weight=1.0):
    """Sibling-contrast (DPO) loss: -w * log sigmoid(beta * (chosen margin - rejected margin)).

    Equals w * log 2 when the policy equals the reference; decreases as the policy
    raises the corrected sibling relative to the original.
    """
    torch = _torch()
    margin = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
    weight = torch.as_tensor(weight, dtype=margin.dtype, device=margin.device)
    if bool((weight < 0).any()) or bool((weight > 1).any()):
        raise LilithTrainerError('lilith::trainer::weight', 'preference weight outside [0, 1]',
                                 'ADR 110 rule 7 forbids negative weights')
    return -(weight * torch.nn.functional.logsigmoid(beta * margin))


# ------------------------------------------------------------------------- stepping

@dataclass(frozen=True)
class StepReceipt:
    kind: str
    step: int
    loss: float
    tokens: int
    grad_norm: float
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


def make_optimizer(model, config: TrainerConfig):
    torch = _torch()
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def _activation_context(config: TrainerConfig):
    import contextlib
    torch = _torch()
    if config.offload_activations:
        return torch.autograd.graph.save_on_cpu(pin_memory=torch.cuda.is_available())
    return contextlib.nullcontext()


def _synchronize():
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clip_and_step(model, optimizer, config: TrainerConfig) -> float:
    torch = _torch()
    parameters = [p for p in model.parameters() if p.requires_grad]
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            optimizer.zero_grad(set_to_none=True)
            raise LilithTrainerError('lilith::trainer::nonfinite', 'non-finite LoRA gradient',
                                     'no optimizer step was taken; inspect the batch and learning rate')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norm


def sft_step(model, optimizer, examples: Sequence[SftExample], config: TrainerConfig, step: int,
            numerics: Any = None) -> StepReceipt:
    """``numerics`` (a ``lilith_numerics.NumericsConfig``) wraps this step's compute; the
    caller resolves the recorded default (see ``lilith_numerics.platform_default``) and
    passes it explicitly. Unset means untouched PyTorch, matching every call site that
    predates numerics wiring."""
    import time
    import lilith_numerics
    numerics = lilith_numerics.NumericsConfig() if numerics is None else numerics
    with numerics.applied():
        model.train()
        _synchronize()
        started = time.perf_counter()
        with _activation_context(config):
            loss, tokens = weighted_sft_loss(model, examples, config)
            loss.backward()
        norm = _clip_and_step(model, optimizer, config)
        _synchronize()
    return StepReceipt('weighted_sft', step, float(loss.detach()), sum(len(e.input_ids) for e in examples),
                       norm, time.perf_counter() - started, {'supervised_tokens': tokens})


def preference_step(model, optimizer, pairs: Sequence[PreferencePair], config: TrainerConfig,
                    step: int, numerics: Any = None) -> StepReceipt:
    """One DPO-style step; the reference is the frozen parent (adapter disabled).

    See ``sft_step`` for ``numerics``.
    """
    import time
    import lilith_numerics
    torch = _torch()
    numerics = lilith_numerics.NumericsConfig() if numerics is None else numerics
    with numerics.applied():
        model.train()
        _synchronize()
        started = time.perf_counter()
        references = []
        with torch.no_grad(), model.disable_adapter():
            for pair in pairs:
                references.append((sequence_logp(model, pair.chosen(), config),
                                   sequence_logp(model, pair.rejected(), config)))
        total = 0.0
        margins = []
        with _activation_context(config):
            for pair, (ref_chosen, ref_rejected) in zip(pairs, references):
                chosen = sequence_logp(model, pair.chosen(), config)
                rejected = sequence_logp(model, pair.rejected(), config)
                loss = preference_loss(chosen, rejected, ref_chosen, ref_rejected,
                                       beta=config.preference_beta, weight=pair.weight) / len(pairs)
                loss.backward()
                total += float(loss.detach())
                margins.append(float(((chosen - ref_chosen) - (rejected - ref_rejected)).detach()))
        norm = _clip_and_step(model, optimizer, config)
        _synchronize()
    tokens = sum(len(p.prompt_ids) * 2 + len(p.chosen_ids) + len(p.rejected_ids) for p in pairs)
    return StepReceipt('sibling_preference', step, total, tokens, norm, time.perf_counter() - started,
                       {'margins': margins, 'beta': config.preference_beta})


# ------------------------------------------------------------------- adapter files

def adapter_tensor_state(model) -> dict[str, Any]:
    """LoRA tensors keyed by adapter-independent names (works for trained and reloaded models)."""
    state = {}
    for name, parameter in model.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            state[_strip_adapter_prefix(name)] = parameter.detach()
    return state


def adapter_tensor_sha256(model) -> str:
    torch = _torch()
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter_tensor_state(model).items()):
        value = tensor.to('cpu', torch.float32).contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class AdapterReceipt:
    directory: str
    files_sha256: dict[str, str]
    tensor_sha256: str
    tensors: int
    rank: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def save_adapter(model, directory: Path, config: TrainerConfig) -> AdapterReceipt:
    directory = Path(directory)
    if directory.exists():
        raise LilithTrainerError('lilith::adapter::exists', f'{directory} already exists',
                                 'adapters are write-once; choose a new directory')
    model.save_pretrained(directory, safe_serialization=True)
    files = {path.name: sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file()}
    if 'adapter_model.safetensors' not in files:
        raise LilithTrainerError('lilith::adapter::missing', 'PEFT wrote no adapter_model.safetensors',
                                 'check the installed PEFT version')
    receipt = AdapterReceipt(str(directory), files, adapter_tensor_sha256(model),
                             len(adapter_tensor_state(model)), config.rank)
    (directory / 'lilith-adapter-receipt.json').write_text(json.dumps(receipt.to_record(), indent=2) + '\n')
    return receipt


def reload_adapter(base, directory: Path, receipt: AdapterReceipt | dict[str, Any]):
    """Cold-reload an adapter onto a fresh parent and prove tensor identity with the receipt."""
    from peft import PeftModel
    record = receipt.to_record() if isinstance(receipt, AdapterReceipt) else dict(receipt)
    directory = Path(directory)
    for name, expected in record['files_sha256'].items():
        if sha256_file(directory / name) != expected:
            raise LilithTrainerError('lilith::adapter::file_hash', f'{name} changed since save',
                                     'never reload an adapter whose files drifted; retrain or restore')
    model = PeftModel.from_pretrained(base, directory, is_trainable=False)
    observed = adapter_tensor_sha256(model)
    if observed != record['tensor_sha256']:
        raise LilithTrainerError('lilith::adapter::tensor_hash', 'reloaded LoRA tensors differ from the receipt',
                                 'check dtype autocasting between PEFT versions')
    return model, observed


def iter_chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]
