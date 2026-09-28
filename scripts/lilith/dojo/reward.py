"""`cos-v1`: the named, versioned reduction of the step vector to a training weight.

ADR 110 Decision 2 / ADR 116 Decision 4:

- the weight is in [0, 1]; there are no negative weights and no scalar-judge gradient;
- a failed step (hard constraint, invalid, or zero effect) trains at weight 0 and
  requires a corrected sibling, which then trains at its own earned weight;
- preference pairs come only from sibling contrast (same situation, corrected vs raw).

Missing components (`None`: not observable for this step) are left out and the
remaining coefficients renormalised, so an ablation can drop a component by
setting its coefficient to 0 without regenerating data.
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping

from dojo.families import EpisodeVerdict, HardConstraint, StepVector, StepVerdict
from dojo.snapshot import canonical_json, sha256_hex

COMPONENTS = ("effect", "validity", "provenance", "disposition", "budget")


@dataclasses.dataclass(frozen=True)
class ReductionPolicy:
    name: str
    version: int
    coefficients: Mapping[str, float]
    failed_effect_threshold: float = 0.0  # effect <= this counts as a failed step

    def __post_init__(self) -> None:
        if set(self.coefficients) != set(COMPONENTS):
            raise ValueError(f"{self.name}: coefficients must name exactly {COMPONENTS}")
        if any(c < 0 for c in self.coefficients.values()) or sum(self.coefficients.values()) <= 0:
            raise ValueError(f"{self.name}: coefficients must be non-negative with a positive sum")

    @property
    def policy_sha256(self) -> str:
        return sha256_hex(canonical_json({"name": self.name, "version": self.version,
                                          "coefficients": dict(self.coefficients),
                                          "failed_effect_threshold": self.failed_effect_threshold}))


COS_V1 = ReductionPolicy(
    name="cos-v1",
    version=1,
    coefficients={"effect": 0.40, "validity": 0.15, "provenance": 0.20, "disposition": 0.15, "budget": 0.10},
)


@dataclasses.dataclass(frozen=True)
class StepWeight:
    step: int
    weight: float
    failed: bool
    corrected_sibling_required: bool
    hard: frozenset[HardConstraint]
    policy: str

    def to_dict(self) -> dict:
        return {"step": self.step, "weight": self.weight, "failed": self.failed,
                "corrected_sibling_required": self.corrected_sibling_required,
                "hard": sorted(h.value for h in self.hard), "weight_policy": self.policy}


def is_failed(vector: StepVector, hard: frozenset[HardConstraint], policy: ReductionPolicy = COS_V1) -> bool:
    if hard:
        return True
    if vector.validity is not None and vector.validity <= 0.0:
        return True
    return vector.effect is not None and vector.effect <= policy.failed_effect_threshold


def reduce_step(verdict: StepVerdict, policy: ReductionPolicy = COS_V1) -> StepWeight:
    failed = is_failed(verdict.vector, verdict.hard, policy)
    if failed:
        weight = 0.0
    else:
        num = den = 0.0
        for name in COMPONENTS:
            value = getattr(verdict.vector, name)
            coeff = policy.coefficients[name]
            if value is None or coeff == 0:
                continue
            num += coeff * min(1.0, max(0.0, float(value)))
            den += coeff
        weight = 0.0 if den == 0 else num / den
    weight = min(1.0, max(0.0, weight))
    return StepWeight(verdict.step, weight, failed, failed, verdict.hard, f"{policy.name}@{policy.version}")


def reduce_episode(verdict: EpisodeVerdict, policy: ReductionPolicy = COS_V1) -> list[StepWeight]:
    return [reduce_step(s, policy) for s in verdict.steps]


@dataclasses.dataclass(frozen=True)
class WeightedSftRecord:
    episode_id: str
    step: int
    weight: float
    source: str  # "raw" | "corrected-sibling"
    policy: str
    policy_sha256: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class PreferencePair:
    episode_id: str
    step: int
    chosen: str  # "corrected-sibling"
    rejected: str  # "raw"
    chosen_weight: float
    rejected_weight: float
    margin: float
    policy: str


def sft_records(raw: EpisodeVerdict, corrected: EpisodeVerdict | None, policy: ReductionPolicy = COS_V1) -> list[WeightedSftRecord]:
    """Raw steps at their weight (0 when failed); corrected-sibling steps at their earned weight.
    A corrected sibling that itself fails trains at 0 too: it never outranks the rule it broke."""
    out = [WeightedSftRecord(raw.episode_id, w.step, w.weight, "raw", w.policy, policy.policy_sha256)
           for w in reduce_episode(raw, policy)]
    if corrected is not None:
        _same_situation(raw, corrected)
        out += [WeightedSftRecord(corrected.episode_id, w.step, w.weight, "corrected-sibling", w.policy, policy.policy_sha256)
                for w in reduce_episode(corrected, policy)]
    return out


def sibling_pairs(raw: EpisodeVerdict, corrected: EpisodeVerdict, policy: ReductionPolicy = COS_V1,
                  min_margin: float = 0.1) -> list[PreferencePair]:
    """Per aligned step: chosen = corrected sibling, rejected = raw, only when the sibling has no
    hard constraint and beats the raw step by at least `min_margin`. Never the other way round."""
    _same_situation(raw, corrected)
    raw_w = {w.step: w for w in reduce_episode(raw, policy)}
    pairs = []
    for cw in reduce_episode(corrected, policy):
        rw = raw_w.get(cw.step)
        if rw is None or cw.hard or cw.failed:
            continue
        margin = cw.weight - rw.weight
        if margin >= min_margin:
            pairs.append(PreferencePair(raw.episode_id, cw.step, "corrected-sibling", "raw", cw.weight, rw.weight,
                                        margin, f"{policy.name}@{policy.version}"))
    return pairs


def _same_situation(raw: EpisodeVerdict, corrected: EpisodeVerdict) -> None:
    if raw.episode_id != corrected.episode_id or raw.family != corrected.family:
        raise ValueError("sibling contrast needs the same situation (episode id and family)")


def weights_in_bounds(weights: Iterable[StepWeight]) -> bool:
    return all(0.0 <= w.weight <= 1.0 and (w.weight == 0.0 or not w.hard) for w in weights)
