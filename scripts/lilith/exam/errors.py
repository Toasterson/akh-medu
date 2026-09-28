"""Diagnostic errors for the Lilith exam harness.

Same shape as `dojo.errors`: a stable, greppable `code` plus a `help` line that names
the options the caller has. `render()` prints both.
"""
from __future__ import annotations

from dojo.errors import DojoError


class ExamError(DojoError):
    code = "lilith::exam"


class UnpinnedManifestError(ExamError):
    """Raised with every missing pin at once, so one fix round is enough."""

    code = "lilith::exam::unpinned"
    help = (
        "the exam manifest is frozen before any candidate is scored (workflow §4 step 2). Pin every listed "
        "item (sha256 hex of the frozen bytes), rebuild the manifest and record its hash in the run WI; "
        "or drop the family from the manifest if it is not part of this exam"
    )

    def __init__(self, missing: list[str]):
        self.missing = tuple(missing)
        super().__init__("exam manifest has unpinned items: " + "; ".join(missing))


class ManifestIntegrityError(ExamError):
    code = "lilith::exam::manifest_integrity"
    help = (
        "the manifest bytes no longer hash to the pin the run WI names. Restore the frozen manifest, "
        "or freeze a new manifest (new hash) and record the change in the run WI before scoring"
    )


class InvalidPlanError(ExamError):
    code = "lilith::exam::invalid_plan"
    help = "check the tier, family and sample count; sentinel and sealed tiers need n >= the sealed floor"


class ControllerConfigError(ExamError):
    code = "lilith::exam::controller_config"
    help = (
        "rates must be finite probabilities, sample floors and warning windows positive, confidence_z "
        "positive; start from ControllerConfig() (the adaptive_exam.rs defaults) and change one field"
    )


class UnknownFamilyError(ExamError):
    code = "lilith::exam::unknown_family"
    help = "register the family in the manifest's sampling plan, or fix the family name in the observation"


class InvalidCountsError(ExamError):
    code = "lilith::exam::invalid_counts"
    help = (
        "pass, precursor and catastrophic counts must each be <= samples, and pass <= samples - catastrophic "
        "(a catastrophic item never passes). Recount from the item outcomes"
    )


class StoppedSegmentError(ExamError):
    code = "lilith::exam::stopped"
    help = (
        "a controller stop is never cleared silently. Either end the run (RESULT doc + stop pods), or "
        "record the operator's reason with `ExamRun.resume(operator=..., reason=...)`, which starts a new "
        "segment with fresh warning streaks"
    )


class ProtectedTierError(ExamError):
    code = "lilith::exam::protected_tier"
    help = "training siblings may only be derived from frontier plans; sentinel/rolling/audit cases stay quarantined"


class CertificationError(ExamError):
    code = "lilith::exam::certification"
    help = (
        "a candidate becomes active only with every gate passed and an explicit operator-acceptance record "
        "naming the candidate manifest hash. Fix the listed problem, or keep serving the current active adapter"
    )


class NoActiveAdapterError(ExamError):
    code = "lilith::exam::no_active"
    help = (
        "serving loads only the active adapter and never falls back to a candidate or the parent. Accept a "
        "candidate (operator) to create active.json, or serve without an adapter explicitly"
    )


class PanelError(ExamError):
    code = "lilith::exam::panel"
    help = (
        "judge results must name an assigned blind id, the frozen rubric hash, a verdict, every rubric "
        "dimension in 0..3 and at least one cited turn. Re-dispatch the work order for the listed item"
    )


class ResultDocError(ExamError):
    code = "lilith::exam::result_doc"
    help = "the run JSON must be a `lilith/run-result@1` object; see result_doc.py for the fields"
