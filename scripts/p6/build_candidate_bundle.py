#!/usr/bin/env python3
"""Build a per-checkpoint offline-eval bundle for Phase 51 P6.2.

The three retained release-recovery checkpoints (355735, 401577, 493261) were
retrieved from the Runpod volume as bare checkpoint directories (meta.json +
model.mpk + optim.mpk) — the adaptive loop's segment directories never
carried a `candidate-model.json`/`tokenizer.json`/`contracts.json` locally.
`eval_tool_contract_exam` (crates/grug-train/examples/eval_tool_contract_exam.rs)
loads a bundle via `KnowledgeModel::load_candidate[_gpu]`, which requires
exactly those three files (see crates/grug-train/src/knowledge_runtime.rs
`verify_knowledge_bundle`). This script reconstructs a minimal candidate
bundle per checkpoint by:

  - copying tokenizer.json + contracts.json unchanged from the P5 SFT bundle
    `knowledge-model-small-v1` (same run lineage; the sentinel exam's
    contract_hash already matches this contracts.json byte-for-byte, so the
    tool contracts did not change across the adaptive loop);
  - copying model.mpk + meta.json from the checkpoint's segment directory
    into a local `checkpoints/step_<n>` dir (optim.mpk is NOT copied —
    checkpoint_optimizer_hash is omitted from the manifest so
    verify_knowledge_bundle never looks for it; inference does not need
    Adam state);
  - writing a candidate-model.json that is a copy of
    knowledge-model-small-v1/candidate-model.json with only
    checkpoint_dir/checkpoint_step/checkpoint_model_hash swapped to the
    target checkpoint's own values (from its meta.json, whose model_hash
    matches the blake3 hashes verified when the checkpoints were retrieved
    from the pod). All other lineage fields (training_corpus_hash,
    training_records, capability_families, foundation, ...) are left as
    inherited from the v1 bundle: KnowledgeModelArtifact::verify() only
    checks that these hashes have a "blake3:" prefix and does not check them
    against any file in the bundle, so this is honest-but-approximate
    lineage metadata, not a re-verified provenance chain. `certification` is
    intentionally absent (candidate, not active).

This script COPIES; it never touches the originals under
runs/checkpoints/adaptive-loop-v9-release-recovery-20260910/ or
runs/checkpoints/knowledge-model-small-v1/.
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE_BUNDLE = REPO / "runs/checkpoints/knowledge-model-small-v1"
ADAPTIVE_LOOP = REPO / "runs/checkpoints/adaptive-loop-v9-release-recovery-20260910/segments"
OUT_ROOT = REPO / "runs/p6-2-decoder-rescore-20260913"

CHECKPOINTS = {
    355735: "segment-0001",
    401577: "segment-0002",
    493261: "segment-0004",
}


def build_one(step: int, segment: str) -> Path:
    src_ckpt_dir = ADAPTIVE_LOOP / segment / "checkpoints" / f"step_{step:08d}"
    meta = json.loads((src_ckpt_dir / "meta.json").read_text())
    assert meta["step"] == step, (meta["step"], step)
    model_hash = meta["model_hash"]

    bundle_dir = OUT_ROOT / f"bundle-{step}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer + contracts: byte-identical copies of the P5 SFT bundle's
    # files (their hashes are what candidate-model.json below declares).
    shutil.copy2(SOURCE_BUNDLE / "tokenizer.json", bundle_dir / "tokenizer.json")
    shutil.copy2(SOURCE_BUNDLE / "contracts.json", bundle_dir / "contracts.json")

    dst_ckpt_dir = bundle_dir / "checkpoints" / f"step_{step:08d}"
    dst_ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt_dir / "model.mpk", dst_ckpt_dir / "model.mpk")
    shutil.copy2(src_ckpt_dir / "meta.json", dst_ckpt_dir / "meta.json")

    manifest = json.loads((SOURCE_BUNDLE / "candidate-model.json").read_text())
    manifest["checkpoint_dir"] = f"checkpoints/step_{step:08d}"
    manifest["checkpoint_step"] = step
    manifest["checkpoint_model_hash"] = model_hash
    manifest.pop("checkpoint_optimizer_hash", None)
    # This candidate never went through certify_knowledge_model locally.
    manifest.pop("certification", None)

    (bundle_dir / "candidate-model.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    )
    print(f"built {bundle_dir} (checkpoint_model_hash={model_hash})")
    return bundle_dir


def main() -> None:
    for step, segment in CHECKPOINTS.items():
        build_one(step, segment)


if __name__ == "__main__":
    main()
