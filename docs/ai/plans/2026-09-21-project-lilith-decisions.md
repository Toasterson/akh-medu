# Lilith decision map

Date: 2026-09-21. Canonical compact map; implementation sequence and evidence inventory live in the [Lilith plan](2026-09-21-project-lilith.md). Local project startup authorized by the user; capability-contract investigation claimed on 2026-09-21.

## role-and-authority: What is Lilith responsible for?

Blocked by: none
Status: resolved
Type: Grilling

### Question

Which model track, product role, learning objective, and autonomy apply?

### Answer

User selected Gemma 4, human interface and coordination of other Akhs as the primary role, preliminary coding improved through VSA-mediated learning from existing agents, and execution end-to-end within assigned scope. ADAM/Kaine capability coverage is the intended destination, including a visible disposition for research extensions. Lilith names the inherited track; preserve sovereign ADAM/Kaine lineage terminology. User addition: illumos is the primary OS and coding domain; Lilith must match or beat Qwen 3.8 27B in the agreed benchmarks, with failure diagnosis and real understanding/coding quality driving tuning.

## capability-contract: What exactly must transfer, and what is already usable?

Blocked by: role-and-authority
Status: in-progress
Type: Research

### Question

Reconcile the plan’s capability ledger with current code, ADAM/Kaine source documents, and live Anima MCP items. Which runtime, curriculum, and exam artifacts can be reused? What are the exact baseline checkpoint/backend, deployment hardware, inherited-model certification mapping, per-family acceptance criteria, and bounded run budgets?

### Answer

Pending. Output a complete gap/coverage ledger and frozen L0 contracts, including the pinned Qwen reference, controlled-model and deployed-system comparison tracks, primary benchmark/family metrics, powered equivalence/superiority criteria, native illumos support matrix, and development/validation/sealed task splits. Reuse ADR 086 / Phase 50 benchmark work where verified. Startup evidence: [local inventory and checks](../../../training/lilith/startup-20260921/README.md), [draft comparison contract](../../../benchmarks/lilith/contract.json), and [illumos benchmark audit](../research/2026-09-21-lilith-illumos-benchmark-inventory.md). Local harness tests pass (26); Gemma artifact pinned. Tracker reconciliation is blocked by missing Anima PAT; The pinned local Ptah derivative is available as a reviewer/reference candidate; formal Qwen reference certification remains unresolved. User selected illumos VMs on hephaistos and deferred validation until host setup is ready; local engineering proceeds independently. Keep this ticket in progress; no certification or parity claim.

## local-runtime: Can Gemma serve an isolated Lilith session on this host?

Blocked by: role-and-authority
Status: resolved
Type: Prototype

### Question

Diagnose the local inference timeout, pin a source-built runtime, and prepare an isolated Lilith configuration without starting existing workspaces. What diagnostics and evaluation contracts can run while Qwen/native validation is deferred?

### Answer

Resolved for the local development foundation. Gemma now runs on Vulkan after
repairs to RMSNorm/rotary/softmax dispatch and the missing GELU kernel. Legacy
rocBLAS offload is restricted to CPU-backed models. Four CPU/Vulkan operation
tests pass; the full-model hidden-state comparison passes the 0.9999 cosine gate.
The [shared-runtime fixture](../../../training/lilith/vulkan-20260921/README.md)
passed Lilith code creation, 2,051 independent integer cases, Ptah read-only review,
and return to the original Lilith session in one process. Generated runtime and
executor binding identities were corrected with regression tests. The dedicated
Vulkan stack is prepared and stopped; [run instructions](../../../examples/lilith/README.md).
This resolves local serving and serialized review feasibility, not autonomous
specialist delegation, VSA learning, model quality parity or native illumos support.
Native validation remains deferred to illumos VMs on hephaistos after host setup.

## coordination-contract: How does an assignment survive distributed execution?

Blocked by: capability-contract
Status: in-progress
Type: Prototype

### Question

Can one scoped assignment travel through human intake, multiple specialists, artifact verification, interruption/restart and completion without duplicate effects or loss of authority boundaries?

### Answer

Implementation started under the local-first authorization. A [bounded local coordination controller](../../../examples/lilith/COORDINATION.md) connects Lilith’s client-tool request to a durable assignment, independently verified snapshot, Ptah child session, and return to the human. Stable session request IDs cover response-loss recovery; scope and stale-artifact checks prevent unsupported completion. This is one specialist and one development fixture, not the full L2 acceptance gate. The second slice adds a concurrent durable cancellation mailbox, cancellation of owned child/report turns, recovery of uncertain submissions, and confirmed terminal-state tracking. The third slice generalizes operator-selected artifact contracts, pins verifier provenance, and adds a bounded unsigned-64 range reference fixture. Optional replay-verified lesson admission is now integrated after successful completion, with store-bound receipts and revocation reconciliation. Native illumos, second-specialist routing and broader grant revocation remain open.

## apprenticeship-contract: What makes another agent’s lesson useful and transferable?

Blocked by: capability-contract
Status: in-progress
Type: Prototype

### Question

Which versioned lesson envelope and VSA encoding provide useful code-task retrieval across Akhs with potentially different symbol spaces? Can evidence admission, local re-encoding, correction and cold reuse work on a small exemplar set?

### Answer

First frozen-weight [apprenticeship slice](../../../examples/lilith/APPRENTICESHIP.md) implemented: replay-verified lesson admission, exact provenance, portable symbolic VSA features, cold reuse, revocation and family-filtered candidate export. A predeclared three-arm development probe passed in all arms, so no quality improvement or VSA advantage is established. Operator-curated lessons/features, tiny corpus and related-family transfer are explicit limits. An authored reference/correction curriculum and a completed 24-run, six-task family-held-out comparison extend the local preparation. Gemma passes 5/6 per arm; Ptah passes 4/6 under the strict grammar, including one grammar-only failure that passes unscored normalization. Retrieval abstains, so no retrieval benefit or parity is established. Autonomous lesson synthesis, native illumos evidence, a practical improvement margin and Gemma 4 adapter certification remain open. This ticket defines the seam; L4 supplies the larger behavioral comparison.

## adaptation-and-release: Which deficits need training, and how is Lilith certified?

Blocked by: coordination-contract, apprenticeship-contract
Status: open
Type: Research

### Question

After the integrated Gemma baseline, which failures call for runtime repairs, better retrieval, or adapter training? What training/export path works on available hardware, and which existing release constraints require a Gemma-specific verifier?

### Answer

The initial adapter audit was followed through to a downloaded, hash-pinned official parent candidate and an isolated ROCm training environment. One actual rank-2, 16-token synthetic LoRA optimizer step passed finite-gradient, frozen-base mutation, forward-effect and exact adapter/optimizer reload checks, with zero reload-logit error in the same parent. This is mechanics evidence, not curriculum tuning or deployed Vulkan compatibility. The 24-run evaluation identifies concrete domain-validation and output-contract deficits; these remain separate from artificial grammar constraints. Legacy placeholder conversion and misleading metadata have been disabled. Larger training, verified Vulkan export/application and independent quality gates remain open; the subsequent bounded RL experiment records its own cold-reload evidence. Still required: a bounded adaptation recipe or evidence that no adaptation is needed, plus exact certification and rollback contracts for illumos. Require Qwen parity/superiority evidence and a failure ledger tying each intervention to a diagnosis and fresh-task outcome; account for native versus remote inference explicitly. Advanced research tickets are added only when baseline evidence identifies their purpose.

## klpo-local-experiment: Can verified outcomes drive a faithful RL update?

Status: implemented (bounded experiment; quality unproven)
Type: Experiment

The user approved the [KLPO assessment](../research/2026-09-21-lilith-klpo.md) and next local slices. Implement actual sampler probability and auxiliary-token capture, independently tested token-MC loss, and a bounded HF/ROCm LoRA run with unchanged/SFT controls. Preserve complete trajectories, failed/truncated outcomes and verifier provenance. The same implementation for collection/training reduces integration mismatch but still requires measured initial probability drift. Existing quantized Vulkan responses lack the required records and cannot be retroactively treated as these rollouts. A fresh-process reload of the resulting KLPO scratch adapter passed exactly. Independent quality gates and later Vulkan export remain open.

The [executed comparison](../../../training/lilith/rl-20260921/README.md) completed 24 responses, one SFT update and one actual KLPO update. Both changed scratch adapters, with unchanged strict scores (training 1/4, development holdout 0/4 per arm). Independent unscored extraction shows all fenced programs pass finite logic checks: this reward contrast is formatting, not established coding weakness. Preserve scores and improve the next versioned task/reward design before claiming useful adaptation.

## Handoff

Completed: the bounded local KLPO trajectory and training experiment; useful coding adaptation remains open. Runtime feasibility, bounded specialist coordination and optional verified lesson admission are implemented. Continue from the apprenticeship documentation and recorded evaluation evidence. Native validation remains deferred.

Next session:

```text
Invoke /decision-mapping with the map at docs/ai/plans/2026-09-21-project-lilith-decisions.md.
```

Pinned equivalent:

```text
Invoke /decision-mapping with the map at docs/ai/plans/2026-09-21-project-lilith-decisions.md, ticket capability-contract.
```
