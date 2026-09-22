# Lilith next five steps — completed bounded run, 22 September 2026

User authorization: execute roadmap steps 1–5 using Sol subagents and prepare a
local testing VM from the existing Solstice CI image; use Refraction Forger for a
newer image if necessary. Native VM preparation is now authorized (the earlier
host-setup deferral no longer blocks this run).

## Work and evidence

| Step | Work | Current evidence location |
|---|---|---|
| 1. Meaningful quality signal | Versioned artifact extraction; logic/bounds/error handling tasks; preserve old scores | `training/lilith/rl-20260922/` |
| 2. Bounded training comparison | Same-parent baseline/SFT/KLPO, fresh family development tasks, explicit retention | `training/lilith/rl-20260922/` |
| 3. Vulkan adapter path | Exact two-tensor export, additive last-query application, zero/effect/reload/rollback controls and timing | `training/lilith/vulkan-adapter-20260922/` |
| 4. Broader coordination | Khnum patch proposal, independent verification, Ptah review, Lilith report; durable correction/cancellation and peer failure | `training/lilith/coordination-20260922/` |
| 5. Native/reference validation | Isolated illumos VM and pinned-image provenance; actual same-pack deployed Gemma/Ptah comparison | `training/lilith/native-20260922/`, `training/lilith/comparison-20260922/` |

The existing Ptah reference is the pinned
`JonathanColetti/Qwen3.8-27B-Uncensored` derivative, not an unmodified upstream
model. Its identity remains explicit in every comparison. Public development
packs and single observations cannot close statistical parity or sealed-release
gates.

## Run constraints

- Three Sol subagents own quality, VM and coordination work. Root owns the Vulkan
  adapter path, paired deployed-backend comparison, integration review and report.
- One GPU owner at a time. Parent model loading, compilation and VM work are
  serialized when host RAM is constrained. VM default is 2 vCPU / 4 GiB RAM.
- Use project-owned copy-on-write VM disks; preserve existing images and guests.
- Preserve the 21 September frozen tasks, trajectories and scores. New reward
  semantics are versioned; no retroactive score improvements.
- No deployed Gemma weight replacement or adapter promotion without independent
  evidence of useful transfer. A successful adapter mechanics probe is not a
  quality certificate, especially across BF16 training and quantized inference.
- Model-generated repository code must not execute on the inference host. Use
  an isolated VM/sandbox executor or a deliberately bounded interpreter.
- Native tests establish only the guest/toolchain/task behavior actually run;
  they do not establish that the complete Akh runtime runs natively on illumos.

## Results

- Steps 1–2: the v2 artifact contract accepts one enclosing Python fence while
  retaining grammar and finite-case checks. All three actual HF arms score 2/4
  train and 2/4 development-family tasks; independent replay agrees. The KLPO
  update used real reward contrast `[1,1,0,0]`, but no quality gain was measured.
  Invalid alignment and width-zero errors are distinct from grammar rejection.
  Fresh-parent retention passes 4/4 earlier tasks in each arm, and KLPO cold
  reload is exact. Temporary ZFS cache budgeting was restored after loading.
- Step 3: exact rank-2 export and two actual Vulkan processes pass zero/effect,
  reload and rollback controls. Full-vocabulary hashes match across processes.
  The nonzero logit effect is 0.0043854; generated tokens on the probe are unchanged.
  Timing is descriptive and slower for the adapter; no deployment promotion.
- Step 4: 50 coordination tests pass. The third model-backed episode completed
  Lilith delegation, Khnum's patch, native illumos verification, Ptah review and
  a truthful Lilith report. Fresh runtime processes enforce the Vulkan→HIP→Vulkan
  boundaries after owned turns complete. Original fixture files remain unchanged;
  `model-run3/candidate.patch` is the reviewable tested result. One unittest method
  contains seven boundary assertions; the identical candidate also passed 1,728
  additional native cases in a separate unscored diagnostic. Earlier failed runs
  and the original unknown report turn remain preserved. The final report took
  452 seconds: this proves bounded completion, not acceptable interactive latency
  or a GPU-driver fix.
- Step 5: OpenIndiana 2026.04 boots with SSH and Python/C/Rust toolchains. Native
  smoke checks, timeout and output limits, and guest restart were exercised.
  The complete sixteen-observation comparison gives Gemma 4/8 and Ptah 6/8.
  This misses the reference target and does not establish release parity.

All five requested engineering slices have implementation and recorded execution
results. This completes this bounded run, not Lilith's release gates. Priorities
are Vulkan report/prefill latency, training on diagnosed invalid-input mistakes,
and broader real illumos repository tasks with independent quality evaluation.
