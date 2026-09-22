# Lilith managed training run 1

Launched 2026-09-22 at 18:05 UTC. **Stopped at 18:30 UTC without an optimizer
update:** akhsessd system retirement under the reconciliation stall watchdog.
See `cancellation-diagnosis.json`. See `result.json` for live stack state and
`worker/result.json` for learning results. Startup is not a completed update.

Session: `01a0ca4b-82d1-7a03-b72c-c0d0831e6605`.
The Python learning worker is owned by the dedicated akhrtd process. The driver
runs independently of the interactive shell, with a 259,200-second ceiling.
No shared service or deployed adapter is replaced.

`worker-config.json` freezes sources, parent, datasets, tool catalog and protocol.
The engine is the qualified synchronous-cache implementation. Training uses the
local BF16 PyTorch/ROCm gradient engine; this is distinct from Vulkan inference.
Reasoning is enabled with context 32,768 and response ceiling 16,384 tokens.

Protocol: four baseline train tasks; one supervised optimizer step on eight
frozen harvested examples; three online reward banks; separate SFT, RL and
adapter-blend candidates; development selection; paired sealed evaluation and
retention. Incomplete generations are excluded from reward updates. Banks with
uninformative rewards perform no update. Ptah reviews are deferred until Gemma
unloads. Cold-reload verification remains a separate follow-up after completion.

BF16 cross-shape calibration remains open; no release promotion or Qwen parity
claim follows this experiment. Do not modify pinned training sources while this
run is active. Numerical cache controls are in
`../../runtime-training-engine-20260922/`.
