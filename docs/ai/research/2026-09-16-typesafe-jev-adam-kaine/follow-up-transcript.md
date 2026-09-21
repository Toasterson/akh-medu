# Follow-up transcript: temporal straightening and Qwen

Verbatim user-visible messages from the two-link question through the request to save and ingest into Seshat. Exported from the same original session as transcript.md. Headings are editorial; message text is unchanged. Runtime instructions, internal reasoning, and tool payloads are excluded.

## User (message)

Did we find this LeCun paper mentioned here? https://x.com/HowToPrompt__/status/2100128683390636034 also there seems to be a qwen 2.5 try port https://x.com/NielsRogge/status/2100239244501430438

---

## Assistant (commentary)

I’ll check both posts, identify the paper and Qwen port, and compare them with the research we saved.

---

## Assistant (commentary)

Both are new additions to our notes. The first tweet reverses the paper’s method: the researchers straighten the learned representation to improve planning. The Qwen repository is an inference implementation using ordinary Qwen2.5 weights; I haven’t found a Jev training reproduction, and its softmax scores alone do not establish calibration.

---

## Assistant (final_answer)

**Both are new—we hadn’t included either in the saved research.** They point to different experiments.

**The LeCun paper is *Temporal Straightening for Latent Planning***, by Ying Wang and colleagues, with LeCun and Mengye Ren as advising authors. It was first posted in March 2026. [Paper](https://arxiv.org/abs/2603.12231)

The tweet reverses its central idea:

- **Tweet:** change the planning algorithms to follow curved representations.
- **Paper:** train the representations so feasible trajectories become straighter, making ordinary latent-distance planning work better.

They combine prediction training with a penalty on curvature between successive latent transitions. This improves planning on studied goal-reaching tasks; it does not establish that models already understand physics perfectly or eliminate hallucinations. [Author explanation and code](https://agenticlearning.ai/temporal-straightening/)

For **Kaine**, this is a useful world-model research direction: learn representations where distance corresponds better to achievable progress. Transfer to branching software workflows remains an experiment. It provides no evidence about Jev’s architecture.

**The Qwen project is a concrete implementation of our first avenue—direct decisions—but its name overstates what is demonstrated.**

I inspected the repository. It loads ordinary **Qwen2.5-1.5B-Instruct**, prefills the context once, batches field-specific queries using the cached context, scores allowed token choices, and constructs JSON programmatically. The inspected repository contains inference code, **not a released RL-trained Jev reproduction**. [Repository and model card](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)

Three findings matter before using it:

- **Softmax isn’t calibration.** The implementation labels normalized scores “calibrated” without demonstrating that their confidence matches observed accuracy.
- **Some choices can become indistinguishable.** Its schema compiler detects shared candidate-token IDs, but the inspected PyTorch scoring loop does not resolve those collisions.
- **Benchmark fairness needs checking.** Its generative baseline explicitly requests verbose, indented JSON; we should compare against compact constrained output. [Pinned implementation](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/2af86848be75847ccb3553b0941cc51d6ef7e4e9)

**My assessment:** the Qwen approach is the more immediate ADAM experiment: test how much improvement comes from changing inference alone, before training anything. Temporal straightening belongs in the separate Kaine world-model investigation.

I added both findings and the implementation caveats to the [Fable handoff](/Users/toasty/ws/akh-medu/docs/ai/research/2026-09-16-typesafe-jev-adam-kaine/README.md).

---

## User (message)

Write it down. Once in the repo also add it to seshat
