# AGENTS.md

Jevlike is a small PyTorch package for one-pass choice models. The text model takes a context and a changing list of text options. The visual model takes a 160 by 120 screen with motion and a subset of one shared 12-entry controller table. Doom uses rows 0–6; chess uses rows 7–11.

Keep the public package self-contained. Do not add machine-specific paths, private data, credentials, run logs or copyrighted audio. Keep downloaded datasets and generated runs out of git. Preserve the JSONL text format in the top-level README and the checkpoint fields used by the examples.

## Install and verify

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,games]'
pytest -q
PYTHONPATH=examples/doom pytest -q examples/doom/test_smoke.py
(cd examples/chess && python test_smoke.py)
```

The top-level README quickstart is the release test. Run it exactly before changing its commands. Core text code must keep working on CPU, MPS and CUDA. Game scripts are intentionally small and may need a new command-line device choice before using CUDA.

## Doom

The environment and seven-button mapping are in `examples/doom/environment.py`. The visual scorer is imported from `jevlike.vision`; do not copy it into the example.

Training order:

1. `train_imitation.py` collects heuristic-expert demonstrations from `deadly_corridor`.
2. `train_dagger.py` mixes expert and student actions, then labels the states the student actually visits. Prefer the expert's natural action frequency; class balancing suppressed attack in our runs.
3. `train_ppo.py` continues with parallel PPO, a value loss and a decaying entropy bonus.
4. `play.py` runs paced evaluation and can write 640 by 480 footage plus a JSON trace containing the live matrices.

The shaped reward is the `shaped_reward` function in `train_ppo.py`. Its weights are the `--reward-scale`, `--forward-bonus`, `--clear-path-forward-bonus`, `--clear-path-idle-penalty`, `--kill-bonus` and `--hit-bonus` flags. Report raw kills and corridor completion alongside shaped return. Compare with `play.py --random-policy`.

Run `audit.py` after each useful checkpoint. It compares the policy on real, blank, averaged and shuffled-patch frames, reports per-button probability ranges and attention entropy, and relates attack and turning to enemy position. Reward above random does not count as visual control if shuffled-patch KL and enemy-conditioned margins are near zero.

## Chess

Install Stockfish and put its executable on `PATH`. Run commands from `examples/chess`.

- `positions.py` makes Stockfish-labelled positions.
- `train.py` turns teacher moves into cursor-key sequences and trains rows 7–11.
- `dagger.py` labels recovery states visited by the current policy.
- `eval.py` measures play against a random mover and Stockfish, including wins, wasted presses, keys per move and key-budget failures.
- `play.py` records a paced film trace.

The frame is the full state. Sampling keys works better than greedy keys because a state-free repeat can otherwise loop until the 40-key budget. Do not describe the supplied checkpoint as good at chess: it learned the controller but loses almost every game to Stockfish level 0.

## One model for both games

`examples/doom/train_joint.py` starts from the two single-game checkpoints, keeps one visual encoder and one 12-row option table, and alternates natural-frequency Doom DAgger updates with chess recovery updates. The current recipe uses several chess updates for each Doom update. Select checkpoints by fixed held-out chess key accuracy and unassisted Doom play, not training-batch accuracy. `train_joint_ppo.py` is the more experimental mixed PPO loop; check its per-game reward scales before blaming the shared option table.

## Film

`play.py --trace` stores every chosen option, probability, attention map, activation matrix, frame and measured inference time. `examples/film/build-film.mjs` chooses or accepts one continuous window and writes `replay.html`. `render-film.mjs` drives `setFrame(t)` with Playwright and pipes 24 fps frames to ffmpeg. Use `make-film.sh TRACE OUTPUT [SECONDS] [EPISODE@START]`. The committed demo contains author-owned music, but its source MP3 is deliberately absent; new renders are silent unless you supply audio you may redistribute.

## What has been tried

Point your agent here and try your own RL; here is what has been tried and what failed:

- PPO from pixels learned a useful fixed action prior but almost ignored the screen. Attention was nearly uniform and shuffled patches barely changed the policy.
- A convolutional stem plus fixed two-dimensional positions added directly to the attention keys fixed the clearest representation bug. Keep both.
- Imitation and DAgger produced screen-dependent combat sooner than RL alone. This matches prior screen-control and game-agent work.
- A flat convolutional policy learned combat faster than the one-read-per-option head on identical DAgger data. The option head remains a capacity bottleneck, while zero corridor completions in both heads also show a data or navigation problem.
- Duplicating an attention read with identical weights did not help; symmetry-breaking noise still gave no clear gain.
- A short 20-update joint imitation run largely erased chess. Use hundreds of updates, weight the chess side, and gate on held-out accuracy.
- Log attention entropy during RL. If representations collapse again, distinguish an entropy-bonus problem from early-experience lock-in before trying a late-layer reset.
