# Doom example

This example uses `jevlike.vision.DoomScorerV2` to choose a controller button from a game screen. It is the same visual model used by the chess example. One table holds 12 option vectors: seven Doom buttons and five chess keys.

The input is a 160 by 120 RGB frame plus one motion channel made from the previous frame. A small convolutional stem makes 80 image patches. Fixed two-dimensional positions are added to the attention keys. Each active controller option reads the patches once and receives one score. A softmax turns the seven Doom scores into probabilities.

## Install

From the repository root:

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,games]'
```

## Evaluate the supplied checkpoints

The controller is turn left, turn right, forward, backward, strafe left, strafe right and attack. Evaluation is paced at 8.75 decisions per second. The policy still sees 160 by 120 input when footage is captured at 640 by 480.

```sh
python examples/doom/play.py examples/doom/checkpoints/deadly-dagger.pt \
  --episodes 10 --game-seconds 60 --device cpu --greedy
python examples/doom/play.py examples/checkpoints/joint-imitation.pt \
  --episodes 10 --game-seconds 60 --device cpu --capture-resolution 640x480 \
  --output runs/doom.mp4 --trace runs/doom-trace.json
```

The single-game checkpoint was trained by imitation and DAgger. DAgger means repeatedly collecting the states visited by the current policy and asking an expert for the right action there. The joint checkpoint has the same weights and one 12-row option table for both games. It is an early checkpoint, not a strong player.

## Train

Pure pixel PPO is included for experimentation, but it was a poor starting point. The more reliable order was imitation, DAgger, then PPO. The heuristic expert uses the object labels supplied by ViZDoom only to make training labels; the policy itself receives pixels and motion.

```sh
python examples/doom/train_imitation.py \
  --episodes 500 --envs 16 --device cpu --output runs/doom-imitation.pt
python examples/doom/train_dagger.py \
  --initialise-from runs/doom-imitation.pt \
  --episodes 500 --envs 16 --device cpu --no-class-balance \
  --output runs/doom-dagger.pt
python examples/doom/train_ppo.py \
  --initialise-from runs/doom-dagger.pt \
  --episodes 2000 --envs 16 --scenario deadly_corridor --device cpu \
  --output runs/doom-ppo.pt
```

Use `--device mps` on Apple silicon. CUDA works through PyTorch, but these small scripts currently expose only CPU and MPS in their command-line choices; adding a CUDA choice requires no model change.

## Audit screen dependence

High reward can come from a fixed action habit. The audit compares predictions on real, blank, averaged and shuffled-patch frames. It also checks whether attack probability rises when an enemy is centred and whether attention remains non-uniform.

```sh
python examples/doom/audit.py examples/doom/checkpoints/deadly-dagger.pt \
  --frames 500 --device cpu --output runs/doom-audit.json
```

## Joint training

First generate chess positions as described in the [chess example](../chess/README.md). Then alternate Doom DAgger batches with more heavily weighted chess batches:

```sh
python examples/doom/train_joint.py \
  --doom-checkpoint examples/doom/checkpoints/deadly-dagger.pt \
  --chess-checkpoint examples/chess/checkpoints/chess-dagger1.pt \
  --updates 500 --chess-steps 5 --device cpu --output runs/joint.pt
```

`train_joint_ppo.py` is the experimental mixed PPO and chess-supervision loop. The short released joint checkpoint lost much of the chess controller, so use the fixed held-out chess accuracy printed by `train_joint.py` as a gate before replacing it.
