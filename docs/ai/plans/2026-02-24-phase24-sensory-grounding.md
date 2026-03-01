# Phase 24 — Sensory Grounding

> Date: 2026-02-24

- **Status**: Planned
- **Phase**: 24 (Lifeform Engine — Embodied Perception)
- **Depends on**: Phase 14m (NLU complete), Phase 23 (Affective System)
- **Enhances**: Knowledge graph grounding, concept formation, world model

## Goal

Give the agent basic sensory grounding — the ability to process images, audio, and
structured data from the physical world and ground symbolic concepts in perceptual
experience. Concepts without percepts are empty; this phase connects the symbolic
reasoning system to actual observations.

Must run on Mac Mini M2 with FLOSS models only. No cloud vision APIs.

## Architecture Overview

```
Raw sensory input (image, audio, structured data)
     │
     ▼
┌──────────────────────────────────────────┐
│ Perceptual Frontend                      │
│ Small vision model (SigLIP/MobileCLIP)   │
│ Audio features (mel spectrograms)        │
│ Structured data parsing                  │
└──────────────┬───────────────────────────┘
               │ Feature vectors
               ▼
┌──────────────────────────────────────────┐
│ VSA Grounding Bridge                     │
│ Feature → HyperVec encoding             │
│ Bind perceptual features with concepts   │
│ Similarity search against known symbols  │
└──────────────┬───────────────────────────┘
               │ Grounded symbols
               ▼
┌──────────────────────────────────────────┐
│ Knowledge Graph Integration              │
│ Perceptual triples: "saw X at time T"    │
│ Provenance: DerivationKind::Perceived    │
│ Affective appraisal of percepts          │
└──────────────────────────────────────────┘
```

## Sub-phases

### 24a — Image Perception (~1,000 lines)

Use a small CLIP-like model to extract image features and ground them in the KG.

**Model candidates** (FLOSS, runs on M2):
- MobileCLIP (Apple, MIT license): ~60 MB, image→text similarity
- SigLIP-base (Google, Apache 2.0): ~350 MB, strong zero-shot
- BLIP-2 (Salesforce, BSD): larger but produces captions

Run via `ort` (ONNX) or `candle` (pure Rust, Metal support on M2).

**Capabilities**:
- Image → feature vector → VSA encoding → concept grounding
- "What is in this image?" → KG entity matching
- Visual similarity: bind image features with known concept vectors
- Scene understanding: extract objects, spatial relations

### 24b — Audio Perception (~600 lines)

Basic audio processing: speech features, environmental sounds. Not speech-to-text
(that's a separate problem) — perceptual grounding of audio patterns.

**Approach**:
- Mel spectrogram extraction (pure Rust, no ML needed)
- Audio event classification via small model (AudioSet-style)
- VSA encoding of audio features for pattern matching

### 24c — Structured Data Perception (~400 lines)

Process structured sensory data: system metrics, sensor readings, API responses.
Ground numerical patterns in symbolic concepts.

**Examples**:
- CPU temperature → "system:hot" concept with somatic marker
- Network latency spike → "network:degraded" with negative valence
- Time of day → circadian awareness, activity patterns

## Memory Budget

| Component | Memory |
|-----------|--------|
| MobileCLIP (ONNX, quantized) | ~60 MB |
| Audio classifier (optional) | ~30 MB |
| VSA grounding bridge | ~0 MB (uses existing ops) |
| **Total** | **~90 MB** |

## Success Criteria

1. Agent can describe contents of an image using KG concepts
2. Image features grounded as VSA vectors, retrievable by similarity
3. Perceived entities linked to existing KG knowledge
4. Affective appraisal triggered by perceptual input
5. Runs on M2 with <200ms latency per image

## Non-Goals

- Real-time video processing
- Speech-to-text transcription (separate capability)
- Robotic motor control
- Generating images or audio
