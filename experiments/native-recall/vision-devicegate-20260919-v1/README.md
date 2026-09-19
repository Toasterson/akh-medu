# Vision projector cross-device gate, 2026-09-19

Projector `sha256-add205b7bfdb3f71...`, 64x64 deterministic input, 4 merged rows
of width 5120 (20,480 elements), 27 blocks. One process per device, run serially.

Reproduce:

    cargo build --release --features hip    --example vision_projector_probe
    ./target/release/examples/vision_projector_probe --model MODEL --device cpu \
        --side 64 --dump cpu.f32
    ./target/release/examples/vision_projector_probe --model MODEL --device hip \
        --side 64 --dump hip.f32
    cargo build --release --features vulkan --example vision_projector_probe
    ./target/release/examples/vision_projector_probe --model MODEL --device vulkan \
        --side 64 --dump vulkan.f32
    scripts/session-platform/compare-f32-dumps.py \
        cpu=cpu.f32 hip=hip.f32 vulkan=vulkan.f32

`gate.txt` is the result. What this does and does not establish is in
[the research note](../../docs/ai/research/2026-09-19-vision-device-gate.md).
