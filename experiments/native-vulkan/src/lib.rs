//! Fast hardware-test entry point for the actual runtime model implementation.
//! The path import deliberately shares production source; there is no model fork.
//! Select one `installed_artifact_vulkan_*` test with `--ignored --test-threads=1`
//! and an explicit AKH_QWEN35_TEST_MODEL during an exclusive local GPU window.
#[cfg(test)]
#[path = "../../../src/nlu/vendored/quantized_qwen35.rs"]
mod qwen35_model;
