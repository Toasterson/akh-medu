//! Phase 45a decode-throughput harness (ADR 073).
//!
//! Establishes the **baseline tok/s** the full `Device::Hip` backend must beat, and
//! reports it as a **% of the memory-bandwidth roofline** — the honest "close to
//! hardware capability" metric, since decode is bandwidth-bound (every weight is
//! streamed once per token).
//!
//! ```sh
//! # op-offload path on the GPU (the current ceiling):
//! AKH_LLM_MODEL=gemma3-12b AKH_GPU_MATMUL=1 AKH_GPU_QUANT=1 \
//!   cargo run --release --example candle_bench --features rocm
//!
//! # pure-CPU contrast:
//! AKH_LLM_MODEL=gemma3-12b cargo run --release --example candle_bench --features candle-backend
//! ```
//!
//! Env knobs: `AKH_BENCH_TOKENS` (default 128), `AKH_BENCH_PROMPT`,
//! `AKH_BENCH_BANDWIDTH_GBS` (default 960 = RX 7900 XTX). Set `AKH_HIP_TRACE=1` is
//! not required — PCIe counters are always live; they are reset after load here.
//!
//! HIP caching-allocator pool (Phase 46d sweep): the default is now ADAPTIVE — a 512 MB
//! floor that grows while ≥ 2.5 GB VRAM stays free (live `hipMemGetInfo`). On the resident
//! `Device::Hip` decode path the per-token transient working set is ~3 GB, so the old fixed
//! 512 MB cap thrashed (7.25 tok/s); the adaptive default auto-reaches the plateau —
//! gemma3-12b 10.4 tok/s (+44%), gemma4-12b 9.6 (+35%) — and falls back to the floor under
//! VRAM pressure (multi-akh / op-offload / 27B), so it never OOMs. Override with
//! `AKH_HIP_POOL_MB=<n>` (plain fixed cap; `=512` restores the old behavior) or
//! `AKH_HIP_POOL_MARGIN_MB=<n>` (tune the headroom; `=0` ⇒ pure fixed floor).

#[cfg(feature = "candle-backend")]
fn main() -> miette::Result<()> {
    use akh_medu::nlu::candle_backend::CandleBackend;
    use akh_medu::paths::AkhPaths;
    use std::time::Instant;

    let prompt = std::env::var("AKH_BENCH_PROMPT").unwrap_or_else(|_| {
        "Explain, in detail and step by step, how photosynthesis converts sunlight \
         into chemical energy inside a plant cell."
            .to_string()
    });
    let max_tokens: u32 = env_parse("AKH_BENCH_TOKENS").unwrap_or(128);
    let bandwidth_gbs: f64 = env_parse("AKH_BENCH_BANDWIDTH_GBS").unwrap_or(960.0);

    let paths = AkhPaths::resolve()?;
    let models = paths.models_dir();
    println!("models dir: {}", models.display());

    // Weight-bytes denominator for the roofline = the chosen GGUF's on-disk size
    // (quantized weight bytes streamed per token). Selection mirrors
    // `CandleBackend::try_load` loosely (AKH_LLM_MODEL hint, else first .gguf).
    let gguf = pick_gguf(&models);
    let weight_bytes = gguf
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    let model_name = gguf
        .as_ref()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<unknown>".into());

    let gpu_on = std::env::var("AKH_GPU_MATMUL").as_deref() == Ok("1");
    let quant_on = std::env::var("AKH_GPU_QUANT").as_deref() == Ok("1");

    let Some(model_path) = gguf.clone() else {
        eprintln!(
            "No Candle model found under {}/llm/. Place a .gguf + tokenizer.json.",
            models.display()
        );
        std::process::exit(1);
    };
    // Tokenizer beside the model: `<stem>.tokenizer.json`, else `tokenizer.json`.
    let stem_tok = model_path.with_extension("tokenizer.json");
    let tokenizer_path = if stem_tok.exists() {
        stem_tok
    } else {
        model_path.with_file_name("tokenizer.json")
    };
    // Load directly (not `try_load`, which swallows load errors into a trace line the
    // bench can't see) so any HIP/op failure surfaces with its full miette diagnostic.
    let load_start = Instant::now();
    let mut backend = CandleBackend::load(&model_path, &tokenizer_path)?;
    let load_secs = load_start.elapsed().as_secs_f32();
    println!(
        "model: {model_name}  family: {:?}  weights: {:.2} GB  load: {:.1}s  \
         [AKH_GPU_MATMUL={} AKH_GPU_QUANT={}]",
        backend.family(),
        weight_bytes as f64 / 1e9,
        load_secs,
        gpu_on as u8,
        quant_on as u8,
    );

    // Reset PCIe counters AFTER load so one-time resident-weight uploads do not
    // count toward per-token crossings. Two sources: akh's op-offload DeviceBuffer
    // (`nlu::hip`) and — for the resident Device::Hip path (Phase 45f) — candle-core's
    // own HIP runtime, where the resident forward's H2D/D2H actually happen.
    #[cfg(feature = "rocm")]
    akh_medu::nlu::hip::reset_pcie_counts();
    #[cfg(feature = "hip")]
    candle_core::cuda::reset_pcie_counts();
    #[cfg(feature = "hip")]
    candle_core::cuda::reset_malloc_counts();

    // Phase 46d-2c: byte-exact A/B of the graph-ready decode forward (device-pos rope +
    // kv_scatter_at + block-padded read_len + device-built keep-mask, run EAGERLY) vs the
    // normal decode forward at a fixed position. The gate for the per-token graph decode:
    // both diffs must be 0 before 46d-2d wraps the forward in a captured graph.
    #[cfg(feature = "hip")]
    if std::env::var("AKH_BENCH_GRAPH_PARITY").as_deref() == Ok("1") {
        let p = backend.bench_graph_decode_parity(&prompt)?;
        let verdict = |d: f32| if d == 0.0 { "BYTE-EXACT" } else { "DIVERGED" };
        println!();
        println!("=== Phase 46d-2c graph-ready decode parity (eager A/B) ===");
        println!("position (P)        : {} tokens", p.position);
        println!(
            "device KV write     : {}",
            if p.used_device_scatter {
                "kv_scatter_at (device *d_pos)"
            } else {
                "slice_set (host offset)"
            }
        );
        println!(
            "exact  read_len=P+1 : max|Δlogits| {:.3e}  ({})",
            p.exact_diff,
            verdict(p.exact_diff)
        );
        println!(
            "padded read_len={:<5}: max|Δlogits| {:.3e}  ({})",
            p.padded_read_len,
            p.padded_diff,
            verdict(p.padded_diff)
        );
        return Ok(());
    }

    // Phase 46c: fixed-position HIP-graph capture/replay probe (AKH_BENCH_GRAPH=1). The
    // decisive measurement — eager forward vs single-dispatch graph replay at one fixed
    // decode position — for whether graph replay can close the dispatch-bound decode gap.
    #[cfg(feature = "hip")]
    if std::env::var("AKH_BENCH_GRAPH").as_deref() == Ok("1") {
        let iters: u32 = env_parse("AKH_BENCH_TOKENS").unwrap_or(128);
        let stats = backend.bench_graph_replay(&prompt, iters)?;
        println!();
        println!("=== Phase 46c fixed-position HIP-graph replay probe ===");
        println!("position (P)       : {} tokens", stats.position);
        println!("iters (each)       : {}", stats.iters);
        println!(
            "warm fwd allocs    : {} miss / {} calls  (0 miss ⇒ forward is capturable)",
            stats.warm_malloc_miss, stats.warm_malloc_calls
        );
        println!("eager  (fwd @P)    : {:.2} tok/s", stats.eager_tok_s);
        println!(
            "stream-eager diff  : max|private−null| {:.3e}  (0 ⇒ all ops honor the stream)",
            stats.stream_eager_diff
        );
        if stats.captured {
            let speedup = stats.replay_tok_s / stats.eager_tok_s.max(1e-6);
            println!("replay (graph @P)  : {:.2} tok/s", stats.replay_tok_s);
            println!("speedup            : {speedup:.2}x  (replay / eager)");
            println!(
                "parity (replay≟fresh): argmax {} | max|Δlogits| {:.3e}",
                if stats.argmax_matches { "MATCH" } else { "MISMATCH" },
                stats.max_abs_diff
            );
            println!(
                "replay determinism : max|first−last| {:.3e}  (0 ⇒ deterministic)",
                stats.replay_determinism
            );
        } else {
            println!(
                "capture            : FAILED — {}",
                stats.capture_error.as_deref().unwrap_or("unknown")
            );
        }
        return Ok(());
    }

    let vram_before = vram_used_bytes();

    let stats = backend.generate_bench(&prompt, max_tokens)?;
    let tok_s = stats.tok_per_s();

    let vram_after = vram_used_bytes();

    #[cfg(feature = "rocm")]
    let (h2d, d2h) = {
        let (h, d) = akh_medu::nlu::hip::pcie_counts();
        // Add the resident-path crossings from candle-core's HIP runtime (the cuda slot).
        #[cfg(feature = "hip")]
        let (rh, rd) = candle_core::cuda::pcie_counts();
        #[cfg(not(feature = "hip"))]
        let (rh, rd) = (0u64, 0u64);
        (h + rh, d + rd)
    };
    #[cfg(not(feature = "rocm"))]
    let (h2d, d2h) = (0u64, 0u64);
    let crossings = h2d + d2h;
    let per_tok = if stats.tokens_generated > 0 {
        crossings as f64 / stats.tokens_generated as f64
    } else {
        0.0
    };

    // Bandwidth roofline: bytes streamed/token ≈ weight_bytes ⇒ max tok/s = BW / weights.
    let roofline = if weight_bytes > 0 {
        bandwidth_gbs * 1e9 / weight_bytes as f64
    } else {
        0.0
    };
    let pct = if roofline > 0.0 {
        100.0 * tok_s as f64 / roofline
    } else {
        0.0
    };

    println!();
    println!("=== Phase 45a decode benchmark ===");
    println!("tokens decoded     : {}", stats.tokens_generated);
    println!("decode wall        : {:.3} s", stats.decode_secs);
    println!("throughput         : {tok_s:.2} tok/s");
    println!("PCIe crossings     : {crossings} (h2d {h2d}, d2h {d2h}) = {per_tok:.1}/token");
    println!(
        "VRAM used          : {:.2} GB (Δ {:+.2} GB during gen)",
        vram_after as f64 / 1e9,
        (vram_after as f64 - vram_before as f64) / 1e9
    );
    println!(
        "bandwidth roofline : {roofline:.1} tok/s  (@ {bandwidth_gbs:.0} GB/s ÷ {:.2} GB weights)",
        weight_bytes as f64 / 1e9
    );
    println!("% of roofline      : {pct:.1}%");
    // Fused-GEMV profile (AKH_HIP_PROFILE=1): the intrinsic time in quantized matmul
    // vs everything else — shows whether the GEMV or the rest of the forward dominates.
    #[cfg(feature = "hip")]
    {
        let (gemv_ns, gemv_calls) = candle_core::quantized::gemv_profile();
        if gemv_calls > 0 {
            let wall = stats.decode_secs as f64;
            let (mm_ns, elt_ns, cp_ns, norm_ns, red_ns) = candle_core::cuda::op_profile();
            let pct = |ns: u64| 100.0 * (ns as f64 / 1e9) / wall;
            let gemv_s = gemv_ns as f64 / 1e9;
            println!(
                "GEMV (quantized)   : {:.3} s in {} calls = {:.1}% ({:.1} µs/call)",
                gemv_s,
                gemv_calls,
                100.0 * gemv_s / wall,
                gemv_ns as f64 / 1e3 / gemv_calls as f64,
            );
            // NB: GEMV/rms_norm/reduce/copy windows are clean (just the kernel launch);
            // matmul/elementwise use the whole-op-body window, so they INCLUDE the
            // host-enqueue idle gap and over-count. The category sum can exceed wall —
            // that gap reflects how launch/dispatch-bound (not compute-bound) decode is.
            println!("matmul (attention)*: {:.3} s = {:.1}%", mm_ns as f64 / 1e9, pct(mm_ns));
            println!("rms_norm (fused)   : {:.3} s = {:.1}%", norm_ns as f64 / 1e9, pct(norm_ns));
            println!("reduce (softmax)   : {:.3} s = {:.1}%", red_ns as f64 / 1e9, pct(red_ns));
            println!("elementwise*       : {:.3} s = {:.1}%", elt_ns as f64 / 1e9, pct(elt_ns));
            println!("copy (KV-cat+strd) : {:.3} s = {:.1}%", cp_ns as f64 / 1e9, pct(cp_ns));
            let measured = (gemv_ns + mm_ns + elt_ns + cp_ns + norm_ns + red_ns) as f64 / 1e9;
            let ratio = 100.0 * measured / wall;
            println!(
                "Σ categories       : {:.1}% of wall  (* = whole-op window incl. host-enqueue gap)",
                ratio
            );
            // Allocator stall: how much wall time the per-op output `hipMalloc` (a
            // device sync on cache miss) cost, and the miss rate per token. A high
            // share here means the bottleneck is malloc-sync, not kernel compute.
            let (mcalls, mmiss, mns) = candle_core::cuda::malloc_profile();
            let toks = stats.tokens_generated.max(1) as f64;
            println!(
                "malloc stall       : {:.3} s = {:.1}% ({} miss / {} calls, {:.1} miss/token)",
                mns as f64 / 1e9,
                100.0 * (mns as f64 / 1e9) / wall,
                mmiss,
                mcalls,
                mmiss as f64 / toks,
            );
        }
    }
    println!();
    let sample: String = stats.text.chars().take(180).collect::<String>().replace('\n', " ");
    println!("sample: {sample}");
    Ok(())
}

/// Parse an env var into any `FromStr` type, returning `None` if unset/invalid.
#[cfg(feature = "candle-backend")]
fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|s| s.parse().ok())
}

/// Find the GGUF the backend will load (hint-aware), so we can read its size for
/// the roofline. Checks `<models>/llm` and `<models>/models/llm` (the loader
/// tolerates both layouts). Excludes `*.mmproj.gguf` projector sidecars.
#[cfg(feature = "candle-backend")]
fn pick_gguf(models: &std::path::Path) -> Option<std::path::PathBuf> {
    let hint = std::env::var("AKH_LLM_MODEL").ok();
    for sub in ["llm", "models/llm"] {
        let dir = models.join(sub);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut ggufs: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("gguf")
                    && !p.to_string_lossy().contains(".mmproj.")
            })
            .collect();
        ggufs.sort();
        if let Some(h) = &hint {
            if let Some(m) = ggufs.iter().find(|p| p.to_string_lossy().contains(h.as_str())) {
                return Some(m.clone());
            }
        }
        if let Some(first) = ggufs.into_iter().next() {
            return Some(first);
        }
    }
    None
}

/// AMD VRAM-in-use, in bytes, via sysfs (`mem_info_vram_used`). 0 if unavailable
/// (rocm-smi is not installed on this box — sysfs is the source of truth).
#[cfg(feature = "candle-backend")]
fn vram_used_bytes() -> u64 {
    for card in ["card0", "card1"] {
        let path = format!("/sys/class/drm/{card}/device/mem_info_vram_used");
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(v) = s.trim().parse::<u64>() {
                return v;
            }
        }
    }
    0
}

#[cfg(not(feature = "candle-backend"))]
fn main() {
    eprintln!("This example requires the `candle-backend` feature.");
    std::process::exit(1);
}
