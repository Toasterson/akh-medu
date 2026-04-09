//! Benchmarks for Phase 26 components: SIMD, PolarQuant, Neural Bridge.

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use akh_medu::simd;
use akh_medu::vsa::neural_bridge::{LinearProjection, NeuralVsaBridge};
use akh_medu::vsa::polar_quant;
use akh_medu::vsa::Dimension;

// ---------------------------------------------------------------------------
// SIMD kernel benchmarks
// ---------------------------------------------------------------------------

fn bench_xor_bind_kernel(c: &mut Criterion) {
    let kernel = simd::best_kernel();
    let len = Dimension::DEFAULT.binary_byte_len(); // 1250 bytes
    let a = vec![0xAA_u8; len];
    let b = vec![0x55_u8; len];
    let mut out = vec![0u8; len];

    c.bench_function(&format!("xor_bind_10k_{}", kernel.isa_level()), |bench| {
        bench.iter(|| {
            kernel.xor_bind(black_box(&a), black_box(&b), black_box(&mut out));
        })
    });
}

fn bench_hamming_kernel(c: &mut Criterion) {
    let kernel = simd::best_kernel();
    let len = Dimension::DEFAULT.binary_byte_len();
    let a = vec![0xAA_u8; len];
    let b = vec![0x55_u8; len];

    c.bench_function(
        &format!("hamming_10k_{}", kernel.isa_level()),
        |bench| {
            bench.iter(|| black_box(kernel.hamming_distance(black_box(&a), black_box(&b))))
        },
    );
}

fn bench_cosine_kernel(c: &mut Criterion) {
    let kernel = simd::best_kernel();
    let len = 10_000;
    let a: Vec<i8> = (0..len).map(|i| ((i % 127) as i8).wrapping_sub(64)).collect();
    let b: Vec<i8> = (0..len).map(|i| ((i * 3 % 127) as i8).wrapping_sub(64)).collect();

    c.bench_function(
        &format!("cosine_i8_10k_{}", kernel.isa_level()),
        |bench| {
            bench.iter(|| black_box(kernel.cosine_similarity_i8(black_box(&a), black_box(&b))))
        },
    );
}

// ---------------------------------------------------------------------------
// FWHT benchmarks
// ---------------------------------------------------------------------------

fn bench_fwht_1024(c: &mut Criterion) {
    let mut data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();

    c.bench_function("fwht_1024", |bench| {
        bench.iter(|| {
            polar_quant::fwht_normalized(black_box(&mut data));
        })
    });
}

fn bench_fwht_16384(c: &mut Criterion) {
    let mut data: Vec<f32> = (0..16384).map(|i| (i as f32) * 0.001).collect();

    c.bench_function("fwht_16384", |bench| {
        bench.iter(|| {
            polar_quant::fwht_normalized(black_box(&mut data));
        })
    });
}

// ---------------------------------------------------------------------------
// PolarQuant benchmarks
// ---------------------------------------------------------------------------

fn bench_polar_quant_compress_1536(c: &mut Criterion) {
    let data: Vec<f32> = (0..1536).map(|i| ((i as f32) * 0.7).sin()).collect();

    c.bench_function("polarquant_compress_1536", |bench| {
        bench.iter(|| black_box(polar_quant::compress(black_box(&data), 42)))
    });
}

fn bench_polar_quant_compress_3584(c: &mut Criterion) {
    let data: Vec<f32> = (0..3584).map(|i| ((i as f32) * 0.7).sin()).collect();

    c.bench_function("polarquant_compress_3584", |bench| {
        bench.iter(|| black_box(polar_quant::compress(black_box(&data), 42)))
    });
}

fn bench_3bit_pack_unpack(c: &mut Criterion) {
    let values: Vec<f32> = (0..4096).map(|i| ((i as f32) * 0.3).sin()).collect();
    let qvec = polar_quant::quantize_vector(&values);

    c.bench_function("3bit_dequantize_4096", |bench| {
        bench.iter(|| black_box(polar_quant::dequantize_vector(black_box(&qvec))))
    });
}

// ---------------------------------------------------------------------------
// Neural bridge benchmarks
// ---------------------------------------------------------------------------

fn bench_neural_bridge_encode_1536(c: &mut Criterion) {
    let bridge = NeuralVsaBridge::new(1536, Dimension::DEFAULT);
    let hidden: Vec<f32> = (0..1536).map(|i| ((i as f32) * 0.01).sin()).collect();

    c.bench_function("neural_bridge_encode_1536_10k", |bench| {
        bench.iter(|| black_box(bridge.encode(black_box(&hidden))))
    });
}

fn bench_linear_projection_1536_10k(c: &mut Criterion) {
    let proj = LinearProjection::new(1536, 10_000);
    let input: Vec<f32> = (0..1536).map(|i| ((i as f32) * 0.01).sin()).collect();

    c.bench_function("linear_proj_1536_10k", |bench| {
        bench.iter(|| black_box(proj.forward(black_box(&input))))
    });
}

fn bench_neural_bridge_confidence(c: &mut Criterion) {
    let bridge = NeuralVsaBridge::new(1536, Dimension::DEFAULT);
    let hidden: Vec<f32> = (0..1536).map(|i| ((i as f32) * 0.01).sin()).collect();

    c.bench_function("neural_bridge_confidence_1536", |bench| {
        bench.iter(|| black_box(bridge.encoding_confidence(black_box(&hidden))))
    });
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    simd_benches,
    bench_xor_bind_kernel,
    bench_hamming_kernel,
    bench_cosine_kernel,
);

criterion_group!(
    fwht_benches,
    bench_fwht_1024,
    bench_fwht_16384,
);

criterion_group!(
    polar_quant_benches,
    bench_polar_quant_compress_1536,
    bench_polar_quant_compress_3584,
    bench_3bit_pack_unpack,
);

criterion_group!(
    neural_bridge_benches,
    bench_neural_bridge_encode_1536,
    bench_linear_projection_1536_10k,
    bench_neural_bridge_confidence,
);

criterion_main!(simd_benches, fwht_benches, polar_quant_benches, neural_bridge_benches);
