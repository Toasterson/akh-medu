//! Benchmarks for VSA operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::SeedableRng;

use akh_medu::simd;
use akh_medu::vsa::ops::VsaOps;
use akh_medu::vsa::{Dimension, Encoding};

fn bench_bind(c: &mut Criterion) {
    let ops = VsaOps::new(simd::best_kernel(), Dimension::DEFAULT, Encoding::Bipolar);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let a = ops.random(&mut rng);
    let b = ops.random(&mut rng);

    c.bench_function("bind_10k", |bench| {
        bench.iter(|| black_box(ops.bind(&a, &b).unwrap()))
    });
}

fn bench_bundle(c: &mut Criterion) {
    let ops = VsaOps::new(simd::best_kernel(), Dimension::DEFAULT, Encoding::Bipolar);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let vecs: Vec<_> = (0..10).map(|_| ops.random(&mut rng)).collect();
    let refs: Vec<&_> = vecs.iter().collect();

    c.bench_function("bundle_10x10k", |bench| {
        bench.iter(|| black_box(ops.bundle(&refs).unwrap()))
    });
}

fn bench_similarity(c: &mut Criterion) {
    let ops = VsaOps::new(simd::best_kernel(), Dimension::DEFAULT, Encoding::Bipolar);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let a = ops.random(&mut rng);
    let b = ops.random(&mut rng);

    c.bench_function("similarity_10k", |bench| {
        bench.iter(|| black_box(ops.similarity(&a, &b).unwrap()))
    });
}

criterion_group!(benches, bench_bind, bench_bundle, bench_similarity);
criterion_main!(benches);
