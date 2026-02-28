use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use ndarray::Array2;

use matrix_test::matrix::{ComputeOptions, Matrix};

use std::time::Duration;

fn matrix_value<const N: usize>(row: usize, col: usize, seed: f32) -> f32 {
    // Deterministic values keep runs reproducible.
    (((row * N + col) as f32) * 0.0007 + seed).sin()
}

fn build_matrix<const N: usize>(seed: f32) -> Matrix<f32, N, N> {
    Matrix::from_fn(|row, col| matrix_value::<N>(row, col, seed))
}

fn build_ndarray<const N: usize>(seed: f32) -> Array2<f32> {
    Array2::from_shape_fn((N, N), |(row, col)| matrix_value::<N>(row, col, seed))
}

fn bench_size<const N: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let mat_a = build_matrix::<N>(0.13);
    let mat_b = build_matrix::<N>(0.79);
    let arr_a = build_ndarray::<N>(0.13);
    let arr_b = build_ndarray::<N>(0.79);

    let gpu_options = ComputeOptions::gpu();
    if mat_a.multiply_with(&mat_b, gpu_options).is_ok() {
        group.bench_function(format!("gpu n={N}"), |b| {
            b.iter(|| {
                black_box(
                    mat_a
                        .multiply_with(&mat_b, gpu_options)
                        .expect("GPU benchmark preflight should guarantee availability"),
                )
            })
        });
    } else {
        eprintln!("Skipping GPU benchmark for n={N}: Metal backend unavailable");
    }

    let cpu_options = ComputeOptions::cpu();
    group.bench_function(format!("cpu n={N}"), |b| {
        b.iter(|| {
            black_box(
                mat_a
                    .multiply_with(&mat_b, cpu_options)
                    .expect("CPU backend path should be infallible"),
            )
        })
    });

    group.bench_function(format!("ndarray n={N}"), |b| {
        b.iter(|| black_box(arr_a.dot(&arr_b)))
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix multiplication scaling");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    bench_size::<128>(&mut group);
    bench_size::<256>(&mut group);
    bench_size::<512>(&mut group);
    bench_size::<1024>(&mut group);

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
