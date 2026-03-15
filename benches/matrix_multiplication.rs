use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;

use matrix_test::{matrix::Matrix, metal::MetalRuntime};

use std::time::Duration;

const SIZE: usize = 256;

fn matrix_value(row: usize, col: usize, seed: f32) -> f32 {
    // Deterministic dense values to keep benchmark inputs stable across runs.
    (((row * SIZE + col) as f32) * 0.0007 + seed).sin()
}

fn build_matrix(seed: f32) -> Matrix<f32, SIZE, SIZE> {
    let mut entries = [[0.0_f32; SIZE]; SIZE];
    for (row, row_entries) in entries.iter_mut().enumerate() {
        for (col, entry) in row_entries.iter_mut().enumerate() {
            *entry = matrix_value(row, col, seed);
        }
    }
    Matrix::from(entries)
}

fn build_ndarray(seed: f32) -> Array2<f32> {
    Array2::from_shape_fn((SIZE, SIZE), |(row, col)| matrix_value(row, col, seed))
}

fn criterion_benchmark(c: &mut Criterion) {
    let mat1 = build_matrix(0.13);
    let mat2 = build_matrix(0.79);
    let mata = build_ndarray(0.79);
    let matb = build_ndarray(0.13);

    let mut group = c.benchmark_group("matrix multiplication");
    group.measurement_time(Duration::from_secs(10));

    match MetalRuntime::new() {
        Ok(runtime) => {
            group.bench_function("gpu warm matrix multiply", |b| {
                b.iter(|| black_box(runtime.multiply(&mat1, &mat2).expect("warm GPU path should succeed")))
            });
            group.bench_function("gpu cold init + multiply", |b| {
                b.iter(|| {
                    let runtime = MetalRuntime::new()
                        .expect("cold GPU benchmark requires runtime initialization to succeed");
                    black_box(
                        runtime
                            .multiply(&mat1, &mat2)
                            .expect("cold GPU benchmark multiply should succeed"),
                    )
                })
            });
        }
        Err(err) => {
            eprintln!("Skipping GPU benchmarks: {err}");
        }
    }

    group.bench_function("cpu matrix multiply", |b| {
        b.iter(|| {
            black_box(mat1.multiply(&mat2))
        })
    });

    group.bench_function("ndarray multiply", |b| b.iter(|| mata.dot(&matb)));

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
