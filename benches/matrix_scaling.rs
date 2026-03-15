use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use ndarray::Array2;

use metal_matrices::matrix::Matrix;
#[cfg(all(feature = "metal", target_os = "macos"))]
use metal_matrices::metal::MetalRuntime;

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

#[cfg(all(feature = "metal", target_os = "macos"))]
type MaybeRuntime<'a> = Option<&'a MetalRuntime>;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
type MaybeRuntime<'a> = Option<&'a ()>;

fn bench_size<const N: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    gpu_runtime: MaybeRuntime<'_>,
) {
    let mat_a = build_matrix::<N>(0.13);
    let mat_b = build_matrix::<N>(0.79);
    let arr_a = build_ndarray::<N>(0.13);
    let arr_b = build_ndarray::<N>(0.79);

    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Some(runtime) = gpu_runtime {
        group.bench_function(format!("gpu warm n={N}"), |b| {
            b.iter(|| {
                black_box(
                    runtime
                        .multiply(&mat_a, &mat_b)
                        .expect("warm GPU benchmark should succeed after runtime init"),
                )
            })
        });
    }

    group.bench_function(format!("cpu n={N}"), |b| {
        b.iter(|| black_box(mat_a.multiply(&mat_b)))
    });

    group.bench_function(format!("ndarray n={N}"), |b| {
        b.iter(|| black_box(arr_a.dot(&arr_b)))
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix multiplication scaling");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let gpu_runtime = match MetalRuntime::new() {
        Ok(runtime) => Some(runtime),
        Err(err) => {
            eprintln!("Skipping warm GPU scaling benchmarks: {err}");
            None
        }
    };

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let gpu_runtime: Option<()> = None;

    bench_size::<128>(&mut group, gpu_runtime.as_ref());
    bench_size::<192>(&mut group, gpu_runtime.as_ref());
    bench_size::<256>(&mut group, gpu_runtime.as_ref());
    bench_size::<320>(&mut group, gpu_runtime.as_ref());
    bench_size::<384>(&mut group, gpu_runtime.as_ref());
    bench_size::<512>(&mut group, gpu_runtime.as_ref());
    bench_size::<768>(&mut group, gpu_runtime.as_ref());
    bench_size::<1024>(&mut group, gpu_runtime.as_ref());

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
