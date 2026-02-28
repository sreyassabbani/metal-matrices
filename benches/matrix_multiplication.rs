use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::arr2;

use matrix_test::matrix::Matrix;

use std::time::Duration;

fn criterion_benchmark(c: &mut Criterion) {
    let mat1 = Matrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mat2 = Matrix::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let mata = arr2(&[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let matb = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let mut group = c.benchmark_group("matrix multiplication");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("gpu matrix multiply", |b| {
        b.iter(|| black_box(mat2.gpu_multiply(&mat2)))
    });

    group.bench_function("naive matrix multiply", |b| {
        b.iter(|| black_box(&mat1) * black_box(&mat2))
    });

    group.bench_function("ndarray multiply", |b| b.iter(|| mata.dot(&matb)));

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
