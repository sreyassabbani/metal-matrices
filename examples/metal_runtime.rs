use metal_matrices::{matrix::Matrix, metal::MetalRuntime};

fn matrices_close<const M: usize, const N: usize>(
    lhs: &Matrix<f32, M, N>,
    rhs: &Matrix<f32, M, N>,
    abs_tol: f32,
    rel_tol: f32,
) -> bool {
    lhs.as_row_major_slice()
        .iter()
        .zip(rhs.as_row_major_slice().iter())
        .all(|(lhs, rhs)| {
            let diff = (lhs - rhs).abs();
            diff <= abs_tol.max(rhs.abs() * rel_tol)
        })
}

fn main() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("Metal runtime unavailable: {err}");
            return;
        }
    };
    println!(
        "Metal runtime ready: shader={}, threadgroup={:?}",
        runtime.shader_kind(),
        runtime.threadgroup_shape()
    );

    let lhs = Matrix::<f32, 3, 3>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let rhs = Matrix::<f32, 3, 3>::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

    let cpu_product = lhs.multiply(&rhs);
    println!("CPU product:\n{cpu_product:?}");

    match runtime.multiply(&lhs, &rhs) {
        Ok(gpu_product) => {
            println!("GPU product:\n{gpu_product:?}");
            println!(
                "GPU matches CPU within tolerance: {}",
                matrices_close(&gpu_product, &cpu_product, 1e-4, 1e-5)
            );
        }
        Err(err) => eprintln!("GPU multiply failed: {err}"),
    }
}
