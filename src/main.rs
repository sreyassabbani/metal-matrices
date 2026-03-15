use metal_matrices::matrix::Matrix;
#[cfg(all(feature = "metal", target_os = "macos"))]
use metal_matrices::metal::MetalRuntime;

type FSquareMatrix<const M: usize> = Matrix<f32, M, M>;

#[cfg(all(feature = "metal", target_os = "macos"))]
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
    let mat1 = FSquareMatrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mat2 = FSquareMatrix::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

    let cpu_product = mat1.multiply(&mat2);
    println!("CPU product:\n{cpu_product:?}");

    #[cfg(all(feature = "metal", target_os = "macos"))]
    match MetalRuntime::new() {
        Ok(runtime) => match runtime.multiply(&mat1, &mat2) {
            Ok(product) => {
                println!("GPU product:\n{product:?}");
                println!(
                    "GPU matches CPU within tolerance: {}",
                    matrices_close(&product, &cpu_product, 1e-4, 1e-5)
                );
                match product.get_vector(2) {
                    Ok(col) => println!("Col 2 Vector:\n{col:?}"),
                    Err(err) => eprintln!("Column extraction failed: {err}"),
                }
            }
            Err(err) => eprintln!("GPU matrix multiplication failed: {err}"),
        },
        Err(err) => eprintln!("Metal runtime unavailable: {err}"),
    }

    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        eprintln!("GPU example path requires --features metal on macOS");
    }
}
