use matrix_test::{matrix::Matrix, metal::MetalRuntime};

type FSquareMatrix<const M: usize> = Matrix<f32, M, M>;

fn main() {
    let mat1 = FSquareMatrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mat2 = FSquareMatrix::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

    let cpu_product = mat1.multiply(&mat2);
    println!("CPU product:\n{cpu_product:?}");

    match MetalRuntime::new() {
        Ok(runtime) => match runtime.multiply(&mat1, &mat2) {
            Ok(product) => {
                println!("GPU product:\n{product:?}");
                match product.get_vector(2) {
                    Ok(col) => println!("Col 2 Vector:\n{col:?}"),
                    Err(err) => eprintln!("Column extraction failed: {err}"),
                }
            }
            Err(err) => eprintln!("GPU matrix multiplication failed: {err}"),
        },
        Err(err) => eprintln!("Metal runtime unavailable: {err}"),
    }
}
