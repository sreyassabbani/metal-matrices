use matrix_test::matrix::Matrix;

type FSquareMatrix<const M: usize> = Matrix<f32, M, M>;

fn main() {
    let mat1 = FSquareMatrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mat2 = FSquareMatrix::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let product = &mat1 * &mat2;
    println!("Product:\n{:?}", product);

    match mat1.gpu_multiply(&mat2) {
        Ok(gpu_product) => println!("Product with GPU:\n{:?}", gpu_product),
        Err(err) => eprintln!("GPU multiply failed: {err}"),
    }

    match product.get_vector(2) {
        Ok(col) => println!("Col 2 Vector:\n{:?}", col),
        Err(err) => eprintln!("Column extraction failed: {err}"),
    }
}
