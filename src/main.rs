use matrix_test::matrix::{ComputeOptions, Matrix};

type FSquareMatrix<const M: usize> = Matrix<f32, M, M>;

fn main() {
    let mat1 = FSquareMatrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mat2 = FSquareMatrix::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    match mat1.multiply_with(&mat2, ComputeOptions::auto().with_fallback(true)) {
        Ok(output) => {
            println!("Product using {:?}:\n{:?}", output.backend, output.matrix);
            match output.matrix.get_vector(2) {
                Ok(col) => println!("Col 2 Vector:\n{:?}", col),
                Err(err) => eprintln!("Column extraction failed: {err}"),
            }
        }
        Err(err) => eprintln!("Matrix multiplication failed: {err}"),
    }
}
