use matrix_test::matrix::Matrix;

type FSquareMatrix<const M: usize> = Matrix<f32, M, M>;
type ISquareMatrix<const M: usize> = Matrix<i64, M, M>;

fn main() {
    let m = FSquareMatrix::<2>::new([[1.0, 1.0], [1.0, 1.0]]);

    // println!("Matrix 1:\n{:?}", m);

    let A = FSquareMatrix::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]);
    let B = FSquareMatrix::new([
        [9.0, 8.0, 7.0],
        [6.0, 5.0, 4.0],
        [3.0, 2.0, 1.0],
    ]);

    // println!("Matrix 2:\n{:?}", r);

    println!("Product:\n{:?}", &A * &B);
    println!("Product with GPU:\n{:?}", A.gpu_multiply(&B));

    // let f = ISquareMatrix::<2>::new([[1, 1], [1, 0]]);
    // println!("Matrix 1 raised to the 4th:\n{:?}", &(&(&f * &f) * &f) * &f);

    // TODO: Implement multiplication for owned values
}
