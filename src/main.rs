use matrix_test::matrix::Matrix;

type SquareMatrix<const M: usize> = Matrix<M, M>;

fn main() {
    let m = SquareMatrix::<2>::new([[1.0, 1.0], [1.0, 1.0]]);

    println!("Matrix 1:\n{:?}", m);

    let mut r = SquareMatrix::<2>::identity();
    r.set(0, 0, 90.0).unwrap();
    r.set(0, 1, 90.0).unwrap();

    println!("Matrix 2:\n{:?}", r);

    println!("Product:\n{:?}", &m * &r);

    // TODO: Implement multiplication for owned values
}
