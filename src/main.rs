use matrix_test::{matrix::Matrix, vector::Vector};

type FSquareMatrix<const M: usize> = Matrix<f32, M, M>;
type ISquareMatrix<const M: usize> = Matrix<i64, M, M>;

fn main() {
    let m = FSquareMatrix::<2>::from([[1.0, 1.0], [1.0, 1.0]]);

    // println!("Matrix 1:\n{:?}", m);
    let v1 = Vector::from([1, 2, 3]);
    let v2 = Vector::from([2, 2, 4]);

    // println!("{:?}", &v1 + &v2);
    // println!("{:?}", v1 + v2);
    let thing: &[i32; 3] = v1.as_ref();

    let mat1 = FSquareMatrix::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mat2 = FSquareMatrix::from([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);

    // println!("Matrix 2:\n{:?}", r);

    println!("Product:\n{:?}", &mat1 * &mat2);
    println!("Product with GPU:\n{:?}", mat1.gpu_multiply(&mat2));
    println!("Col 1 Vector:\n{:?}", (&mat1 * &mat2).get_vector(2));

    // let f = ISquareMatrix::<2>::from([[1, 1], [1, 0]]);
    // println!("Matrix 1 raised to the 4th:\n{:?}", &(&(&f * &f) * &f) * &f);

    // TODO: Implement multiplication for owned values
}
