pub mod matrix;

pub(crate) trait Numeric:
    std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + Copy
{
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    type SquareMatrix<const M: usize> = Matrix<M, M>;

    #[test]
    fn null_matrices() {
        let mat1 = Matrix::<0, 0>::null();
        let mat2 = Matrix::<0, 0>::null();
        let expected = Matrix::<0, 0>::null();

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn square_and_null_matrices() {
        let mat1 = SquareMatrix::<2>::null();
        let mat2 = SquareMatrix::<2>::new([[1.0, 3.0], [1.0, 2.0]]);
        let expected = SquareMatrix::<2>::null();

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn nonsq_and_nonsq_matrices() {
        let mat1 = Matrix::new([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
        let mat2 = Matrix::new([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);
        let expected = SquareMatrix::new([[12.0, 0.0], [29.0, -5.0]]);
        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn square_and_square_matrices() {
        let mat1 = SquareMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let mat2 = SquareMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let expected = SquareMatrix::new([[19.0, 22.0], [43.0, 50.0]]);
        assert_eq!(&mat1 * &mat2, expected);
    }
}
