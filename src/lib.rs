pub mod matrix;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    type FSquareMatrix<const M: usize> = Matrix<f64, M, M>;
    type FMatrix<const M: usize, const N: usize> = Matrix<f64, M, N>;

    #[test]
    fn mul_empty_matrices() {
        let mat1 = FMatrix::<0, 0>::new([]);
        let mat2 = FMatrix::<0, 0>::new([]);
        let expected = FMatrix::<0, 0>::new([]);

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn mul_square_and_null_matrices() {
        let mat1 = FSquareMatrix::<2>::null();
        let mat2 = FSquareMatrix::<2>::new([[1.0, 3.0], [1.0, 2.0]]);
        let expected = FSquareMatrix::<2>::null();

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn mul_nonsq_and_nonsq_matrices() {
        let mat1 = FMatrix::new([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
        let mat2 = FMatrix::new([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);
        let expected = FSquareMatrix::new([[12.0, 0.0], [29.0, -5.0]]);

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn mul_square_and_square_matrices() {
        let mat1 = FSquareMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
        let mat2 = FSquareMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
        let expected = FSquareMatrix::new([[19.0, 22.0], [43.0, 50.0]]);

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn add_empty_matrices() {
        let mat1 = FMatrix::<0, 0>::new([]);
        let mat2 = FMatrix::<0, 0>::new([]);
        let expected = FMatrix::<0, 0>::new([]);

        assert_eq!(&mat1 + &mat2, expected);
    }

    #[test]
    fn add_nonsq_matrices() {
        let mat1 = FMatrix::<4, 3>::new([
            [1.0, 3.1, 5.9],
            [4.8, 9.2, 1.0],
            [2.0, 3.0, 4.0],
            [2.3, 4.3, 4.8],
        ]);
        let mat2 = FMatrix::<4, 3>::new([
            [1.0, 1.0, 2.0],
            [5.0, 9.0, 4.5],
            [-1.3, 9.1, -18.0],
            [3.0, -0.5, -0.1],
        ]);
        let expected = FMatrix::<4, 3>::new([
            [2.0, 4.1, 7.9],
            [9.8, 18.2, 5.5],
            [0.7, 12.1, -14.0],
            [5.3, 3.8, 4.7],
        ]);

        assert_eq!(&mat1 + &mat2, expected);
    }
}
