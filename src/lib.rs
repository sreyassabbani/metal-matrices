pub mod matrix;
pub mod metal;
pub mod numeric;
pub mod vector;

#[cfg(test)]
mod tests {
    use crate::{
        matrix::{Matrix, MatrixError},
        metal::{MetalError, MetalRuntime},
        vector::Vector,
    };

    type FSquareMatrix<const M: usize> = Matrix<f64, M, M>;
    type FMatrix<const M: usize, const N: usize> = Matrix<f64, M, N>;

    #[test]
    fn matrix_from_fn_builds_expected_entries() {
        let matrix = Matrix::<i32, 2, 3>::from_fn(|row, col| (row as i32 * 10) + col as i32);

        assert_eq!(matrix, Matrix::from([[0, 1, 2], [10, 11, 12]]));
    }

    #[test]
    fn vector_from_fn_builds_expected_entries() {
        let vector = Vector::<i32, 4>::from_fn(|idx| idx as i32 * 2);

        assert_eq!(vector, Vector::from([0, 2, 4, 6]));
    }

    #[test]
    fn mul_empty_matrices() {
        let mat1 = FMatrix::<0, 0>::from([]);
        let mat2 = FMatrix::<0, 0>::from([]);
        let expected = FMatrix::<0, 0>::from([]);

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn mul_square_and_null_matrices() {
        let mat1 = FSquareMatrix::<2>::null();
        let mat2 = FSquareMatrix::<2>::from([[1.0, 3.0], [1.0, 2.0]]);
        let expected = FSquareMatrix::<2>::null();

        assert_eq!(mat1.multiply(&mat2), expected);
    }

    #[test]
    fn mul_nonsq_and_nonsq_matrices() {
        let mat1 = FMatrix::from([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
        let mat2 = FMatrix::from([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);
        let expected = FSquareMatrix::from([[12.0, 0.0], [29.0, -5.0]]);

        assert_eq!(mat1.multiply(&mat2), expected);
    }

    #[test]
    fn mul_square_and_square_matrices() {
        let mat1 = FSquareMatrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let mat2 = FSquareMatrix::from([[5.0, 6.0], [7.0, 8.0]]);
        let expected = FSquareMatrix::from([[19.0, 22.0], [43.0, 50.0]]);

        assert_eq!(mat1.multiply(&mat2), expected);
    }

    #[test]
    fn add_empty_matrices() {
        let mat1 = FMatrix::<0, 0>::from([]);
        let mat2 = FMatrix::<0, 0>::from([]);
        let expected = FMatrix::<0, 0>::from([]);

        assert_eq!(&mat1 + &mat2, expected);
    }

    #[test]
    fn add_nonsq_matrices() {
        let mat1 = FMatrix::<4, 3>::from([
            [1.0, 3.1, 5.9],
            [4.8, 9.2, 1.0],
            [2.0, 3.0, 4.0],
            [2.3, 4.3, 4.8],
        ]);
        let mat2 = FMatrix::<4, 3>::from([
            [1.0, 1.0, 2.0],
            [5.0, 9.0, 4.5],
            [-1.3, 9.1, -18.0],
            [3.0, -0.5, -0.1],
        ]);
        let expected = FMatrix::<4, 3>::from([
            [2.0, 4.1, 7.9],
            [9.8, 18.2, 5.5],
            [0.7, 12.1, -14.0],
            [5.3, 3.8, 4.7],
        ]);

        assert_eq!(&mat1 + &mat2, expected);
    }

    #[test]
    fn sub_nonsq_matrices() {
        let mat1 = FMatrix::<2, 2>::from([[5.0, 4.0], [3.0, 2.0]]);
        let mat2 = FMatrix::<2, 2>::from([[1.0, 1.5], [0.5, 2.0]]);

        assert_eq!(&mat1 - &mat2, FMatrix::<2, 2>::from([[4.0, 2.5], [2.5, 0.0]]));
    }

    #[test]
    fn transpose_returns_expected_matrix() {
        let mat = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(mat.transpose(), Matrix::<i32, 3, 2>::from([[1, 4], [2, 5], [3, 6]]));
    }

    #[test]
    fn identity_returns_expected_matrix() {
        assert_eq!(
            Matrix::<i32, 3, 3>::identity(),
            Matrix::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        );
    }

    #[test]
    fn get_matrix_entry() {
        let mat = FMatrix::<3, 3>::from([[1.0, 2.0, 3.0], [42.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        assert_eq!(mat.get(1, 0).unwrap(), 42.0);
    }

    #[test]
    fn set_matrix_entry() {
        let mut mat = FMatrix::<2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);
        mat.set(0, 1, 20.0).unwrap();

        assert_eq!(mat.get(0, 1).unwrap(), 20.0);
        assert_eq!(mat, FMatrix::<2, 2>::from([[1.0, 20.0], [3.0, 4.0]]));
    }

    #[test]
    fn get_out_of_bounds_returns_error() {
        let mat = FMatrix::<2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);

        assert_eq!(
            mat.get(2, 0),
            Err(MatrixError::OutOfBounds {
                row_found: 2,
                row_max: 2,
                col_found: 0,
                col_max: 2,
            })
        );
        assert!(matches!(
            mat.get(0, 2),
            Err(MatrixError::OutOfBounds {
                row_found: 0,
                row_max: 2,
                col_found: 2,
                col_max: 2,
            })
        ));
    }

    #[test]
    fn get_vector_returns_column() {
        let mat = FMatrix::<3, 3>::from([[1.0, 2.0, 3.0], [42.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        assert_eq!(mat.get_vector(0).unwrap(), Vector::from([1.0, 42.0, 7.0]));
    }

    #[test]
    fn get_vector_out_of_bounds_returns_error() {
        let mat = FMatrix::<3, 2>::from([[1.0, 2.0], [42.0, 5.0], [7.0, 8.0]]);

        assert_eq!(
            mat.get_vector(2),
            Err(MatrixError::ColumnOutOfBounds {
                col_found: 2,
                col_max: 2,
            })
        );
    }

    #[test]
    fn determinant_and_inverse_are_explicitly_unimplemented() {
        let mat = Matrix::<f64, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);

        assert!(matches!(
            mat.determinant(),
            Err(MatrixError::UnsupportedOperation {
                operation: "determinant"
            })
        ));
        assert!(matches!(
            mat.inverse(),
            Err(MatrixError::UnsupportedOperation {
                operation: "inverse"
            })
        ));
    }

    #[test]
    fn vector_addition_is_pairwise() {
        let lhs = Vector::from([1, 2, 3]);
        let rhs = Vector::from([4, 5, 6]);

        assert_eq!(&lhs + &rhs, Vector::from([5, 7, 9]));
    }

    #[test]
    fn gpu_runtime_initializes_or_reports_missing_device() {
        match MetalRuntime::new() {
            Ok(_) => {}
            Err(MetalError::NoMetalDevice) => {}
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        }
    }

    #[test]
    fn gpu_square_multiply_matches_cpu() {
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => runtime,
            Err(MetalError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        };
        let lhs = Matrix::<f32, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let rhs = Matrix::<f32, 2, 2>::from([[5.0, 6.0], [7.0, 8.0]]);

        assert_eq!(
            runtime.multiply(&lhs, &rhs).unwrap(),
            lhs.multiply(&rhs),
        );
    }

    #[test]
    fn gpu_rectangular_multiply_matches_cpu() {
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => runtime,
            Err(MetalError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        };
        let lhs = Matrix::<f32, 2, 3>::from([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
        let rhs = Matrix::<f32, 3, 2>::from([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);

        assert_eq!(
            runtime.multiply(&lhs, &rhs).unwrap(),
            lhs.multiply(&rhs),
        );
    }
}
