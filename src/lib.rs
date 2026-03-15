pub mod matrix;
pub mod numeric;
pub mod vector;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    use crate::metal::{MetalError, MetalRuntime};
    use crate::{
        matrix::{Matrix, MatrixError},
        vector::Vector,
    };
    use ndarray::Array2;

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
    fn rows_and_cols_report_const_dimensions() {
        let matrix = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
    }

    #[test]
    fn row_access_and_row_slices_work() {
        let matrix = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(matrix.row(1).unwrap(), &[4, 5, 6]);
        assert_eq!(matrix.row_slice(0).unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn row_out_of_bounds_returns_error() {
        let matrix = Matrix::<i32, 2, 2>::from([[1, 2], [3, 4]]);

        assert_eq!(
            matrix.row(2),
            Err(MatrixError::RowOutOfBounds {
                row_found: 2,
                row_max: 2,
            })
        );
    }

    #[test]
    fn row_major_access_round_trips() {
        let values = [1, 2, 3, 4, 5, 6];
        let matrix = Matrix::<i32, 2, 3>::from_row_major(&values).unwrap();

        assert_eq!(matrix.as_row_major_slice(), &values);
        assert_eq!(
            Matrix::<i32, 2, 3>::try_from(values.as_slice()).unwrap(),
            matrix
        );
    }

    #[test]
    fn row_major_length_mismatch_returns_shape_error() {
        let err = Matrix::<i32, 2, 3>::from_row_major(&[1, 2, 3]).unwrap_err();

        assert_eq!(
            err,
            MatrixError::ShapeMismatch {
                expected_len: 6,
                actual_len: 3,
            }
        );
    }

    #[test]
    fn ndarray_round_trip_preserves_square_matrix() {
        let matrix = Matrix::<f32, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let array: ndarray::Array2<f32> = (&matrix).into();

        assert_eq!(Matrix::<f32, 2, 2>::try_from(array).unwrap(), matrix);
    }

    #[test]
    fn ndarray_round_trip_preserves_rectangular_matrix() {
        let matrix = Matrix::<f32, 2, 3>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let array: ndarray::Array2<f32> = (&matrix).into();

        assert_eq!(Matrix::<f32, 2, 3>::try_from(array).unwrap(), matrix);
    }

    #[test]
    fn ndarray_shape_mismatch_returns_error() {
        let array = Array2::from_shape_vec((3, 1), vec![1.0_f32, 2.0, 3.0]).unwrap();

        assert_eq!(
            Matrix::<f32, 2, 2>::try_from(array),
            Err(MatrixError::ShapeMismatch {
                expected_len: 4,
                actual_len: 3,
            })
        );
    }

    #[test]
    fn ndarray_dot_matches_matrix_multiply() {
        let lhs = Matrix::<f32, 2, 3>::from([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
        let rhs = Matrix::<f32, 3, 2>::from([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);

        let lhs_array: Array2<f32> = (&lhs).into();
        let rhs_array: Array2<f32> = (&rhs).into();
        let product = lhs.multiply(&rhs);
        let dot = lhs_array.dot(&rhs_array);

        assert_eq!(product, Matrix::<f32, 2, 2>::try_from(dot).unwrap());
    }

    #[test]
    fn randomized_ndarray_dot_matches_matrix_multiply() {
        fn value(seed: u64, row: usize, col: usize) -> f32 {
            let mixed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add((row as u64 + 1) * 1_048_583)
                .wrapping_add((col as u64 + 1) * 97_531);
            ((mixed % 20_000) as f32 / 1000.0) - 10.0
        }

        let lhs = Matrix::<f32, 4, 5>::from_fn(|row, col| value(7, row, col));
        let rhs = Matrix::<f32, 5, 3>::from_fn(|row, col| value(19, row, col));

        let lhs_array: Array2<f32> = (&lhs).into();
        let rhs_array: Array2<f32> = (&rhs).into();
        let dot = lhs_array.dot(&rhs_array);

        assert_matrix_close(
            &lhs.multiply(&rhs),
            &Matrix::<f32, 4, 3>::try_from(dot).unwrap(),
            1e-4,
            1e-5,
        );
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

        assert_eq!(
            &mat1 - &mat2,
            FMatrix::<2, 2>::from([[4.0, 2.5], [2.5, 0.0]])
        );
    }

    #[test]
    fn transpose_returns_expected_matrix() {
        let mat = Matrix::<i32, 2, 3>::from([[1, 2, 3], [4, 5, 6]]);

        assert_eq!(
            mat.transpose(),
            Matrix::<i32, 3, 2>::from([[1, 4], [2, 5], [3, 6]])
        );
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

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_runtime_initializes_or_reports_missing_device() {
        match MetalRuntime::new() {
            Ok(_) => {}
            Err(MetalError::NoMetalDevice) => {}
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_square_multiply_matches_cpu() {
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => runtime,
            Err(MetalError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        };
        let lhs = Matrix::<f32, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let rhs = Matrix::<f32, 2, 2>::from([[5.0, 6.0], [7.0, 8.0]]);

        assert_matrix_close(
            &runtime.multiply(&lhs, &rhs).unwrap(),
            &lhs.multiply(&rhs),
            1e-4,
            1e-5,
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_rectangular_multiply_matches_cpu() {
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => runtime,
            Err(MetalError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        };
        let lhs = Matrix::<f32, 2, 3>::from([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
        let rhs = Matrix::<f32, 3, 2>::from([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);

        assert_matrix_close(
            &runtime.multiply(&lhs, &rhs).unwrap(),
            &lhs.multiply(&rhs),
            1e-4,
            1e-5,
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_tiled_kernel_handles_non_multiple_dimensions() {
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => runtime,
            Err(MetalError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        };
        let lhs = Matrix::<f32, 17, 19>::from_fn(|row, col| ((row * 19 + col) as f32 * 0.1).sin());
        let rhs = Matrix::<f32, 19, 13>::from_fn(|row, col| ((row * 13 + col) as f32 * 0.07).cos());

        assert_matrix_close(
            &runtime.multiply(&lhs, &rhs).unwrap(),
            &lhs.multiply(&rhs),
            1e-4,
            1e-5,
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_runtime_reuses_pipeline_across_repeated_multiplies() {
        let runtime = match MetalRuntime::new() {
            Ok(runtime) => runtime,
            Err(MetalError::NoMetalDevice) => return,
            Err(err) => panic!("unexpected Metal runtime init failure: {err}"),
        };
        let lhs =
            Matrix::<f32, 64, 64>::from_fn(|row, col| ((row * 64 + col) as f32 * 0.0013).sin());
        let rhs =
            Matrix::<f32, 64, 64>::from_fn(|row, col| ((row * 64 + col) as f32 * 0.0017).cos());
        let expected = lhs.multiply(&rhs);

        for _ in 0..3 {
            let actual = runtime.multiply(&lhs, &rhs).unwrap();
            assert_matrix_close(&actual, &expected, 1e-3, 1e-5);
        }
    }

    fn assert_matrix_close<const M: usize, const N: usize>(
        actual: &Matrix<f32, M, N>,
        expected: &Matrix<f32, M, N>,
        abs_tol: f32,
        rel_tol: f32,
    ) {
        for (idx, (actual, expected)) in actual
            .as_row_major_slice()
            .iter()
            .zip(expected.as_row_major_slice().iter())
            .enumerate()
        {
            let diff = (actual - expected).abs();
            let tolerance = abs_tol.max(expected.abs() * rel_tol);
            assert!(
                diff <= tolerance,
                "matrix mismatch at flat index {idx}: actual={actual}, expected={expected}, diff={diff}, tolerance={tolerance}"
            );
        }
    }
}
