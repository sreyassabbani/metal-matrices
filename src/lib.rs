pub mod matrix;
pub mod numeric;
pub mod vector;

#[cfg(test)]
mod tests {
    use crate::{
        matrix::{ComputeBackend, ComputeOptions, Matrix},
        vector::Vector,
    };
    type FSquareMatrix<const M: usize> = Matrix<f64, M, M>;
    type FMatrix<const M: usize, const N: usize> = Matrix<f64, M, N>;

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

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn mul_nonsq_and_nonsq_matrices() {
        let mat1 = FMatrix::from([[1.0, 2.0, 0.0], [2.0, 3.0, 3.0]]);
        let mat2 = FMatrix::from([[4.0, 2.0], [4.0, -1.0], [3.0, -2.0]]);
        let expected = FSquareMatrix::from([[12.0, 0.0], [29.0, -5.0]]);

        assert_eq!(&mat1 * &mat2, expected);
    }

    #[test]
    fn mul_square_and_square_matrices() {
        let mat1 = FSquareMatrix::from([[1.0, 2.0], [3.0, 4.0]]);
        let mat2 = FSquareMatrix::from([[5.0, 6.0], [7.0, 8.0]]);
        let expected = FSquareMatrix::from([[19.0, 22.0], [43.0, 50.0]]);

        assert_eq!(&mat1 * &mat2, expected);
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

        assert!(mat.get(2, 0).is_err());
        assert!(mat.get(0, 2).is_err());
    }

    #[test]
    fn get_vector_returns_column() {
        let mat = FMatrix::<3, 3>::from([[1.0, 2.0, 3.0], [42.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

        assert_eq!(mat.get_vector(0).unwrap(), Vector::from([1.0, 42.0, 7.0]));
    }

    #[test]
    fn get_vector_out_of_bounds_returns_error() {
        let mat = FMatrix::<3, 2>::from([[1.0, 2.0], [42.0, 5.0], [7.0, 8.0]]);

        assert!(mat.get_vector(2).is_err());
    }

    #[test]
    fn configured_cpu_multiply_uses_cpu_backend() {
        let mat1 = Matrix::<f32, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let mat2 = Matrix::<f32, 2, 2>::from([[5.0, 6.0], [7.0, 8.0]]);

        let output = mat1.multiply_with(&mat2, ComputeOptions::cpu()).unwrap();

        assert_eq!(output.backend, ComputeBackend::Cpu);
        assert_eq!(
            output.matrix,
            Matrix::<f32, 2, 2>::from([[19.0, 22.0], [43.0, 50.0]])
        );
    }

    #[test]
    fn default_multiply_uses_cpu_backend() {
        let mat1 = Matrix::<f32, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let mat2 = Matrix::<f32, 2, 2>::from([[5.0, 6.0], [7.0, 8.0]]);

        let output = mat1.multiply_with(&mat2, Default::default()).unwrap();

        assert_eq!(output.backend, ComputeBackend::Cpu);
        assert_eq!(
            output.matrix,
            Matrix::<f32, 2, 2>::from([[19.0, 22.0], [43.0, 50.0]])
        );
    }

    #[test]
    fn auto_with_fallback_produces_result() {
        let mat1 = Matrix::<f32, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);
        let mat2 = Matrix::<f32, 2, 2>::from([[5.0, 6.0], [7.0, 8.0]]);

        let output = mat1
            .multiply_with(&mat2, ComputeOptions::auto().with_fallback(true))
            .unwrap();

        assert_eq!(
            output.matrix,
            Matrix::<f32, 2, 2>::from([[19.0, 22.0], [43.0, 50.0]])
        );
    }

    #[test]
    fn determinant_and_inverse_are_explicitly_unimplemented() {
        let mat = Matrix::<f64, 2, 2>::from([[1.0, 2.0], [3.0, 4.0]]);

        assert!(mat.determinant().is_err());
        assert!(mat.inverse().is_err());
    }

    #[test]
    fn i8_auto_prefers_cpu_backend() {
        let lhs = Matrix::<i8, 2, 2>::from([[1, 2], [3, 4]]);
        let rhs = Matrix::<i8, 2, 2>::from([[5, 6], [7, 8]]);

        let output = lhs
            .multiply_with(&rhs, ComputeOptions::auto())
            .expect("auto mode should always produce a result");

        assert_eq!(output.backend, ComputeBackend::Cpu);
        assert_eq!(
            output.matrix,
            Matrix::<i8, 2, 2>::from([[19, 22], [43, 50]])
        );
    }

    #[test]
    fn i8_gpu_without_fallback_errors() {
        let lhs = Matrix::<i8, 2, 2>::from([[1, 2], [3, 4]]);
        let rhs = Matrix::<i8, 2, 2>::from([[5, 6], [7, 8]]);

        let err = lhs
            .multiply_with(&rhs, ComputeOptions::gpu())
            .expect_err("integer GPU path should be explicitly unsupported");
        assert!(matches!(
            err,
            crate::matrix::Error::GpuUnsupportedScalar { .. }
        ));
    }

    #[test]
    fn i8_gpu_with_fallback_uses_cpu() {
        let lhs = Matrix::<i8, 2, 2>::from([[1, 2], [3, 4]]);
        let rhs = Matrix::<i8, 2, 2>::from([[5, 6], [7, 8]]);

        let output = lhs
            .multiply_with(&rhs, ComputeOptions::gpu().with_fallback(true))
            .expect("fallback should keep integer multiplication available");

        assert_eq!(output.backend, ComputeBackend::Cpu);
        assert_eq!(
            output.matrix,
            Matrix::<i8, 2, 2>::from([[19, 22], [43, 50]])
        );
    }
}
