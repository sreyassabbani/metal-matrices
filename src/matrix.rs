use crate::{numeric::Numeric, vector::Vector};
use std::{fmt, mem::MaybeUninit};

use thiserror::Error;

/// A matrix struct, where
/// - `const M: usize` = number of rows
/// - `const N: usize` = number of cols
#[derive(PartialEq)]
pub struct Matrix<T: Numeric, const M: usize, const N: usize> {
    entries: Box<[[T; N]; M]>,
}

/// Errors produced by matrix operations that are independent of any compute backend.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MatrixError {
    #[error("Accessing element by index out of bounds of matrix dimensions: got row {row_found} when number of rows is {row_max}, got column {col_found} when number of columns is {col_max}")]
    OutOfBounds {
        row_found: usize,
        row_max: usize,
        col_found: usize,
        col_max: usize,
    },
    #[error(
        "Accessing column index out of bounds of matrix dimensions: got column {col_found} when number of columns is {col_max}"
    )]
    ColumnOutOfBounds { col_found: usize, col_max: usize },
    #[error("Operation '{operation}' is not implemented for this matrix type")]
    UnsupportedOperation { operation: &'static str },
}

/// Implementation of [`From`] trait to build a [`Matrix<T, M, N>`] from `[[T; N]; M]`.
impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N>
where
    T: Numeric,
{
    fn from(value: [[T; N]; M]) -> Self {
        Self {
            entries: Box::new(value),
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Build a matrix from a generator without allocating large temporary arrays on the stack.
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut uninit_entries: Box<MaybeUninit<[[T; N]; M]>> = Box::new_uninit();
        let base_ptr = uninit_entries.as_mut_ptr() as *mut T;
        for row in 0..M {
            for col in 0..N {
                let idx = row * N + col;
                unsafe {
                    // SAFETY: `idx` is in-bounds for the contiguous `M * N` allocation.
                    base_ptr.add(idx).write(f(row, col));
                }
            }
        }

        Self {
            // SAFETY: every slot was initialized exactly once in the loops above.
            entries: unsafe { uninit_entries.assume_init() },
        }
    }

    pub(crate) fn from_row_major_slice(values: &[T]) -> Self {
        assert_eq!(values.len(), M * N, "row-major slice length must match matrix dimensions");
        Self::from_fn(|row, col| values[row * N + col])
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.entries.as_ptr() as *const T
    }

    /// Returns the null matrix of a given dimension.
    ///
    /// ```rust
    /// use matrix_test::matrix::Matrix;
    ///
    /// let null_mat = Matrix::<i32, 3, 3>::null();
    /// let expected = Matrix::from([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    ///
    /// assert_eq!(null_mat, expected);
    /// ```
    pub fn null() -> Self {
        Self::from_fn(|_, _| T::add_idnt())
    }

    /// Get the entry at `row` and `col` for a given [`Matrix<T, M, N>`].
    ///
    /// ```rust
    /// use matrix_test::matrix::Matrix;
    ///
    /// let mat = Matrix::<i32, 3, 3>::from([[1, 2, 3], [42, 5, 6], [7, 8, 9]]);
    /// let found = mat.get(1, 0);
    ///
    /// assert_eq!(found, Ok(42));
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Result<T, MatrixError> {
        if row >= M || col >= N {
            return Err(MatrixError::OutOfBounds {
                row_found: row,
                row_max: M,
                col_found: col,
                col_max: N,
            });
        }
        Ok(self.entries[row][col])
    }

    pub fn set(&mut self, row: usize, col: usize, val: T) -> Result<(), MatrixError> {
        if row >= M || col >= N {
            return Err(MatrixError::OutOfBounds {
                row_found: row,
                row_max: M,
                col_found: col,
                col_max: N,
            });
        }
        self.entries[row][col] = val;
        Ok(())
    }

    /// Get the column [`Vector<T, M>`] at a given column. Note argument `col` is 0-indexed.
    pub fn get_vector(&self, col: usize) -> Result<Vector<T, M>, MatrixError> {
        if col >= N {
            return Err(MatrixError::ColumnOutOfBounds {
                col_found: col,
                col_max: N,
            });
        }

        Ok(Vector::from_fn(|row| self.entries[row][col]))
    }

    /// Returns the transpose of the given matrix.
    ///
    /// ```rust
    /// use matrix_test::matrix::Matrix;
    ///
    /// let matrix = Matrix::from([[1, 2, 3], [4, 5, 6]]);
    /// let transpose = matrix.transpose();
    ///
    /// assert_eq!(transpose, Matrix::from([[1, 4], [2, 5], [3, 6]]));
    /// ```
    pub fn transpose(&self) -> Matrix<T, N, M> {
        Matrix::from_fn(|row, col| self.entries[col][row])
    }

    pub fn multiply<const K: usize>(&self, other: &Matrix<T, N, K>) -> Matrix<T, M, K> {
        self * other
    }

    pub fn determinant(&self) -> Result<T, MatrixError> {
        Err(MatrixError::UnsupportedOperation {
            operation: "determinant",
        })
    }

    pub fn inverse(&self) -> Result<Matrix<T, M, M>, MatrixError> {
        Err(MatrixError::UnsupportedOperation {
            operation: "inverse",
        })
    }
}

// Methods exclusive to square matrices
impl<T: Numeric, const M: usize> Matrix<T, M, M> {
    /// Returns the identity matrix.
    ///
    /// ```rust
    /// use matrix_test::matrix::Matrix;
    ///
    /// let identity = Matrix::<i32, 2, 2>::identity();
    /// let expected = Matrix::from([[1, 0], [0, 1]]);
    ///
    /// assert_eq!(identity, expected);
    /// ```
    pub fn identity() -> Self {
        Self::from_fn(|row, col| {
            if row == col {
                T::mul_idnt()
            } else {
                T::add_idnt()
            }
        })
    }
}

impl<T: Numeric, const M: usize, const N: usize, const K: usize> std::ops::Mul<&Matrix<T, N, K>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, K>;

    fn mul(self, other: &Matrix<T, N, K>) -> Self::Output {
        Matrix::from_fn(|row, col| {
            let mut sum = T::add_idnt();
            for idx in 0..N {
                sum += self.entries[row][idx] * other.entries[idx][col];
            }
            sum
        })
    }
}

impl<T: Numeric, const M: usize, const N: usize> std::ops::Add<&Matrix<T, M, N>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, N>;

    fn add(self, other: &Matrix<T, M, N>) -> Self::Output {
        Matrix::from_fn(|row, col| self.entries[row][col] + other.entries[row][col])
    }
}

impl<T: Numeric, const M: usize, const N: usize> std::ops::Sub<&Matrix<T, M, N>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, N>;

    fn sub(self, other: &Matrix<T, M, N>) -> Self::Output {
        Matrix::from_fn(|row, col| self.entries[row][col] - other.entries[row][col])
    }
}

impl<T: Numeric, const M: usize, const N: usize> fmt::Debug for Matrix<T, M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &*self.entries {
            if row.is_empty() {
                writeln!(f)?;
                continue;
            }
            for entry in &row[..row.len() - 1] {
                write!(f, "{entry:?} ")?;
            }
            writeln!(f, "{:?}", row[row.len() - 1])?;
        }
        Ok(())
    }
}
