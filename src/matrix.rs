// File for matrix things

/// A matrix struct, where
/// - `const M: usize` = number of rows
/// - `const N: usize` = number of cols
#[derive(PartialEq)]
pub struct Matrix<const M: usize, const N: usize> {
    entries: Box<[[f64; N]; M]>,
}

// TODO: Use proper error handling
impl<const M: usize, const N: usize> Matrix<M, N> {
    pub fn new(entries: [[f64; N]; M]) -> Self {
        Self {
            entries: Box::new(entries),
        }
    }

    pub fn null() -> Self {
        Self {
            entries: Box::new([[0_f64; N]; M]),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Result<f64, ()> {
        if row >= M || col >= N {
            return Err(());
        }
        Ok(self.entries[col][row])
    }

    pub fn set(&mut self, row: usize, col: usize, val: f64) -> Result<(), ()> {
        if row >= M || col >= N {
            return Err(());
        }
        self.entries[col][row] = val;
        Ok(())
    }
}

// Methods exclusive to square matrices
impl<const M: usize> Matrix<M, M> {
    /// Returns the identity matrix
    pub fn identity() -> Self {
        Self {
            entries: Box::new({
                let mut unit = [[0_f64; M]; M];
                for i in 0..M {
                    unit[i][i] = 1_f64;
                }
                unit
            }),
        }
    }
}

// Implement naive matrix multiplication
impl<const M: usize, const N: usize, const K: usize> std::ops::Mul<&Matrix<N, K>>
    for &Matrix<M, N>
{
    type Output = Matrix<M, K>;

    /// Multiply (two?) matrices together
    fn mul(self, other: &Matrix<N, K>) -> Matrix<M, K> {
        let mut res_entries = [[0_f64; K]; M];
        for i in 0..M {
            for j in 0..K {
                let mut sum = 0_f64;
                for k in 0..N {
                    sum += self.entries[i][k] * other.entries[k][j]
                }
                res_entries[i][j] = sum;
            }
        }

        Self::Output {
            entries: Box::new(res_entries),
        }
    }
}

use std::fmt;
impl<const M: usize, const N: usize> fmt::Debug for Matrix<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in self.entries.into_iter() {
            for e in &row[..row.len() - 1] {
                write!(f, "{} ", e)?;
            }
            write!(f, "{}\n", row[row.len() - 1])?;
        }
        Ok(())
    }
}
