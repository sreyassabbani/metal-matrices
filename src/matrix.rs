use crate::{numeric::Numeric, vector::Vector};
use std::{mem, path::PathBuf};

use thiserror::Error;

use metal::*;
use objc::rc::autoreleasepool;

/// A matrix struct, where
/// - `const M: usize` = number of rows
/// - `const N: usize` = number of cols
#[derive(PartialEq)]
pub struct Matrix<T: Numeric, const M: usize, const N: usize> {
    entries: Box<[[T; N]; M]>,
}

/// Implementation of [`From`] trait to build a [`Matrix<T, M, N>`] from `[[T; M]; N]`
impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N>
where
    T: Numeric,
{
    /// Convert from a `[[T; M]; N]` to a [`Matrix<T, M, N>`].
    /// Build a matrix from given entries
    fn from(value: [[T; N]; M]) -> Self {
        Self {
            entries: Box::new(value),
        }
    }
}

/// Build a matrix from vectors
// impl<T: Numeric, const M: usize, const N: usize> From<[Vector<T, M>; N]> for Matrix<T, M, N> {
//     fn from(value: [Vector<T, M>; N]) -> Self {
//         let v: &[T; N] = value.as_ref();
//         Self {
//             entries: Box::new(*value.as_ref()),
//         }
//     }
// }

// TODO: Use proper error handling
impl<T: Numeric, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Returns the null matrix of a given dimension
    ///
    /// ```rs
    /// let null_mat = Matrix::<i32, 3, 3>::null();
    /// let expected = Matrix::new([0, 0, 0], [0, 0, 0], [0, 0, 0]);
    ///
    /// assert_eq!(null_mat, expected);
    /// ```
    pub fn null() -> Self {
        Self {
            entries: Box::new([[T::add_idnt(); N]; M]),
        }
    }

    /// Get the entry at `row` and `col` for a given [`Matrix<T, M, N>`]
    ///
    /// ```rs
    /// let mat = Matrix::<i32, 3, 3>::from([[1, 2, 3], [42, 5, 6], [7, 8, 9]]);
    /// let found = mat.get(2, 1);
    /// let expected = 42;
    ///
    /// assert_eq!(found, expected);
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Result<T, Error> {
        if row >= M || col >= N {
            return Err(Error::OutOfBounds {
                row_found: row,
                row_max: M,
                col_found: col,
                col_max: N,
            });
        }
        Ok(self.entries[col][row])
    }

    pub fn set(&mut self, row: usize, col: usize, val: T) -> Result<(), Error> {
        if row >= M || col >= N {
            return Err(Error::OutOfBounds {
                row_found: row,
                row_max: M,
                col_found: col,
                col_max: N,
            });
        }
        self.entries[col][row] = val;
        Ok(())
    }

    /// Get the column [`Vector<T, M>`] at a given column. Note argument `col` is 0-indexed.
    pub fn get_vector(&self, col: usize) -> Vector<T, M> {
        let mut entries = [T::default(); M];
        for i in 0..M {
            println!("{:?}", self.entries[i]);
            entries[i] = self.entries[i][col];
        }
        Vector::from(entries)
    }

    /// Returns the transpose of the given matrix
    ///
    /// ```rs
    /// let m = Matrix::new(entries);
    /// let t = m.transpose();
    /// ```
    /// Note `t` and `m` will be of different types if `m` if not square.
    pub fn transpose(&self) -> Matrix<T, N, M> {
        let mut result_entries = [[T::default(); M]; N];
        for i in 0..N {
            for j in 0..M {
                result_entries[i][j] = self.entries[j][i]
            }
        }
        Matrix::from(result_entries)
    }

    pub fn determinant(&self) -> Result<T, Error> {
        todo!()
    }

    pub fn inverse(&self) -> Result<Matrix<T, N, M>, Error> {
        todo!()
    }
    // pub fn from_transformed_unit_vectors() -> Self {}
}

impl<const M: usize, const N: usize> Matrix<f32, M, N> {
    pub fn gpu_multiply<const K: usize>(&self, other: &Matrix<f32, N, K>) -> Matrix<f32, M, K> {
        let mut result_entries: Box<[[f32; K]; M]> = Box::new([[0_f32; K]; M]);
        autoreleasepool(|| {
            // Set up GPU and command queue
            let device = &Device::system_default().expect("No default device");
            let queue = device.new_command_queue();

            // Load the metal compute shader
            let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.push("shaders");
            path.push("shaders.metallib");
            let library = device.new_library_with_file(path).unwrap();
            let kernel = library.get_function("matrix_multiply", None).unwrap();

            // Set up pipeline state
            let pipeline_state = device
                .new_compute_pipeline_state_with_function(&kernel)
                .unwrap();

            // Create buffers with data for initial matrices
            let buffer_a = device.new_buffer_with_data(
                self.entries.as_ptr() as *const f32 as *const _,
                (M * N * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buffer_b = device.new_buffer_with_data(
                other.entries.as_ptr() as *const f32 as *const _,
                (N * K * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Empty buffer for resultant matrix
            let buffer_result = device.new_buffer(
                (M * K * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Create a command buffer and compute encoder
            let cmd_buf = queue.new_command_buffer();
            let compute_encoder = cmd_buf.new_compute_command_encoder();

            // Set the pipeline state and buffers
            compute_encoder.set_compute_pipeline_state(&pipeline_state);
            // compute_encoder.set_buffers(
            //     0,
            //     &[Some(&buffer_a), Some(&buffer_b), Some(&buffer_result)],
            //     &[0, 2],
            // );
            compute_encoder.set_buffer(0, Some(&buffer_a), 0);
            compute_encoder.set_buffer(1, Some(&buffer_b), 0);
            compute_encoder.set_buffer(2, Some(&buffer_result), 0);

            // Set constants for matrix dimensions
            use std::ffi::c_void;
            let m = &(M as u32) as *const u32 as *const c_void;
            let n = &(N as u32) as *const u32 as *const c_void;
            let k = &(K as u32) as *const u32 as *const c_void;
            let b_size = mem::size_of::<u32>() as u64;
            compute_encoder.set_bytes(3, b_size, m);
            compute_encoder.set_bytes(4, b_size, n);
            compute_encoder.set_bytes(5, b_size, k);

            // Set grid size
            let w = pipeline_state.thread_execution_width();
            let h = pipeline_state.max_total_threads_per_threadgroup() / w;

            // Set specifications for threads
            let threads_per_threadgroup = MTLSize::new(w as u64, h as u64, 1);
            let threads_per_grid = MTLSize::new(K as u64, M as u64, 1);
            compute_encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);

            // Finalize encoding and commit the command buffer
            compute_encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let result_ptr = buffer_result.contents() as *const f32;

            let size = M * K;
            // ??
            result_entries = unsafe {
                // Allocate uninitialized memory for the 2D array
                let mut uninit_array: Box<std::mem::MaybeUninit<[[f32; K]; M]>> = Box::new_uninit();
                let ptr = uninit_array.as_mut_ptr() as *mut f32;

                // Copy data from the raw pointer into the allocated memory
                std::ptr::copy_nonoverlapping(result_ptr, ptr, size);

                // Convert the uninitialized array to a fully initialized array
                uninit_array.assume_init()
            };
        });

        Matrix {
            entries: result_entries,
        }
    }
}

// Methods exclusive to square matrices
impl<T: Numeric, const M: usize> Matrix<T, M, M> {
    /// Returns the identity matrix
    ///
    /// # Example
    /// ```rs
    /// let identity = Matrix::<i32, 2, 2>::identity();
    /// let expected = Matrix::new([[1, 0], [0, 1]]);
    /// assert_eq!(identity, expected);
    /// ```
    pub fn identity() -> Self {
        Self {
            entries: Box::new({
                let mut unit = [[T::add_idnt(); M]; M];
                for i in 0..M {
                    unit[i][i] = T::mul_idnt();
                }
                unit
            }),
        }
    }
}

// Implement naive matrix multiplication
impl<T: Numeric, const M: usize, const N: usize, const K: usize> std::ops::Mul<&Matrix<T, N, K>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, K>;

    /// Multiply matrices together
    fn mul(self, other: &Matrix<T, N, K>) -> Self::Output {
        let mut res_entries = [[T::default(); K]; M];
        for i in 0..M {
            for j in 0..K {
                let mut sum = T::add_idnt();
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

impl<T: Numeric, const M: usize, const N: usize> std::ops::Add<&Matrix<T, M, N>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, N>;

    /// Add matrices together
    fn add(self, other: &Matrix<T, M, N>) -> Self::Output {
        let mut res_entries = [[T::default(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res_entries[i][j] = self.entries[i][j] + other.entries[i][j]
            }
        }

        Self::Output {
            entries: Box::new(res_entries),
        }
    }
}

impl<T: Numeric, const M: usize, const N: usize> std::ops::Sub<&Matrix<T, M, N>>
    for &Matrix<T, M, N>
{
    type Output = Matrix<T, M, N>;

    /// Add matrices together
    fn sub(self, other: &Matrix<T, M, N>) -> Self::Output {
        let mut res_entries = [[T::default(); N]; M];
        for i in 0..M {
            for j in 0..N {
                res_entries[i][j] = self.entries[i][j] - other.entries[i][j]
            }
        }

        Self::Output {
            entries: Box::new(res_entries),
        }
    }
}

use std::fmt;
impl<T: Numeric, const M: usize, const N: usize> fmt::Debug for Matrix<T, M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &*self.entries {
            for e in &row[..row.len() - 1] {
                write!(f, "{:?} ", e)?;
            }
            write!(f, "{:?}\n", row[row.len() - 1])?;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Accessing element by index out of bounds of matrix dimensions: get at row number {row_found} when number of rows is {row_max}, get at column number {col_found} when number of columns is {col_max}")]
    OutOfBounds {
        row_found: usize,
        row_max: usize,
        col_found: usize,
        col_max: usize,
    },
}
