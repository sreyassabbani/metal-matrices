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

// Build a matrix from vectors
// impl<T: Numeric, const M: usize, const N: usize> From<[Vector<T, M>; N]> for Matrix<T, M, N> {
//     fn from(value: [Vector<T, M>; N]) -> Self {
//         let v: &[T; N] = value.as_ref();
//         Self {
//             entries: Box::new(*value.as_ref()),
//         }
//     }
// }

impl<T: Numeric, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Build a matrix from a generator without allocating large temporary arrays on the stack.
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut uninit_entries: Box<std::mem::MaybeUninit<[[T; N]; M]>> = Box::new_uninit();
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

    /// Returns the null matrix of a given dimension
    ///
    /// ```rs
    /// let null_mat = Matrix::<i32, 3, 3>::null();
    /// let expected = Matrix::from([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
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
    /// let found = mat.get(1, 0);
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
        Ok(self.entries[row][col])
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
        self.entries[row][col] = val;
        Ok(())
    }

    /// Get the column [`Vector<T, M>`] at a given column. Note argument `col` is 0-indexed.
    pub fn get_vector(&self, col: usize) -> Result<Vector<T, M>, Error> {
        if col >= N {
            return Err(Error::ColumnOutOfBounds {
                col_found: col,
                col_max: N,
            });
        }
        let mut entries = [T::default(); M];
        for (i, entry) in entries.iter_mut().enumerate() {
            *entry = self.entries[i][col];
        }
        Ok(Vector::from(entries))
    }

    /// Returns the transpose of the given matrix
    ///
    /// ```rs
    /// let m = Matrix::from(entries);
    /// let t = m.transpose();
    /// ```
    /// Note `t` and `m` will be of different types if `m` if not square.
    pub fn transpose(&self) -> Matrix<T, N, M> {
        let mut result_entries = [[T::default(); M]; N];
        for (i, row) in result_entries.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry = self.entries[j][i]
            }
        }
        Matrix::from(result_entries)
    }

    pub fn determinant(&self) -> Result<T, Error> {
        Err(Error::UnsupportedOperation {
            operation: "determinant",
        })
    }

    pub fn inverse(&self) -> Result<Matrix<T, N, M>, Error> {
        Err(Error::UnsupportedOperation {
            operation: "inverse",
        })
    }
    // pub fn from_transformed_unit_vectors() -> Self {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendPreference {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeOptions {
    pub preference: ComputeBackendPreference,
    pub allow_fallback: bool,
}

impl ComputeOptions {
    pub fn auto() -> Self {
        Self {
            preference: ComputeBackendPreference::Auto,
            allow_fallback: true,
        }
    }

    pub fn cpu() -> Self {
        Self {
            preference: ComputeBackendPreference::Cpu,
            allow_fallback: false,
        }
    }

    pub fn gpu() -> Self {
        Self {
            preference: ComputeBackendPreference::Gpu,
            allow_fallback: false,
        }
    }

    pub fn with_fallback(mut self, allow_fallback: bool) -> Self {
        self.allow_fallback = allow_fallback;
        self
    }
}

impl Default for ComputeOptions {
    fn default() -> Self {
        Self {
            preference: ComputeBackendPreference::Cpu,
            allow_fallback: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub cpu: bool,
    pub gpu: bool,
}

#[derive(Debug, PartialEq)]
pub struct ComputeOutput<T: Numeric, const M: usize, const K: usize> {
    pub matrix: Matrix<T, M, K>,
    pub backend: ComputeBackend,
}

#[doc(hidden)]
pub trait ComputeScalar: Numeric + 'static {
    const GPU_CAPABLE: bool;

    fn gpu_multiply<const M: usize, const N: usize, const K: usize>(
        lhs: &Matrix<Self, M, N>,
        rhs: &Matrix<Self, N, K>,
    ) -> Result<Matrix<Self, M, K>, Error>
    where
        Self: Sized;
}

macro_rules! impl_cpu_only_scalar {
    ($($t:ty),* $(,)?) => {
        $(
            impl ComputeScalar for $t {
                const GPU_CAPABLE: bool = false;

                fn gpu_multiply<const M: usize, const N: usize, const K: usize>(
                    _lhs: &Matrix<Self, M, N>,
                    _rhs: &Matrix<Self, N, K>,
                ) -> Result<Matrix<Self, M, K>, Error> {
                    Err(Error::GpuUnsupportedScalar {
                        scalar: std::any::type_name::<Self>(),
                    })
                }
            }
        )*
    };
}

impl_cpu_only_scalar!(f64, i64, i32, i16, i8, u64, u32, u16, u8);

impl ComputeScalar for f32 {
    const GPU_CAPABLE: bool = true;

    fn gpu_multiply<const M: usize, const N: usize, const K: usize>(
        lhs: &Matrix<Self, M, N>,
        rhs: &Matrix<Self, N, K>,
    ) -> Result<Matrix<Self, M, K>, Error> {
        gpu_multiply_f32(lhs, rhs)
    }
}

impl<T: ComputeScalar, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn backend_capabilities() -> BackendCapabilities {
        BackendCapabilities {
            cpu: true,
            gpu: T::GPU_CAPABLE,
        }
    }

    pub fn multiply_cpu<const K: usize>(&self, other: &Matrix<T, N, K>) -> Matrix<T, M, K> {
        self * other
    }

    pub fn multiply<const K: usize>(
        &self,
        other: &Matrix<T, N, K>,
    ) -> Result<Matrix<T, M, K>, Error> {
        self.multiply_with(other, ComputeOptions::default())
            .map(|output| output.matrix)
    }

    pub fn multiply_with<const K: usize>(
        &self,
        other: &Matrix<T, N, K>,
        options: ComputeOptions,
    ) -> Result<ComputeOutput<T, M, K>, Error> {
        match options.preference {
            ComputeBackendPreference::Cpu => Ok(ComputeOutput {
                matrix: self.multiply_cpu(other),
                backend: ComputeBackend::Cpu,
            }),
            ComputeBackendPreference::Gpu => match T::gpu_multiply(self, other) {
                Ok(matrix) => Ok(ComputeOutput {
                    matrix,
                    backend: ComputeBackend::Gpu,
                }),
                Err(_err) if options.allow_fallback => Ok(ComputeOutput {
                    matrix: self.multiply_cpu(other),
                    backend: ComputeBackend::Cpu,
                }),
                Err(err) => Err(err),
            },
            ComputeBackendPreference::Auto => {
                if !T::GPU_CAPABLE {
                    return Ok(ComputeOutput {
                        matrix: self.multiply_cpu(other),
                        backend: ComputeBackend::Cpu,
                    });
                }

                match T::gpu_multiply(self, other) {
                    Ok(matrix) => Ok(ComputeOutput {
                        matrix,
                        backend: ComputeBackend::Gpu,
                    }),
                    Err(_err) if options.allow_fallback => Ok(ComputeOutput {
                        matrix: self.multiply_cpu(other),
                        backend: ComputeBackend::Cpu,
                    }),
                    Err(err) => Err(err),
                }
            }
        }
    }
}

fn gpu_multiply_f32<const M: usize, const N: usize, const K: usize>(
    lhs: &Matrix<f32, M, N>,
    rhs: &Matrix<f32, N, K>,
) -> Result<Matrix<f32, M, K>, Error> {
    let result_entries = autoreleasepool(|| -> Result<Box<[[f32; K]; M]>, Error> {
        // Set up GPU and command queue
        let device = Device::system_default().ok_or(Error::NoMetalDevice)?;
        let queue = device.new_command_queue();

        // Load the metal compute shader
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("shaders");
        path.push("shaders.metallib");
        let library =
            device
                .new_library_with_file(&path)
                .map_err(|source| Error::MetalLibraryLoad {
                    path: path.display().to_string(),
                    message: source.to_string(),
                })?;
        let kernel = library
            .get_function("matrix_multiply", None)
            .map_err(|source| Error::MetalFunctionLoad {
                function: "matrix_multiply".to_string(),
                message: source.to_string(),
            })?;

        // Set up pipeline state
        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|source| Error::MetalPipelineCreation {
                message: source.to_string(),
            })?;

        // Create buffers with data for initial matrices
        let buffer_a = device.new_buffer_with_data(
            lhs.entries.as_ptr() as *const f32 as *const _,
            (M * N * mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let buffer_b = device.new_buffer_with_data(
            rhs.entries.as_ptr() as *const f32 as *const _,
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
        let threads_per_threadgroup = MTLSize::new(w, h, 1);
        let threads_per_grid = MTLSize::new(K as u64, M as u64, 1);
        compute_encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);

        // Finalize encoding and commit the command buffer
        compute_encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let result_ptr = buffer_result.contents() as *const f32;

        let size = M * K;
        // ??
        let result_entries = unsafe {
            // Allocate uninitialized memory for the 2D array
            let mut uninit_array: Box<std::mem::MaybeUninit<[[f32; K]; M]>> = Box::new_uninit();
            let ptr = uninit_array.as_mut_ptr() as *mut f32;

            // Copy data from the raw pointer into the allocated memory
            std::ptr::copy_nonoverlapping(result_ptr, ptr, size);

            // Convert the uninitialized array to a fully initialized array
            uninit_array.assume_init()
        };

        Ok(result_entries)
    })?;

    Ok(Matrix {
        entries: result_entries,
    })
}

// Methods exclusive to square matrices
impl<T: Numeric, const M: usize> Matrix<T, M, M> {
    /// Returns the identity matrix
    ///
    /// # Example
    /// ```rs
    /// let identity = Matrix::<i32, 2, 2>::identity();
    /// let expected = Matrix::from([[1, 0], [0, 1]]);
    /// assert_eq!(identity, expected);
    /// ```
    pub fn identity() -> Self {
        Self {
            entries: Box::new({
                let mut unit = [[T::add_idnt(); M]; M];
                for (i, row) in unit.iter_mut().enumerate() {
                    row[i] = T::mul_idnt();
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
        for (i, row) in res_entries.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                let mut sum = T::add_idnt();
                for k in 0..N {
                    sum += self.entries[i][k] * other.entries[k][j]
                }
                *entry = sum;
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
        for (i, row) in res_entries.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry = self.entries[i][j] + other.entries[i][j]
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
        for (i, row) in res_entries.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry = self.entries[i][j] - other.entries[i][j]
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
            if row.is_empty() {
                writeln!(f)?;
                continue;
            }
            for e in &row[..row.len() - 1] {
                write!(f, "{:?} ", e)?;
            }
            writeln!(f, "{:?}", row[row.len() - 1])?;
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
    #[error(
        "Accessing column index out of bounds of matrix dimensions: got column {col_found} when number of columns is {col_max}"
    )]
    ColumnOutOfBounds { col_found: usize, col_max: usize },
    #[error("No Metal device available on this machine")]
    NoMetalDevice,
    #[error("Failed to load Metal library from '{path}': {message}")]
    MetalLibraryLoad { path: String, message: String },
    #[error("Failed to load Metal function '{function}': {message}")]
    MetalFunctionLoad { function: String, message: String },
    #[error("Failed to create Metal compute pipeline: {message}")]
    MetalPipelineCreation { message: String },
    #[error("GPU backend is not supported for scalar type '{scalar}'")]
    GpuUnsupportedScalar { scalar: &'static str },
    #[error("Operation '{operation}' is not implemented for this matrix type")]
    UnsupportedOperation { operation: &'static str },
}
