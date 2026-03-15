use crate::matrix::Matrix;
use metal::{
    CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};
use objc::rc::autoreleasepool;
use std::{ffi::c_void, mem};

use thiserror::Error;

const TILE_SIZE: u64 = 16;
static SHADER_SOURCE: &str = include_str!("../shaders/matrix_multiplication.metal");

pub struct MetalRuntime {
    device: Device,
    queue: metal::CommandQueue,
    pipeline_state: ComputePipelineState,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MetalError {
    #[error("No Metal device available on this machine")]
    NoMetalDevice,
    #[error("Failed to compile Metal shader during {stage}: {message}")]
    ShaderBuildFailed {
        stage: &'static str,
        message: String,
    },
    #[error("Failed to load Metal function '{function}': {message}")]
    MetalFunctionLoad {
        function: &'static str,
        message: String,
    },
    #[error("Failed to create Metal compute pipeline: {message}")]
    MetalPipelineCreation { message: String },
    #[error("Matrix dimension '{dimension}' with value {value} does not fit in a u32")]
    DimensionOverflow {
        dimension: &'static str,
        value: usize,
    },
    #[error("Metal command buffer did not complete successfully: {status:?}")]
    CommandExecution { status: MTLCommandBufferStatus },
}

impl MetalRuntime {
    pub fn new() -> Result<Self, MetalError> {
        autoreleasepool(|| {
            let device = Device::system_default().ok_or(MetalError::NoMetalDevice)?;
            let queue = device.new_command_queue();
            let compile_options = CompileOptions::new();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &compile_options)
                .map_err(|message| MetalError::ShaderBuildFailed {
                    stage: "source compilation",
                    message,
                })?;
            let kernel = library
                .get_function("matrix_multiply", None)
                .map_err(|message| MetalError::MetalFunctionLoad {
                    function: "matrix_multiply",
                    message,
                })?;
            let pipeline_state = device
                .new_compute_pipeline_state_with_function(&kernel)
                .map_err(|message| MetalError::MetalPipelineCreation { message })?;

            Ok(Self {
                device,
                queue,
                pipeline_state,
            })
        })
    }

    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }

    pub fn shader_kind(&self) -> &'static str {
        "tiled-f32"
    }

    pub fn threadgroup_shape(&self) -> (u64, u64) {
        (TILE_SIZE, TILE_SIZE)
    }

    pub fn multiply<const M: usize, const N: usize, const K: usize>(
        &self,
        lhs: &Matrix<f32, M, N>,
        rhs: &Matrix<f32, N, K>,
    ) -> Result<Matrix<f32, M, K>, MetalError> {
        if M == 0 || N == 0 || K == 0 {
            return Ok(Matrix::null());
        }

        let m = checked_dim("M", M)?;
        let n = checked_dim("N", N)?;
        let k = checked_dim("K", K)?;

        autoreleasepool(|| {
            let buffer_a = self.device.new_buffer_with_data(
                lhs.as_ptr() as *const _,
                (M * N * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buffer_b = self.device.new_buffer_with_data(
                rhs.as_ptr() as *const _,
                (N * K * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let buffer_result = self.device.new_buffer(
                (M * K * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline_state);
            encoder.set_buffer(0, Some(&buffer_a), 0);
            encoder.set_buffer(1, Some(&buffer_b), 0);
            encoder.set_buffer(2, Some(&buffer_result), 0);

            let m_ref = &m as *const u32 as *const c_void;
            let n_ref = &n as *const u32 as *const c_void;
            let k_ref = &k as *const u32 as *const c_void;
            let bytes_len = mem::size_of::<u32>() as u64;
            encoder.set_bytes(3, bytes_len, m_ref);
            encoder.set_bytes(4, bytes_len, n_ref);
            encoder.set_bytes(5, bytes_len, k_ref);

            let threads_per_threadgroup = MTLSize::new(TILE_SIZE, TILE_SIZE, 1);
            let threadgroups_per_grid = MTLSize::new(
                ceil_div(K as u64, TILE_SIZE),
                ceil_div(M as u64, TILE_SIZE),
                1,
            );
            encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                return Err(MetalError::CommandExecution { status });
            }

            let result_ptr = buffer_result.contents() as *const f32;
            let values = unsafe {
                // SAFETY: the output buffer is `M * K` contiguous `f32` values written by the kernel.
                std::slice::from_raw_parts(result_ptr, M * K)
            };

            Ok(Matrix::from_row_major(values)
                .expect("GPU output length must match matrix dimensions"))
        })
    }
}

fn checked_dim(dimension: &'static str, value: usize) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::DimensionOverflow { dimension, value })
}

fn ceil_div(value: u64, divisor: u64) -> u64 {
    value.div_ceil(divisor)
}
