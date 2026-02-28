#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void matrix_multiply(
    constant float *A [[buffer(0)]],
    constant float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;

    if (row < M && col < K) {
        float sum = 0.0;
        for (uint i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }

        C[row * K + col] = sum;
    }
}
