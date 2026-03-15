#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 16;

kernel void matrix_multiply(
    constant float *A [[buffer(0)]],
    constant float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float tile_a[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_b[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;
    float sum = 0.0;

    uint tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (uint tile = 0; tile < tiles; tile++) {
        uint a_col = tile * TILE_SIZE + tid.x;
        uint b_row = tile * TILE_SIZE + tid.y;

        tile_a[tid.y][tid.x] = (row < M && a_col < N) ? A[row * N + a_col] : 0.0;
        tile_b[tid.y][tid.x] = (b_row < N && col < K) ? B[b_row * K + col] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint idx = 0; idx < TILE_SIZE; idx++) {
            sum += tile_a[tid.y][idx] * tile_b[idx][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}
