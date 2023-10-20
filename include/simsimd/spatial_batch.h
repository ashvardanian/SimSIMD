/**
 *  @file   x86_amx_f16.h
 *  @brief  x86 AMX implementation of the most common similarity metrics for 16-bit floating point numbers.
 *  @author Ash Vardanian
 *
 *  - Implements @b batch metrics for: L2 squared, inner product, cosine similarity.
 *  - Uses `f16` for both storage, but not accumulation.
 *  - Requires compiler capabilities: amx.
 */
#include <immintrin.h>

#include "types.h"

void mul(simsimd_f16_t const* a, simsimd_f16_t const* b, simsimd_f16_t* c, int n, int N, int a_stride, int b_stride) {
    if (n <= 32) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    c[i * N + j] += a[i * N + k] * b[k * N + j];
    } else {
        int k = n / 2;

        // c11 = a11 b11 + a12 b21
        mul(a, b, c, k, N);
        mul(a + k, b + k * N, c, k, N);

        // c12 = a11 b12 + a12 b22
        mul(a, b + k, c + k, k, N);
        mul(a + k, b + k * N + k, c + k, k, N);

        // c21 = a21 b11 + a22 b21
        mul(a + k * N, b, c + k * N, k, N);
        mul(a + k * N + k, b + k * N, c + k * N, k, N);

        // c22 = a21 b12 + a22 b22
        mul(a + k * N, b + k, c + k * N + k, k, N);
        mul(a + k * N + k, b + k * N + k, c + k * N + k, k, N);

        if (n & 1) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = (i < n - 1 && j < n - 1) ? n - 1 : 0; k < n; k++)
                        c[i * N + j] += a[i * N + k] * b[k * N + j];
        }
    }
}

/**
 *  Uses a tiled matrix multiplication to compute the inner product distances between two
 *  sets of vectors, packed into the `a` and `b` matrix as rows.
 */
void simsimd_serial_inner_f32(                                               //
    simsimd_f16_t const* a, simsimd_size_t a_count, simsimd_size_t a_stride, //
    simsimd_f16_t const* b, simsimd_size_t b_count, simsimd_size_t b_stride, //
    simsimd_size_t d, simsimd_f16_t* similarities) {

    mul(a, b, similarities, d, d, a_stride, b_stride);
}

static void simsimd_amx_f16_ip(                                              //
    simsimd_f16_t const* a, simsimd_size_t a_count, simsimd_size_t a_stride, //
    simsimd_f16_t const* b, simsimd_size_t b_count, simsimd_size_t b_stride, //
    simsimd_size_t d, simsimd_f16_t* similarities) {

    // Prepare a tile configuration
    struct {
        unsigned char palette;
        unsigned char start_row;
        unsigned char reserved1[14];
        unsigned short colsb[8];
        unsigned char reserved2[16];
        unsigned char rows[8];
        unsigned char reserved3[8];
    } tilecfg = {};

    // Assuming 16x16 tiles for this example
    tilecfg.palette = 1; // assuming palette 1 is valid for your system
    tilecfg.start_row = 0;
    tilecfg.colsb[0] = 16 * sizeof(simsimd_f16_t); // bytes per column for tile 0
    tilecfg.rows[0] = 16;                          // rows for tile 0

    // Load the tile configuration
    _tile_loadconfig(&tilecfg);

    // Iterate over the rows of matrix A
    for (simsimd_size_t i = 0; i < a_count; ++i) {
        // Iterate over the columns of matrix B
        for (simsimd_size_t j = 0; j < b_count; ++j) {
            // Set up a tile to accumulate the results
            __tile1024i acc_tile;
            _tile_zero(&acc_tile);

            // Iterate over the inner dimension (d)
            for (simsimd_size_t k = 0; k < d; k += 16) { // Assuming a tile size of 16x16
                __tile1024i a_tile, b_tile;
                // Load tiles from A and B
                _tile_loadd(&a_tile, &a[i * a_stride + k], a_stride * sizeof(simsimd_f16_t));
                _tile_loadd(&b_tile, &b[j * b_stride + k], b_stride * sizeof(simsimd_f16_t));
                // Compute the dot product and accumulate the result
                _tile_dpbf16ps(&acc_tile, a_tile, b_tile);
            }

            // Store the accumulated result back to the similarities matrix
            _tile_stored(&similarities[i * b_count + j], b_count * sizeof(simsimd_f16_t), acc_tile);
        }
    }

    // Release the tile state
    _tile_release();
}
