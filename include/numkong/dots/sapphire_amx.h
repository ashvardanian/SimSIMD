/**
 *  @brief SIMD-accelerated Dot Products for Real and Complex Numbers optimized for Intel Sapphire Rapids CPUs.
 *  @file include/numkong/dots/sapphire.h
 *  @sa include/numkong/dots.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  AMX tile dimensions:
 *
 *    BF16 tiles: 16 rows × 32 elements = 512 BF16 values = 1KB per tile
 *    INT8 tiles: 16 rows × 64 elements = 1024 INT8 values = 1KB per tile
 *
 *  Output pattern: 2×2 tile layout produces 32×32 output blocks.
 *  Register allocation:
 *    TMM0, TMM1: A matrix tiles (row blocks i and i+16)
 *    TMM2, TMM3: B matrix tiles (column blocks j and j+16)
 *    TMM4-7: C accumulator tiles (2×2 output grid)
 *
 *  Performance characteristics (single-threaded):
 *    - BF16 peak: ~500 GFLOPS per core (2× FP32 throughput)
 *    - INT8 peak: ~1000 GOPS per core (4× FP32 throughput)
 *    - Memory bandwidth: ~80 GB/s DDR5 per core
 *    - Optimal K dimension: multiples of 32 (BF16) or 64 (INT8)
 *
 *  Acceleration opportunities:
 *    - Pre-pack B matrix once for repeated inference (avoids runtime reordering)
 *    - Morton Z-curve tile ordering improves L2 cache hit rate by 5-25%
 *    - Partition A rows across threads for parallel execution
 *    - Use streaming stores for large C matrices to avoid cache pollution
 */
#ifndef NK_DOTS_SAPPHIRE_AMX_H
#define NK_DOTS_SAPPHIRE_AMX_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE_AMX
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "bmi2", "avx512bw", "avx512fp16")
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,bmi2,avx512bw,avx512fp16"))), \
                             apply_to = function)

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  AMX-specific packed buffer header (64-byte aligned).
 *  Different from nk_dots_amx_packed_header_t as AMX uses tile-based layout.
 */
typedef struct {
    nk_u32_t full_n_tiles;  // Number of full N tiles (16 rows each)
    nk_u32_t full_k_tiles;  // Number of K tiles (32 cols for BF16, 64 for I8)
    nk_u32_t n_edge_rows;   // Remaining rows after full tiles (0-15)
    nk_u32_t n_edge_offset; // Byte offset to edge data region
    nk_u32_t reserved[12];  // Padding to 64 bytes
} nk_dots_amx_packed_header_t;

/*  Morton Z-curve encoding for cache-friendly tile traversal.
 *  Uses BMI2 PDEP instruction for fast (2-3 cycle) bit interleaving.
 *  Interleaves bits of (tile_row, tile_col) to produce Z-curve index.
 */
NK_INTERNAL nk_u64_t nk_morton_encode_sapphire_amx_(nk_u32_t tile_row, nk_u32_t tile_col) {
    return _pdep_u64(tile_row, 0x5555555555555555ULL) | _pdep_u64(tile_col, 0xAAAAAAAAAAAAAAAAULL);
}

/*  Configure AMX tile registers.
 *  Called once per kernel invocation (idempotent within a thread).
 *  Sets all 8 tiles to standard 16 rows × 64 bytes layout.
 *
 *  Note: OS permission for AMX must be requested before using AMX instructions.
 *  Call `nk_configure_thread(nk_capabilities())` once per thread
 *  before using any Sapphire matmul functions.
 */
NK_INTERNAL void nk_amx_tile_configure_sapphire_amx_(void) {
    NK_ALIGN64 nk_u8_t tile_config[64] = {0};
    tile_config[0] = 1; // palette 1 (standard tile configuration)

    nk_u16_t *bytes_per_row = (nk_u16_t *)&tile_config[16];
    nk_u8_t *rows_per_tile = &tile_config[48];

    for (int tile_id = 0; tile_id < 8; tile_id++) {
        rows_per_tile[tile_id] = 16; // 16 rows per tile
        bytes_per_row[tile_id] = 64; // 64 bytes per row (1KB total)
    }
    _tile_loadconfig(tile_config);
}

/*  Compiler memory barrier to ensure stores complete before AMX tile loads.
 *  AMX _tile_loadd reads from memory written by AVX-512 stores. Without this barrier,
 *  the compiler may reorder or optimize away the stores, causing _tile_loadd to read stale data.
 *  This is a compiler-only fence (no CPU fence needed - same core, same memory).
 */
NK_INTERNAL void nk_compiler_barrier_sapphire_amx_(void) { __asm__ volatile("" ::: "memory"); }

/*  AVX-512 masked load for A tile (BF16): loads up to 16 rows × 32 cols into aligned buffer.
 *  Uses masked loads to handle edge tiles without element-wise loops.
 *  Includes memory barrier to ensure stores complete before subsequent _tile_loadd.
 */
NK_INTERNAL void nk_load_a_tile_bf16_masked_(            //
    nk_bf16_t const *src, nk_size_t src_stride_elements, //
    nk_size_t valid_rows, nk_size_t valid_cols, nk_bf16_t *dst /*[16][32]*/) {

    __mmask32 col_mask = (valid_cols >= 32) ? 0xFFFFFFFF : ((__mmask32)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t r = 0; r < 16; r++) {
        if (r < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi16(col_mask, src + r * src_stride_elements);
            _mm512_store_si512((__m512i *)(dst + r * 32), row);
        }
        else { _mm512_store_si512((__m512i *)(dst + r * 32), zero); }
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  AVX-512 masked load for A tile (I8): loads up to 16 rows × 64 cols into aligned buffer.
 *  Includes memory barrier to ensure stores complete before subsequent _tile_loadd.
 */
NK_INTERNAL void nk_load_a_tile_i8_masked_(   //
    nk_i8_t const *src, nk_size_t src_stride, //
    nk_size_t valid_rows, nk_size_t valid_cols, nk_i8_t *dst /*[16][64]*/) {

    __mmask64 col_mask = (valid_cols >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((__mmask64)1 << valid_cols) - 1;
    __m512i zero = _mm512_setzero_si512();

    for (nk_size_t r = 0; r < 16; r++) {
        if (r < valid_rows) {
            __m512i row = _mm512_maskz_loadu_epi8(col_mask, src + r * src_stride);
            _mm512_store_si512((__m512i *)(dst + r * 64), row);
        }
        else { _mm512_store_si512((__m512i *)(dst + r * 64), zero); }
    }
    nk_compiler_barrier_sapphire_amx_();
}

/*  AVX-512 masked store for C tile (F32): stores up to 16 rows × 16 cols from aligned buffer.
 */
NK_INTERNAL void nk_store_c_tile_f32_masked_(                              //
    nk_f32_t const *src /*[16][16]*/, nk_f32_t *dst, nk_size_t dst_stride, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 col_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t r = 0; r < valid_rows; r++) {
        __m512 row = _mm512_load_ps(src + r * 16);
        _mm512_mask_storeu_ps(dst + r * dst_stride, col_mask, row);
    }
}

/*  AVX-512 masked store for C tile (I32): stores up to 16 rows × 16 cols from aligned buffer.
 */
NK_INTERNAL void nk_store_c_tile_i32_masked_(                              //
    nk_i32_t const *src /*[16][16]*/, nk_i32_t *dst, nk_size_t dst_stride, //
    nk_size_t valid_rows, nk_size_t valid_cols) {

    __mmask16 col_mask = (valid_cols >= 16) ? 0xFFFF : ((__mmask16)1 << valid_cols) - 1;

    for (nk_size_t r = 0; r < valid_rows; r++) {
        __m512i row = _mm512_load_si512((__m512i const *)(src + r * 16));
        _mm512_mask_storeu_epi32(dst + r * dst_stride, col_mask, row);
    }
}

/*  AVX-512 edge matmul for BF16 → F32.
 *  Computes C[m_start:m_end, n_start:n_end] using row-major B edge data.
 *  Used for boundary regions where AMX is overkill.
 *
 *  B is stored in row-major as b_edge[row * k + col] where row is the N index.
 *  This computes: C[i,j] = sum_over_k(A[i,k] * B[j,k])
 */
NK_INTERNAL void nk_dots_bf16bf16f32_avx512_edge_(            //
    nk_bf16_t const *a, nk_bf16_t const *b_edge, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                    //
    nk_size_t a_stride_elements, nk_size_t b_stride_k, nk_size_t c_stride_elements) {

    // Process each output row
    for (nk_size_t i = 0; i < m; i++) {
        nk_bf16_t const *a_row = a + i * a_stride_elements;

        // Process output columns in chunks of 16 (AVX-512 width for F32)
        for (nk_size_t j = 0; j < n; j += 16) {
            nk_size_t const j_count = (j + 16 <= n) ? 16 : n - j;
            __m512 acc = _mm512_setzero_ps();

            // Dot product over K dimension - process 2 at a time for BF16 pairs
            for (nk_size_t kk = 0; kk < k; kk++) {
                // Broadcast A[i,k] to F32
                nk_f32_t a_val = (nk_f32_t)a_row[kk];
                __m512 a_bc = _mm512_set1_ps(a_val);

                // Gather B[j:j+16, k] - each B row is stored with stride b_stride_k
                // b_edge[row * b_stride_k + col] where row in [0,n), col in [0,k)
                NK_ALIGN64 nk_f32_t b_vals[16];
                for (nk_size_t jj = 0; jj < j_count; jj++) {
                    b_vals[jj] = (nk_f32_t)b_edge[(j + jj) * b_stride_k + kk];
                }
                for (nk_size_t jj = j_count; jj < 16; jj++) b_vals[jj] = 0.0f;

                __m512 b_vec = _mm512_load_ps(b_vals);
                acc = _mm512_fmadd_ps(a_bc, b_vec, acc);
            }

            // Store with mask
            __mmask16 mask = (j_count >= 16) ? 0xFFFF : ((__mmask16)1 << j_count) - 1;
            _mm512_mask_storeu_ps(c + i * c_stride_elements + j, mask, acc);
        }
    }
}

/*  AVX-512 edge matmul for I8 → I32.
 *  Computes C[m_start:m_end, n_start:n_end] using row-major B edge data.
 *  Used for boundary regions where AMX is overkill.
 */
NK_INTERNAL void nk_dots_i8i8i32_avx512_edge_(            //
    nk_i8_t const *a, nk_i8_t const *b_edge, nk_i32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                //
    nk_size_t a_stride, nk_size_t b_stride_k, nk_size_t c_stride_elements) {

    // Process each output row
    for (nk_size_t i = 0; i < m; i++) {
        nk_i8_t const *a_row = a + i * a_stride;

        // Process output columns in chunks of 16 (AVX-512 width for I32)
        for (nk_size_t j = 0; j < n; j += 16) {
            nk_size_t const j_count = (j + 16 <= n) ? 16 : n - j;
            __m512i acc = _mm512_setzero_si512();

            // Dot product over K dimension
            for (nk_size_t kk = 0; kk < k; kk++) {
                // Broadcast A[i,k] to I32
                nk_i32_t a_val = (nk_i32_t)a_row[kk];
                __m512i a_bc = _mm512_set1_epi32(a_val);

                // Gather B[j:j+16, k]
                NK_ALIGN64 nk_i32_t b_vals[16];
                for (nk_size_t jj = 0; jj < j_count; jj++) {
                    b_vals[jj] = (nk_i32_t)b_edge[(j + jj) * b_stride_k + kk];
                }
                for (nk_size_t jj = j_count; jj < 16; jj++) b_vals[jj] = 0;

                __m512i b_vec = _mm512_load_si512((__m512i const *)b_vals);
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(a_bc, b_vec));
            }

            // Store with mask
            __mmask16 mask = (j_count >= 16) ? 0xFFFF : ((__mmask16)1 << j_count) - 1;
            _mm512_mask_storeu_epi32(c + i * c_stride_elements + j, mask, acc);
        }
    }
}

/*  BF16 packed buffer size: header + all tiles for full N rows + N edge.
 *  Hybrid layout:
 *    - Tiles include K remainder (zero-padded) for AMX to handle full dot products
 *    - N edge (remaining rows) stored row-major for simple AVX-512 edge kernel
 */
NK_PUBLIC nk_size_t nk_dots_bf16bf16f32_packed_size_sapphire_amx(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_rows = 16;
    nk_size_t const tile_cols = 32;
    nk_size_t const tile_bytes = 512 * sizeof(nk_bf16_t); // 16×32×2 = 1KB

    nk_size_t const full_n_tiles = n / tile_rows;
    nk_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Ceiling division
    nk_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full N rows (Morton-ordered, pair-interleaved, K remainder zero-padded)
    size += full_n_tiles * tiles_along_k * tile_bytes;

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_bf16_t);

    return size;
}

/*  I8 packed buffer size: header + all tiles for full N rows + N edge.
 */
NK_PUBLIC nk_size_t nk_dots_i8i8i32_packed_size_sapphire_amx(nk_size_t n, nk_size_t k) {
    nk_size_t const tile_rows = 16;
    nk_size_t const tile_cols = 64;
    nk_size_t const tile_bytes = 1024 * sizeof(nk_i8_t); // 16×64×1 = 1KB

    nk_size_t const full_n_tiles = n / tile_rows;
    nk_size_t const tiles_along_k = (k + tile_cols - 1) / tile_cols; // Ceiling division
    nk_size_t const n_edge_rows = n - full_n_tiles * tile_rows;

    // Header (64 bytes aligned)
    nk_size_t size = sizeof(nk_dots_amx_packed_header_t);

    // All tiles for full N rows (Morton-ordered, quad-interleaved, K remainder zero-padded)
    size += full_n_tiles * tiles_along_k * tile_bytes;

    // N edge: remaining rows for ALL K columns, stored row-major
    if (n_edge_rows > 0) size += n_edge_rows * k * sizeof(nk_i8_t);

    return size;
}

/*  Pack BF16 B matrix with hybrid layout:
 *    - Header with layout metadata
 *    - All tiles for full N rows: Morton Z-curve ordered, pair-interleaved (for AMX)
 *      Including K remainder tiles (zero-padded) so AMX can compute full dot products
 *    - N edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX BF16 tile format: for TDPBF16PS, B tile should have elements arranged so that
 *  consecutive pairs of columns are interleaved by rows:
 *    [col0_row0, col1_row0, col0_row1, col1_row1, ..., col0_row15, col1_row15,
 *     col2_row0, col3_row0, col2_row1, col3_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 2) * 32 + row * 2 + (col % 2)
 */
NK_PUBLIC void nk_dots_bf16bf16f32_pack_sapphire_amx( //
    nk_bf16_t const *b, nk_size_t n, nk_size_t k,     //
    nk_size_t b_stride, void *b_packed) {

    // AMX BF16 tile dimensions: 16 rows × 32 columns (512 BF16 elements = 1KB)
    nk_size_t const tile_rows = 16;
    nk_size_t const tile_cols = 32;
    nk_size_t const tile_elements = 512;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_bf16_t);
    nk_size_t const b_stride_elements = b_stride / sizeof(nk_bf16_t);

    // Compute layout dimensions
    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_cols - 1) / tile_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    // Pointers to packed data regions
    nk_bf16_t *tiles_ptr = (nk_bf16_t *)((char *)b_packed + tiles_offset);
    nk_bf16_t *n_edge_ptr = (nk_bf16_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize all tiles (handles K remainder padding)
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles using LINEAR ordering: tile_index = n_tile * num_k_tiles + k_tile
    // This provides sequential memory access when streaming along K dimension,
    // which is critical for cache efficiency in the compute kernel.
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {

            // Linear tile index: all K-tiles for one N-tile are contiguous
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_bf16_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_cols <= k) ? tile_cols : (k - src_col_start);

            // Pack with pair-interleaving as required by TDPBF16PS instruction.
            // AMX expects: [col0_row0, col1_row0, col0_row1, col1_row1, col2_row0, col3_row0, ...]
            // Formula: packed_idx = (col / 2) * 32 + row * 2 + (col % 2)
            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride_elements + src_col_start + col;
                    nk_size_t const dst_idx = (col / 2) * 32 + row * 2 + (col % 2);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder rows in simple row-major format (for AVX-512 fallback)
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride_elements + col];
            }
        }
    }
}

/*  Pack I8 B matrix with hybrid layout:
 *    - Header with layout metadata
 *    - All tiles for full N rows: linearly ordered, quad-interleaved (for AMX)
 *      Including K remainder tiles (zero-padded) so AMX can compute full dot products
 *    - N edge rows: row-major (for AVX-512 edge kernel)
 *
 *  AMX INT8 tile format: for TDPBSSD, B tile should have 4 consecutive columns
 *  interleaved by rows:
 *    [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, col1_row1, ...]
 *
 *  Interleaving formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
 */
NK_PUBLIC void nk_dots_i8i8i32_pack_sapphire_amx( //
    nk_i8_t const *b, nk_size_t n, nk_size_t k,   //
    nk_size_t b_stride, void *b_packed) {

    // AMX I8 tile dimensions: 16 rows × 64 columns (1024 I8 elements = 1KB)
    nk_size_t const tile_rows = 16;
    nk_size_t const tile_cols = 64;
    nk_size_t const tile_elements = 1024;
    nk_size_t const tile_bytes = tile_elements * sizeof(nk_i8_t);

    // Compute layout dimensions
    nk_size_t const num_n_tiles = n / tile_rows;
    nk_size_t const num_k_tiles = (k + tile_cols - 1) / tile_cols;
    nk_size_t const n_remainder_rows = n - num_n_tiles * tile_rows;
    nk_size_t const total_tiles = num_n_tiles * num_k_tiles;

    // Write header with layout metadata
    nk_dots_amx_packed_header_t *header = (nk_dots_amx_packed_header_t *)b_packed;
    header->full_n_tiles = (nk_u32_t)num_n_tiles;
    header->full_k_tiles = (nk_u32_t)num_k_tiles;
    header->n_edge_rows = (nk_u32_t)n_remainder_rows;

    // Compute memory region offsets
    nk_size_t const tiles_offset = sizeof(nk_dots_amx_packed_header_t);
    nk_size_t const n_edge_offset = tiles_offset + total_tiles * tile_bytes;
    header->n_edge_offset = (nk_u32_t)n_edge_offset;

    // Pointers to packed data regions
    nk_i8_t *tiles_ptr = (nk_i8_t *)((char *)b_packed + tiles_offset);
    nk_i8_t *n_edge_ptr = (nk_i8_t *)((char *)b_packed + n_edge_offset);

    // Zero-initialize all tiles (handles K remainder padding)
    for (nk_size_t i = 0; i < total_tiles * tile_elements; i++) tiles_ptr[i] = 0;

    // Pack tiles using LINEAR ordering: tile_index = n_tile * num_k_tiles + k_tile
    // This provides sequential memory access when streaming along K dimension.
    for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
        for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {

            // Linear tile index: all K-tiles for one N-tile are contiguous
            nk_size_t const tile_index = n_tile * num_k_tiles + k_tile;
            nk_i8_t *tile_output = tiles_ptr + tile_index * tile_elements;

            // Source coordinates in original B matrix
            nk_size_t const src_row_start = n_tile * tile_rows;
            nk_size_t const src_col_start = k_tile * tile_cols;
            nk_size_t const cols_to_pack = (src_col_start + tile_cols <= k) ? tile_cols : (k - src_col_start);

            // Pack with quad-interleaving as required by TDPBSSD instruction.
            // AMX expects: [col0_row0, col1_row0, col2_row0, col3_row0, col0_row1, ...]
            // Formula: packed_idx = (col / 4) * 64 + row * 4 + (col % 4)
            for (nk_size_t row = 0; row < tile_rows; row++) {
                for (nk_size_t col = 0; col < cols_to_pack; col++) {
                    nk_size_t const src_idx = (src_row_start + row) * b_stride + src_col_start + col;
                    nk_size_t const dst_idx = (col / 4) * 64 + row * 4 + (col % 4);
                    tile_output[dst_idx] = b[src_idx];
                }
            }
        }
    }

    // Pack N-remainder rows in simple row-major format (for AVX-512 fallback)
    if (n_remainder_rows > 0) {
        nk_size_t const remainder_start_row = num_n_tiles * tile_rows;
        for (nk_size_t row = 0; row < n_remainder_rows; row++) {
            for (nk_size_t col = 0; col < k; col++) {
                n_edge_ptr[row * k + col] = b[(remainder_start_row + row) * b_stride + col];
            }
        }
    }
}

/*  BF16 → F32 matmul (aligned path): Direct tile loads/stores when stride >= 64 bytes.
 *
 *  Optimized with:
 *  - Nc=2 panel blocking: process 2 N-blocks (64 columns) at a time to maximize B-tile reuse
 *  - Software pipelining: overlap tile loads with compute operations
 *  - Linear B indexing: sequential memory access along K dimension
 *
 *  AMX tile usage:
 *    TMM0-1: A tiles (2 rows of 16×32 from current M-block)
 *    TMM2-3: B tiles (2 tiles from current N-block)
 *    TMM4-7: C accumulators (2×2 = 4 output tiles of 16×16 each)
 */
NK_INTERNAL void nk_dots_bf16bf16f32_sapphire_aligned_(    //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                 //
    nk_size_t a_stride, nk_size_t c_stride) {

    // Read packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;
    nk_size_t const num_k_tiles = header->full_k_tiles;
    nk_size_t const n_remainder_rows = header->n_edge_rows;

    // Pointers to packed data regions
    nk_bf16_t const *b_tiles = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *n_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->n_edge_offset);

    // Constants for BF16 AMX tiles
    nk_size_t const tile_k_cols = 32;    // K-dimension of one tile
    nk_size_t const tile_elements = 512; // 16 rows × 32 cols

    // Stride conversions
    nk_size_t const a_stride_bf16 = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_f32 = c_stride / sizeof(nk_f32_t);

    // Block dimensions
    nk_size_t const num_m_blocks = m / 32;          // Each M-block = 32 rows (2 tiles)
    nk_size_t const num_n_blocks = num_n_tiles / 2; // Each N-block = 32 cols (2 tiles)
    nk_size_t const full_n_cols = num_n_tiles * 16;

    // Nc=2 panel size: process 2 N-blocks (64 columns) per outer iteration
    // This keeps B tiles hot in L2 while streaming through A rows
    nk_size_t const panel_size = 2;

    // AMX: Full 32×32 output blocks with Nc=2 blocking and software pipelining
    if (num_m_blocks > 0 && num_n_blocks > 0 && num_k_tiles > 0) {
        nk_amx_tile_configure_sapphire_amx_();

        // Outer loop: N-panels of size Nc=2
        for (nk_size_t n_panel_start = 0; n_panel_start < num_n_blocks; n_panel_start += panel_size) {
            nk_size_t const n_panel_end = (n_panel_start + panel_size < num_n_blocks) ? (n_panel_start + panel_size)
                                                                                      : num_n_blocks;

            // Middle loop: all M-blocks (B tiles stay hot for each M-block)
            for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
                nk_size_t const m_row = m_block * 32;

                // A tile base addresses for this M-block
                nk_bf16_t const *a_row0 = a + m_row * a_stride_bf16;
                nk_bf16_t const *a_row1 = a + (m_row + 16) * a_stride_bf16;

                // Inner loop: N-blocks within current panel
                for (nk_size_t n_block = n_panel_start; n_block < n_panel_end; n_block++) {
                    nk_size_t const n_col = n_block * 32;

                    // B tile base indices for this N-block (linear layout)
                    nk_size_t const b_n0_base = (n_block * 2) * num_k_tiles;     // First N-tile
                    nk_size_t const b_n1_base = (n_block * 2 + 1) * num_k_tiles; // Second N-tile

                    // Zero accumulators
                    _tile_zero(4);
                    _tile_zero(5);
                    _tile_zero(6);
                    _tile_zero(7);

                    // Software-pipelined K-loop
                    if (num_k_tiles > 1) {
                        // Prologue: load first tiles
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_elements, 64);

                        // Main loop: compute current, load next
                        for (nk_size_t k_tile = 0; k_tile < num_k_tiles - 1; k_tile++) {
                            nk_size_t const k_offset = k_tile * tile_k_cols;
                            nk_size_t const next_k_offset = (k_tile + 1) * tile_k_cols;

                            // Compute A0×B0, load A1 and B1
                            _tile_dpbf16ps(4, 0, 2);
                            _tile_loadd(1, a_row1 + k_offset, (int)a_stride);
                            _tile_loadd(3, b_tiles + (b_n1_base + k_tile) * tile_elements, 64);

                            // Compute A1×B0, A0×B1, A1×B1
                            _tile_dpbf16ps(6, 1, 2);
                            _tile_dpbf16ps(5, 0, 3);
                            _tile_dpbf16ps(7, 1, 3);

                            // Load next iteration's A0 and B0
                            _tile_loadd(0, a_row0 + next_k_offset, (int)a_stride);
                            _tile_loadd(2, b_tiles + (b_n0_base + k_tile + 1) * tile_elements, 64);
                        }

                        // Epilogue: last K iteration
                        nk_size_t const last_k = num_k_tiles - 1;
                        nk_size_t const last_k_offset = last_k * tile_k_cols;

                        _tile_dpbf16ps(4, 0, 2);
                        _tile_loadd(1, a_row1 + last_k_offset, (int)a_stride);
                        _tile_dpbf16ps(6, 1, 2);
                        _tile_loadd(3, b_tiles + (b_n1_base + last_k) * tile_elements, 64);
                        _tile_dpbf16ps(5, 0, 3);
                        _tile_dpbf16ps(7, 1, 3);
                    }
                    else {
                        // Single K-tile: no pipelining needed
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(1, a_row1, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_elements, 64);
                        _tile_loadd(3, b_tiles + b_n1_base * tile_elements, 64);
                        _tile_dpbf16ps(4, 0, 2);
                        _tile_dpbf16ps(5, 0, 3);
                        _tile_dpbf16ps(6, 1, 2);
                        _tile_dpbf16ps(7, 1, 3);
                    }

                    // Store 2×2 output block directly to C
                    nk_f32_t *c_block = c + m_row * c_stride_f32 + n_col;
                    _tile_stored(4, c_block, (int)c_stride);
                    _tile_stored(5, c_block + 16, (int)c_stride);
                    _tile_stored(6, c_block + 16 * c_stride_f32, (int)c_stride);
                    _tile_stored(7, c_block + 16 * c_stride_f32 + 16, (int)c_stride);
                }
            }
        }

        _tile_release();
    }

    // AVX-512: N-remainder rows (rows beyond full N-tiles)
    if (n_remainder_rows > 0) {
        nk_dots_bf16bf16f32_avx512_edge_(a, n_edge_ptr, c + full_n_cols, m, n_remainder_rows, k, a_stride_bf16, k,
                                         c_stride_f32);
    }

    // AMX: M-remainder rows (rows beyond full M-blocks) for full N-tiles
    if (m > num_m_blocks * 32 && num_n_tiles > 0) {
        nk_size_t const m_remainder_start = num_m_blocks * 32;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        nk_amx_tile_configure_sapphire_amx_();

        // Process each N-tile individually for M-remainder
        for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
            nk_size_t const n_col = n_tile * 16;

            _tile_zero(4);
            _tile_zero(6);

            // Staging buffers for partial A tiles
            NK_ALIGN64 nk_bf16_t a_tile_upper[16][32] = {{0}};
            NK_ALIGN64 nk_bf16_t a_tile_lower[16][32] = {{0}};

            nk_size_t const rows_upper = (m_remainder_count > 16) ? 16 : m_remainder_count;
            nk_size_t const rows_lower = (m_remainder_count > 16) ? m_remainder_count - 16 : 0;

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * tile_k_cols;
                nk_size_t const k_valid = (k_offset + tile_k_cols <= k) ? tile_k_cols : (k - k_offset);

                // Load partial A tiles with masking
                nk_load_a_tile_bf16_masked_(a + m_remainder_start * a_stride_bf16 + k_offset, a_stride_bf16, rows_upper,
                                            k_valid, (nk_bf16_t *)a_tile_upper);
                if (rows_lower > 0) {
                    nk_load_a_tile_bf16_masked_(a + (m_remainder_start + 16) * a_stride_bf16 + k_offset, a_stride_bf16,
                                                rows_lower, k_valid, (nk_bf16_t *)a_tile_lower);
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // Linear B tile index
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                _tile_loadd(2, b_tiles + b_tile_idx * tile_elements, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            // Store with masking for partial rows
            NK_ALIGN64 nk_f32_t c_tile_buf[16][16];

            _tile_stored(4, c_tile_buf, 64);
            nk_store_c_tile_f32_masked_((nk_f32_t *)c_tile_buf, c + m_remainder_start * c_stride_f32 + n_col,
                                        c_stride_f32, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile_buf, 64);
                nk_store_c_tile_f32_masked_((nk_f32_t *)c_tile_buf, c + (m_remainder_start + 16) * c_stride_f32 + n_col,
                                            c_stride_f32, rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: M-remainder × N-remainder corner
    if (m > num_m_blocks * 32 && n_remainder_rows > 0) {
        nk_size_t const m_remainder_start = num_m_blocks * 32;
        nk_size_t const m_remainder_count = m - m_remainder_start;

        nk_dots_bf16bf16f32_avx512_edge_(a + m_remainder_start * a_stride_bf16, n_edge_ptr,
                                         c + m_remainder_start * c_stride_f32 + full_n_cols, m_remainder_count,
                                         n_remainder_rows, k, a_stride_bf16, k, c_stride_f32);
    }
}

/*  BF16 → F32 matmul (misaligned path): All I/O through aligned buffers with AVX-512.
 *  Used when stride < 64 bytes (can't use direct tile loads/stores).
 */
NK_INTERNAL void nk_dots_bf16bf16f32_sapphire_misaligned_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                 //
    nk_size_t a_stride, nk_size_t c_stride) {

    // Read header for hybrid layout
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const full_n_tiles = header->full_n_tiles;
    nk_size_t const full_k_tiles = header->full_k_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;

    nk_bf16_t const *tiles_ptr = (nk_bf16_t const *)((char const *)b_packed + sizeof(nk_dots_amx_packed_header_t));
    nk_bf16_t const *n_edge_ptr = (nk_bf16_t const *)((char const *)b_packed + header->n_edge_offset);

    nk_size_t const tile_cols_bf16 = 32;
    nk_size_t const tile_elements_bf16 = 512;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_size_t const full_n = full_n_tiles * 16;
    nk_size_t const full_m_blocks = m / 32;
    nk_size_t const full_n_blocks = full_n_tiles / 2;
    nk_size_t const total_full_tiles = full_n_tiles * full_k_tiles;

    // Stack buffers for tile I/O
    NK_ALIGN64 nk_bf16_t a_buf_upper[16][32];
    NK_ALIGN64 nk_bf16_t a_buf_lower[16][32];
    NK_ALIGN64 nk_f32_t c_buf[16][16];

    // AMX: Full 32×32 blocks through buffers
    if (full_m_blocks > 0 && full_n_blocks > 0 && full_k_tiles > 0) {
        nk_amx_tile_configure_sapphire_amx_();

        for (nk_size_t bi = 0; bi < full_m_blocks; bi++) {
            nk_size_t const row_block = bi * 32;

            for (nk_size_t bj = 0; bj < full_n_blocks; bj++) {
                nk_size_t const col_block = bj * 32;

                _tile_zero(4);
                _tile_zero(5);
                _tile_zero(6);
                _tile_zero(7);

                nk_size_t const b_tile_n0 = bj * 2;
                nk_size_t const b_tile_n1 = bj * 2 + 1;

                for (nk_size_t bk = 0; bk < full_k_tiles; bk++) {
                    nk_size_t const k_offset = bk * tile_cols_bf16;

                    // Load A through buffers using AVX-512
                    for (nk_size_t r = 0; r < 16; r++) {
                        nk_bf16_t const *src_upper = a + (row_block + r) * a_stride_elements + k_offset;
                        nk_bf16_t const *src_lower = a + (row_block + 16 + r) * a_stride_elements + k_offset;
                        __m512i upper_row = _mm512_loadu_si512((__m512i const *)src_upper);
                        __m512i lower_row = _mm512_loadu_si512((__m512i const *)src_lower);
                        _mm512_store_si512((__m512i *)a_buf_upper[r], upper_row);
                        _mm512_store_si512((__m512i *)a_buf_lower[r], lower_row);
                    }
                    __asm__ volatile("" ::: "memory");

                    _tile_loadd(0, a_buf_upper, 64);
                    _tile_loadd(1, a_buf_lower, 64);

                    // B tiles via Morton indexing
                    nk_size_t morton_idx0 = nk_morton_encode_sapphire_amx_((nk_u32_t)b_tile_n0, (nk_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    nk_bf16_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_bf16;

                    nk_size_t morton_idx1 = nk_morton_encode_sapphire_amx_((nk_u32_t)b_tile_n1, (nk_u32_t)bk);
                    if (morton_idx1 >= total_full_tiles) morton_idx1 = b_tile_n1 * full_k_tiles + bk;
                    nk_bf16_t const *b_tile_ptr1 = tiles_ptr + morton_idx1 * tile_elements_bf16;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbf16ps(4, 0, 2);
                    _tile_dpbf16ps(5, 0, 3);
                    _tile_dpbf16ps(6, 1, 2);
                    _tile_dpbf16ps(7, 1, 3);
                }

                // Store C through buffers using AVX-512
                _tile_stored(4, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + r) * c_stride_elements + col_block, row);
                }

                _tile_stored(5, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + r) * c_stride_elements + col_block + 16, row);
                }

                _tile_stored(6, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + 16 + r) * c_stride_elements + col_block, row);
                }

                _tile_stored(7, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512 row = _mm512_load_ps(c_buf[r]);
                    _mm512_storeu_ps(c + (row_block + 16 + r) * c_stride_elements + col_block + 16, row);
                }
            }
        }

        _tile_release();
    }

    // AVX-512: N edge rows
    if (n_edge_rows > 0) {
        nk_dots_bf16bf16f32_avx512_edge_(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride_elements, k,
                                         c_stride_elements);
    }

    // AMX: M edge rows for full N tiles (through buffers)
    if (m > full_m_blocks * 32 && full_n_tiles > 0) {
        nk_size_t const m_edge_start = full_m_blocks * 32;
        nk_size_t const m_edge_rows = m - m_edge_start;

        nk_amx_tile_configure_sapphire_amx_();

        for (nk_size_t tj = 0; tj < full_n_tiles; tj++) {
            nk_size_t const col_block = tj * 16;
            nk_size_t const b_tile_n0 = tj;

            _tile_zero(4);
            _tile_zero(6);

            // Zero buffers for edge
            for (nk_size_t r = 0; r < 16; r++) {
                _mm512_store_si512((__m512i *)a_buf_upper[r], _mm512_setzero_si512());
                _mm512_store_si512((__m512i *)a_buf_lower[r], _mm512_setzero_si512());
            }

            nk_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            nk_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (nk_size_t bk = 0; bk < full_k_tiles; bk++) {
                nk_size_t const k_offset = bk * tile_cols_bf16;
                nk_size_t const k_valid = (k_offset + tile_cols_bf16 <= k) ? tile_cols_bf16 : (k - k_offset);

                nk_load_a_tile_bf16_masked_(a + m_edge_start * a_stride_elements + k_offset, a_stride_elements,
                                            rows_upper, k_valid, (nk_bf16_t *)a_buf_upper);
                if (rows_lower > 0) {
                    nk_load_a_tile_bf16_masked_(a + (m_edge_start + 16) * a_stride_elements + k_offset,
                                                a_stride_elements, rows_lower, k_valid, (nk_bf16_t *)a_buf_lower);
                }

                _tile_loadd(0, a_buf_upper, 64);
                _tile_loadd(1, a_buf_lower, 64);

                nk_size_t morton_idx0 = nk_morton_encode_sapphire_amx_((nk_u32_t)b_tile_n0, (nk_u32_t)bk);
                if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                nk_bf16_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_bf16;

                _tile_loadd(2, b_tile_ptr0, 64);

                _tile_dpbf16ps(4, 0, 2);
                _tile_dpbf16ps(6, 1, 2);
            }

            _tile_stored(4, c_buf, 64);
            nk_store_c_tile_f32_masked_((nk_f32_t *)c_buf, c + m_edge_start * c_stride_elements + col_block,
                                        c_stride_elements, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_buf, 64);
                nk_store_c_tile_f32_masked_((nk_f32_t *)c_buf, c + (m_edge_start + 16) * c_stride_elements + col_block,
                                            c_stride_elements, rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: M edge × N edge corner
    if (m > full_m_blocks * 32 && n_edge_rows > 0) {
        nk_size_t const m_edge_start = full_m_blocks * 32;
        nk_size_t const m_edge_count = m - m_edge_start;

        nk_dots_bf16bf16f32_avx512_edge_(a + m_edge_start * a_stride_elements, n_edge_ptr,
                                         c + m_edge_start * c_stride_elements + full_n, m_edge_count, n_edge_rows, k,
                                         a_stride_elements, k, c_stride_elements);
    }
}

/*  BF16 → F32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Dispatcher that selects aligned or misaligned path based on stride.
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
NK_PUBLIC void nk_dots_bf16bf16f32_sapphire_amx(           //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,                 //
    nk_size_t a_stride, nk_size_t c_stride) {

    // Check if strides allow direct tile operations (need 64 bytes = 32 BF16 or 16 F32)
    int const can_direct = (a_stride >= 64) && (c_stride >= 64);
    if (can_direct) nk_dots_bf16bf16f32_sapphire_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
    else nk_dots_bf16bf16f32_sapphire_misaligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/*  BF16 compact: truncate F32 → BF16 in-place using AVX512.
 *  Reads F32 matrix, writes BF16 to same buffer (safe since F32 is larger).
 *  Uses masked loads/stores to handle all sizes without scalar fallback.
 *  Output is tightly packed with stride = n * sizeof(bf16).
 */
NK_PUBLIC void nk_dots_bf16bf16bf16_sapphire_amx( //
    void *c, nk_size_t m, nk_size_t n,            //
    nk_size_t c_stride) {

    nk_size_t const c_stride_f32 = c_stride / sizeof(nk_f32_t);
    nk_f32_t const *c_f32 = (nk_f32_t const *)c;
    nk_bf16_t *c_bf16 = (nk_bf16_t *)c;

    for (nk_size_t row = 0; row < m; row++) {
        nk_f32_t const *src_row = c_f32 + row * c_stride_f32;
        nk_bf16_t *dst_row = c_bf16 + row * n;
        nk_size_t col = 0;

        // Process 16 floats at a time using AVX512-BF16
        for (; col + 16 <= n; col += 16) {
            __m512 f32_vec = _mm512_loadu_ps(src_row + col);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_storeu_si256((__m256i *)(dst_row + col), (__m256i)bf16_vec);
        }

        // Handle remaining elements with masked operations
        if (col < n) {
            __mmask16 tail_mask = (__mmask16)((1u << (n - col)) - 1);
            __m512 f32_vec = _mm512_maskz_loadu_ps(tail_mask, src_row + col);
            __m256bh bf16_vec = _mm512_cvtneps_pbh(f32_vec);
            _mm256_mask_storeu_epi16(dst_row + col, tail_mask, (__m256i)bf16_vec);
        }
    }
}

/*  I8 → I32 matmul (aligned path): Direct tile loads/stores when stride >= 64 bytes.
 *  This is the fast path - no intermediate buffers for A or C.
 *
 *  Optimization strategy:
 *  - Nc=2 panel blocking: Process 2 N-blocks (64 columns) per outer iteration
 *    to maximize B tile reuse across M-blocks
 *  - Linear B indexing: tile_index = n_tile * num_k_tiles + k_tile for sequential
 *    memory access when streaming along K dimension
 *  - Software pipelining: Overlap tile loads with compute operations
 *  - 2×2 output blocking: 4 accumulator tiles (TMM4-7) for each 32×32 C block
 */
NK_INTERNAL void nk_dots_i8i8i32_sapphire_aligned_(      //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,               //
    nk_size_t a_stride, nk_size_t c_stride) {

    // Parse packed B header
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const num_n_tiles = header->full_n_tiles;        // Number of 16-column N tiles
    nk_size_t const num_k_tiles = header->full_k_tiles;           // Number of 64-element K tiles
    nk_size_t const n_edge_cols = header->n_edge_rows;     // Columns in N edge (0-15)
    nk_size_t const n_edge_offset = header->n_edge_offset;   // Byte offset to N edge data

    // AMX I8 tile dimensions: 16 rows × 64 columns = 1024 I8 elements = 1KB
    nk_size_t const tile_k_elements = 64;  // K elements per tile
    nk_size_t const tile_byte_size = 1024; // Total bytes per packed tile

    // Pointer to packed B tiles (after 64-byte header)
    nk_i8_t const *b_tiles = (nk_i8_t const *)((char const *)b_packed + 64);
    nk_i8_t const *n_edge_tiles = (n_edge_offset > 0) ? (nk_i8_t const *)((char const *)b_packed + n_edge_offset)
                                                      : NULL;

    // Dimension calculations
    nk_size_t const c_stride_i32 = c_stride / sizeof(nk_i32_t); // C stride in elements
    nk_size_t const num_m_blocks = m / 32;                      // Number of 32-row M blocks
    nk_size_t const num_n_blocks = n / 32;                      // Number of 32-column N blocks (each = 2 N tiles)
    nk_size_t const full_n_cols = num_n_tiles * 16;             // Total columns covered by full N tiles

    // Nc=2 panel size: process 2 N-blocks (64 columns) per outer iteration
    // This keeps 2 × 2 × num_k_tiles B tiles hot in L2 cache
    nk_size_t const panel_size = 2;

    // AMX: Full 32×32 blocks with direct I/O
    if (num_m_blocks > 0 && num_n_blocks > 0 && num_k_tiles > 0) {
        nk_amx_tile_configure_sapphire_amx_();

        // Outer loop: N-panels of size Nc=2
        for (nk_size_t n_panel_start = 0; n_panel_start < num_n_blocks; n_panel_start += panel_size) {
            nk_size_t const n_panel_end = (n_panel_start + panel_size < num_n_blocks) ? (n_panel_start + panel_size)
                                                                                      : num_n_blocks;

            // Middle loop: all M-blocks (B tiles stay hot for each M-block)
            for (nk_size_t m_block = 0; m_block < num_m_blocks; m_block++) {
                nk_size_t const m_row = m_block * 32; // Starting row in A and C

                // Pointer to A rows for this M-block
                nk_i8_t const *a_row0 = a + m_row * a_stride;        // Upper 16 rows
                nk_i8_t const *a_row1 = a + (m_row + 16) * a_stride; // Lower 16 rows

                // Inner loop: N-blocks within current panel
                for (nk_size_t n_block = n_panel_start; n_block < n_panel_end; n_block++) {
                    nk_size_t const n_col = n_block * 32; // Starting column in C

                    // Initialize accumulator tiles
                    _tile_zero(4); // C[row0:row0+16, col:col+16]
                    _tile_zero(5); // C[row0:row0+16, col+16:col+32]
                    _tile_zero(6); // C[row0+16:row0+32, col:col+16]
                    _tile_zero(7); // C[row0+16:row0+32, col+16:col+32]

                    // B tile base indices in packed buffer (linear layout)
                    // Linear: tile_index = n_tile * num_k_tiles + k_tile
                    nk_size_t const b_n0_base = (n_block * 2) * num_k_tiles;     // Left B column
                    nk_size_t const b_n1_base = (n_block * 2 + 1) * num_k_tiles; // Right B column

                    // Software-pipelined K-loop
                    if (num_k_tiles > 1) {
                        // Prologue: Load first A and B tiles
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_byte_size, 64);

                        // Main loop: compute current tiles while loading next
                        for (nk_size_t k_tile = 0; k_tile < num_k_tiles - 1; k_tile++) {
                            nk_size_t const k_offset = k_tile * tile_k_elements;
                            nk_size_t const next_k_offset = (k_tile + 1) * tile_k_elements;

                            // Compute: A0 × B0 → C[0,0]
                            _tile_dpbssd(4, 0, 2);

                            // Load: A1 (lower rows), B1 (right column)
                            _tile_loadd(1, a_row1 + k_offset, (int)a_stride);
                            _tile_loadd(3, b_tiles + (b_n1_base + k_tile) * tile_byte_size, 64);

                            // Compute: A1 × B0 → C[1,0], A0 × B1 → C[0,1], A1 × B1 → C[1,1]
                            _tile_dpbssd(6, 1, 2);
                            _tile_dpbssd(5, 0, 3);
                            _tile_dpbssd(7, 1, 3);

                            // Load next iteration: A0, B0
                            _tile_loadd(0, a_row0 + next_k_offset, (int)a_stride);
                            _tile_loadd(2, b_tiles + (b_n0_base + k_tile + 1) * tile_byte_size, 64);
                        }

                        // Epilogue: Process last K tile
                        nk_size_t const last_k = num_k_tiles - 1;
                        nk_size_t const last_k_offset = last_k * tile_k_elements;

                        _tile_dpbssd(4, 0, 2);
                        _tile_loadd(1, a_row1 + last_k_offset, (int)a_stride);
                        _tile_loadd(3, b_tiles + (b_n1_base + last_k) * tile_byte_size, 64);
                        _tile_dpbssd(6, 1, 2);
                        _tile_dpbssd(5, 0, 3);
                        _tile_dpbssd(7, 1, 3);
                    }
                    else {
                        // Single K tile: no pipelining needed
                        _tile_loadd(0, a_row0, (int)a_stride);
                        _tile_loadd(1, a_row1, (int)a_stride);
                        _tile_loadd(2, b_tiles + b_n0_base * tile_byte_size, 64);
                        _tile_loadd(3, b_tiles + b_n1_base * tile_byte_size, 64);

                        _tile_dpbssd(4, 0, 2);
                        _tile_dpbssd(5, 0, 3);
                        _tile_dpbssd(6, 1, 2);
                        _tile_dpbssd(7, 1, 3);
                    }

                    // Store C tiles directly (aligned path)
                    nk_i32_t *c_block = c + m_row * c_stride_i32 + n_col;
                    _tile_stored(4, c_block, (int)c_stride);
                    _tile_stored(5, c_block + 16, (int)c_stride);
                    _tile_stored(6, c_block + 16 * c_stride_i32, (int)c_stride);
                    _tile_stored(7, c_block + 16 * c_stride_i32 + 16, (int)c_stride);
                }
            }
        }

        _tile_release();
    }

    // AMX: M edge rows for full N tiles (rows that don't fill a complete 32-row block)
    if (m > num_m_blocks * 32 && num_n_tiles > 0) {
        nk_size_t const m_edge_start = num_m_blocks * 32;
        nk_size_t const m_edge_rows = m - m_edge_start;

        nk_amx_tile_configure_sapphire_amx_();

        for (nk_size_t n_tile = 0; n_tile < num_n_tiles; n_tile++) {
            nk_size_t const n_col = n_tile * 16;

            _tile_zero(4);
            _tile_zero(6);

            NK_ALIGN64 nk_i8_t a_tile_upper[16][64] = {{0}};
            NK_ALIGN64 nk_i8_t a_tile_lower[16][64] = {{0}};

            nk_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            nk_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (nk_size_t k_tile = 0; k_tile < num_k_tiles; k_tile++) {
                nk_size_t const k_offset = k_tile * tile_k_elements;
                nk_size_t const k_valid = (k_offset + tile_k_elements <= k) ? tile_k_elements : (k - k_offset);

                nk_load_a_tile_i8_masked_(a + m_edge_start * a_stride + k_offset, a_stride, rows_upper, k_valid,
                                          (nk_i8_t *)a_tile_upper);
                if (rows_lower > 0) {
                    nk_load_a_tile_i8_masked_(a + (m_edge_start + 16) * a_stride + k_offset, a_stride, rows_lower,
                                              k_valid, (nk_i8_t *)a_tile_lower);
                }

                _tile_loadd(0, a_tile_upper, 64);
                _tile_loadd(1, a_tile_lower, 64);

                // Linear B tile index
                nk_size_t const b_tile_idx = n_tile * num_k_tiles + k_tile;
                _tile_loadd(2, b_tiles + b_tile_idx * tile_byte_size, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            NK_ALIGN64 nk_i32_t c_tile[16][16];

            _tile_stored(4, c_tile, 64);
            nk_store_c_tile_i32_masked_((nk_i32_t *)c_tile, c + m_edge_start * c_stride_i32 + n_col, c_stride_i32,
                                        rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_tile, 64);
                nk_store_c_tile_i32_masked_((nk_i32_t *)c_tile, c + (m_edge_start + 16) * c_stride_i32 + n_col,
                                            c_stride_i32, rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: N edge columns (columns that don't fill a complete 16-column tile)
    if (n_edge_cols > 0 && n_edge_tiles != NULL) {
        nk_dots_i8i8i32_avx512_edge_(a, n_edge_tiles, c + full_n_cols, m, n_edge_cols, k, a_stride, k, c_stride_i32);
    }

    // AVX-512: M edge × N edge corner
    if (m > num_m_blocks * 32 && n_edge_cols > 0 && n_edge_tiles != NULL) {
        nk_size_t const m_edge_start = num_m_blocks * 32;
        nk_size_t const m_edge_rows = m - m_edge_start;

        nk_dots_i8i8i32_avx512_edge_(a + m_edge_start * a_stride, n_edge_tiles,
                                     c + m_edge_start * c_stride_i32 + full_n_cols, m_edge_rows, n_edge_cols, k,
                                     a_stride, k, c_stride_i32);
    }
}

/*  I8 → I32 matmul (misaligned path): All I/O through aligned buffers with AVX-512.
 *  Used when stride < 64 bytes (can't use direct tile loads/stores).
 */
NK_INTERNAL void nk_dots_i8i8i32_sapphire_misaligned_(   //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,               //
    nk_size_t a_stride, nk_size_t c_stride) {

    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_size_t const full_n_tiles = header->full_n_tiles;
    nk_size_t const full_k_tiles = header->full_k_tiles;
    nk_size_t const n_edge_rows = header->n_edge_rows;
    nk_size_t const n_edge_offset = header->n_edge_offset;

    nk_size_t const tile_cols_i8 = 64;
    nk_size_t const tile_elements_i8 = 1024;

    nk_i8_t const *tiles_ptr = (nk_i8_t const *)((char const *)b_packed + 64);
    nk_i8_t const *n_edge_ptr = (n_edge_offset > 0) ? (nk_i8_t const *)((char const *)b_packed + n_edge_offset) : NULL;

    nk_size_t const c_stride_elements = c_stride / sizeof(nk_i32_t);
    nk_size_t const full_m_blocks = m / 32;
    nk_size_t const full_n_blocks = n / 32;
    nk_size_t const full_n = full_n_tiles * 16;
    nk_size_t const total_full_tiles = full_n_tiles * full_k_tiles;

    // Stack buffers for tile I/O
    NK_ALIGN64 nk_i8_t a_buf_upper[16][64];
    NK_ALIGN64 nk_i8_t a_buf_lower[16][64];
    NK_ALIGN64 nk_i32_t c_buf[16][16];

    // AMX: Full 32×32 blocks through buffers
    if (full_m_blocks > 0 && full_n_blocks > 0 && full_k_tiles > 0) {
        nk_amx_tile_configure_sapphire_amx_();

        for (nk_size_t bi = 0; bi < full_m_blocks; bi++) {
            nk_size_t const row_block = bi * 32;

            for (nk_size_t bj = 0; bj < full_n_blocks; bj++) {
                nk_size_t const col_block = bj * 32;

                _tile_zero(4);
                _tile_zero(5);
                _tile_zero(6);
                _tile_zero(7);

                nk_size_t const b_tile_n0 = bj * 2;
                nk_size_t const b_tile_n1 = bj * 2 + 1;

                for (nk_size_t bk = 0; bk < full_k_tiles; bk++) {
                    nk_size_t const k_offset = bk * 64;

                    // Load A through buffers using AVX-512
                    for (nk_size_t r = 0; r < 16; r++) {
                        nk_i8_t const *src_upper = a + (row_block + r) * a_stride + k_offset;
                        nk_i8_t const *src_lower = a + (row_block + 16 + r) * a_stride + k_offset;
                        __m512i upper_row = _mm512_loadu_si512((__m512i const *)src_upper);
                        __m512i lower_row = _mm512_loadu_si512((__m512i const *)src_lower);
                        _mm512_store_si512((__m512i *)a_buf_upper[r], upper_row);
                        _mm512_store_si512((__m512i *)a_buf_lower[r], lower_row);
                    }
                    __asm__ volatile("" ::: "memory");

                    _tile_loadd(0, a_buf_upper, 64);
                    _tile_loadd(1, a_buf_lower, 64);

                    nk_size_t morton_idx0 = nk_morton_encode_sapphire_amx_((nk_u32_t)b_tile_n0, (nk_u32_t)bk);
                    if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                    nk_i8_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_i8;

                    nk_size_t morton_idx1 = nk_morton_encode_sapphire_amx_((nk_u32_t)b_tile_n1, (nk_u32_t)bk);
                    if (morton_idx1 >= total_full_tiles) morton_idx1 = b_tile_n1 * full_k_tiles + bk;
                    nk_i8_t const *b_tile_ptr1 = tiles_ptr + morton_idx1 * tile_elements_i8;

                    _tile_loadd(2, b_tile_ptr0, 64);
                    _tile_loadd(3, b_tile_ptr1, 64);

                    _tile_dpbssd(4, 0, 2);
                    _tile_dpbssd(5, 0, 3);
                    _tile_dpbssd(6, 1, 2);
                    _tile_dpbssd(7, 1, 3);
                }

                // Store C through buffers using AVX-512
                _tile_stored(4, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + r) * c_stride_elements + col_block), row);
                }

                _tile_stored(5, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + r) * c_stride_elements + col_block + 16), row);
                }

                _tile_stored(6, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + 16 + r) * c_stride_elements + col_block), row);
                }

                _tile_stored(7, c_buf, 64);
                for (nk_size_t r = 0; r < 16; r++) {
                    __m512i row = _mm512_load_si512((__m512i const *)c_buf[r]);
                    _mm512_storeu_si512((__m512i *)(c + (row_block + 16 + r) * c_stride_elements + col_block + 16),
                                        row);
                }
            }
        }

        _tile_release();
    }

    // AMX: M edge rows for full N tiles (through buffers)
    if (m > full_m_blocks * 32 && full_n_tiles > 0) {
        nk_size_t const m_edge_start = full_m_blocks * 32;
        nk_size_t const m_edge_rows = m - m_edge_start;

        nk_amx_tile_configure_sapphire_amx_();

        for (nk_size_t tj = 0; tj < full_n_tiles; tj++) {
            nk_size_t const col_block = tj * 16;
            nk_size_t const b_tile_n0 = tj;

            _tile_zero(4);
            _tile_zero(6);

            // Zero buffers for edge
            for (nk_size_t r = 0; r < 16; r++) {
                _mm512_store_si512((__m512i *)a_buf_upper[r], _mm512_setzero_si512());
                _mm512_store_si512((__m512i *)a_buf_lower[r], _mm512_setzero_si512());
            }

            nk_size_t const rows_upper = (m_edge_rows > 16) ? 16 : m_edge_rows;
            nk_size_t const rows_lower = (m_edge_rows > 16) ? m_edge_rows - 16 : 0;

            for (nk_size_t bk = 0; bk < full_k_tiles; bk++) {
                nk_size_t const k_offset = bk * tile_cols_i8;
                nk_size_t const k_valid = (k_offset + tile_cols_i8 <= k) ? tile_cols_i8 : (k - k_offset);

                nk_load_a_tile_i8_masked_(a + m_edge_start * a_stride + k_offset, a_stride, rows_upper, k_valid,
                                          (nk_i8_t *)a_buf_upper);
                if (rows_lower > 0) {
                    nk_load_a_tile_i8_masked_(a + (m_edge_start + 16) * a_stride + k_offset, a_stride, rows_lower,
                                              k_valid, (nk_i8_t *)a_buf_lower);
                }

                _tile_loadd(0, a_buf_upper, 64);
                _tile_loadd(1, a_buf_lower, 64);

                nk_size_t morton_idx0 = nk_morton_encode_sapphire_amx_((nk_u32_t)b_tile_n0, (nk_u32_t)bk);
                if (morton_idx0 >= total_full_tiles) morton_idx0 = b_tile_n0 * full_k_tiles + bk;
                nk_i8_t const *b_tile_ptr0 = tiles_ptr + morton_idx0 * tile_elements_i8;

                _tile_loadd(2, b_tile_ptr0, 64);

                _tile_dpbssd(4, 0, 2);
                _tile_dpbssd(6, 1, 2);
            }

            _tile_stored(4, c_buf, 64);
            nk_store_c_tile_i32_masked_((nk_i32_t *)c_buf, c + m_edge_start * c_stride_elements + col_block,
                                        c_stride_elements, rows_upper, 16);

            if (rows_lower > 0) {
                _tile_stored(6, c_buf, 64);
                nk_store_c_tile_i32_masked_((nk_i32_t *)c_buf, c + (m_edge_start + 16) * c_stride_elements + col_block,
                                            c_stride_elements, rows_lower, 16);
            }
        }

        _tile_release();
    }

    // AVX-512: N edge rows
    if (n_edge_rows > 0 && n_edge_ptr != NULL) {
        nk_dots_i8i8i32_avx512_edge_(a, n_edge_ptr, c + full_n, m, n_edge_rows, k, a_stride, k, c_stride_elements);
    }

    // AVX-512: M edge × N edge corner
    if (m > full_m_blocks * 32 && n_edge_rows > 0 && n_edge_ptr != NULL) {
        nk_size_t const m_edge_start = full_m_blocks * 32;
        nk_size_t const m_edge_rows = m - m_edge_start;

        nk_dots_i8i8i32_avx512_edge_(a + m_edge_start * a_stride, n_edge_ptr,
                                     c + m_edge_start * c_stride_elements + full_n, m_edge_rows, n_edge_rows, k,
                                     a_stride, k, c_stride_elements);
    }
}

/*  I8 → I32 matmul: C[m×n] = A[m×k] × B[n×k]ᵀ
 *
 *  Dispatcher that selects aligned or misaligned path based on stride.
 *  Single-threaded. For parallel execution, partition A rows across threads.
 */
NK_PUBLIC void nk_dots_i8i8i32_sapphire_amx(             //
    nk_i8_t const *a, void const *b_packed, nk_i32_t *c, //
    nk_size_t m, nk_size_t n, nk_size_t k,               //
    nk_size_t a_stride, nk_size_t c_stride) {

    // Check if strides allow direct tile operations (need 64 bytes for I8 A tile row and I32 C tile row)
    int const can_direct = (a_stride >= 64) && (c_stride >= 64);
    if (can_direct) nk_dots_i8i8i32_sapphire_aligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
    else nk_dots_i8i8i32_sapphire_misaligned_(a, b_packed, c, m, n, k, a_stride, c_stride);
}

/*  I8 compact: re-normalize I32 → I8 using precomputed squared norms.
 *  Formula: c_i8[i][j] = c_i32[i][j] * 127 * rsqrt(a_norm[i] * b_norm[j])
 *  Uses AVX512 rsqrt14 with Newton-Raphson refinement for 16 elements at a time.
 *  Output is tightly packed with stride = n * sizeof(i8).
 */
NK_PUBLIC void nk_dots_i8i8i8_sapphire_amx( //
    void *c, nk_size_t m, nk_size_t n,      //
    nk_size_t c_stride,                     //
    nk_i32_t const *a_squared_norms, nk_i32_t const *b_squared_norms) {

    nk_size_t const c_stride_i32 = c_stride / sizeof(nk_i32_t);
    nk_i32_t const *c_i32 = (nk_i32_t const *)c;
    nk_i8_t *c_i8 = (nk_i8_t *)c;

    // Use space after I8 output for precomputed b_rsqrt (I8 output is 4x smaller than I32 input)
    nk_f32_t *b_rsqrt = (nk_f32_t *)(c_i8 + m * n);

    // Precompute rsqrt of all b_norms using AVX512 (16 at a time)
    __m512 half_vec = _mm512_set1_ps(0.5f);
    __m512 three_halves_vec = _mm512_set1_ps(1.5f);
    nk_size_t j = 0;

    for (; j + 16 <= n; j += 16) {
        __m512i b_norms_i32 = _mm512_loadu_si512(b_squared_norms + j);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        // Newton-Raphson refinement
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        // Zero out rsqrt where norm was zero
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask, rsqrt_vec);
        _mm512_storeu_ps(b_rsqrt + j, rsqrt_vec);
    }

    // Handle remaining b_norms with masked operations
    if (j < n) {
        __mmask16 tail_mask = (__mmask16)((1u << (n - j)) - 1);
        __m512i b_norms_i32 = _mm512_maskz_loadu_epi32(tail_mask, b_squared_norms + j);
        __m512 b_norms_f32 = _mm512_cvtepi32_ps(b_norms_i32);
        __m512 rsqrt_vec = _mm512_rsqrt14_ps(b_norms_f32);
        rsqrt_vec = _mm512_mul_ps(
            rsqrt_vec,
            _mm512_sub_ps(three_halves_vec,
                          _mm512_mul_ps(half_vec, _mm512_mul_ps(b_norms_f32, _mm512_mul_ps(rsqrt_vec, rsqrt_vec)))));
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(b_norms_i32, _mm512_setzero_si512());
        rsqrt_vec = _mm512_maskz_mov_ps(nonzero_mask & tail_mask, rsqrt_vec);
        _mm512_mask_storeu_ps(b_rsqrt + j, tail_mask, rsqrt_vec);
    }

    __m512 scale_vec = _mm512_set1_ps(127.0f);

    for (nk_size_t row = 0; row < m; row++) {
        nk_i32_t const *src_row = c_i32 + row * c_stride_i32;
        nk_i8_t *dst_row = c_i8 + row * n;

        // Compute rsqrt of a_norm for this row, broadcast to vector
        nk_f32_t a_norm_f32 = (nk_f32_t)a_squared_norms[row];
        nk_f32_t a_rsqrt_val = 0.0f;
        if (a_norm_f32 > 0.0f) {
            __m128 a_vec = _mm_set_ss(a_norm_f32);
            __m128 rsqrt_s = _mm_rsqrt_ss(a_vec);
            rsqrt_s = _mm_mul_ss(
                rsqrt_s, _mm_sub_ss(_mm_set_ss(1.5f),
                                    _mm_mul_ss(_mm_set_ss(0.5f), _mm_mul_ss(a_vec, _mm_mul_ss(rsqrt_s, rsqrt_s)))));
            a_rsqrt_val = _mm_cvtss_f32(rsqrt_s);
        }
        __m512 a_rsqrt_vec = _mm512_set1_ps(a_rsqrt_val);
        __m512 row_scale = _mm512_mul_ps(a_rsqrt_vec, scale_vec);

        nk_size_t col = 0;

        // Process 16 elements at a time
        for (; col + 16 <= n; col += 16) {
            __m512i c_vals = _mm512_loadu_si512(src_row + col);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_loadu_ps(b_rsqrt + col);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            // Saturating pack I32 → I8 (16 values → 16 bytes in low 128 bits)
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_storeu_si128((__m128i *)(dst_row + col), result_i8);
        }

        // Handle remaining elements with masked operations
        if (col < n) {
            __mmask16 tail_mask = (__mmask16)((1u << (n - col)) - 1);
            __m512i c_vals = _mm512_maskz_loadu_epi32(tail_mask, src_row + col);
            __m512 c_f32 = _mm512_cvtepi32_ps(c_vals);
            __m512 b_rsqrt_vec = _mm512_maskz_loadu_ps(tail_mask, b_rsqrt + col);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(c_f32, row_scale), b_rsqrt_vec);
            __m512i result_i32 = _mm512_cvtps_epi32(normalized);
            __m128i result_i8 = _mm512_cvtsepi32_epi8(result_i32);
            _mm_mask_storeu_epi8(dst_row + col, tail_mask, result_i8);
        }
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#pragma clang attribute pop
#pragma GCC pop_options
#endif // NK_TARGET_SAPPHIRE_AMX
#endif // NK_TARGET_X86_

#endif // NK_DOTS_SAPPHIRE_AMX_H