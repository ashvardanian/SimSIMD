/**
 *  @file       matrix.hpp
 *  @brief      Helper structures for loading, multiplying, tiling, and logging matrices.
 *  @author     Ash Vardanian
 *  @date       September 14, 2024
 */
#ifndef SIMSIMD_MATRIX_HPP
#define SIMSIMD_MATRIX_HPP

#include <sys/syscall.h> // `syscall`
#include <unistd.h>      // `syscall`

#include <cstdio>   // `std::printf`
#include <cstdlib>  // `std::rand`
#include <typeinfo> // `typeid`

#include <simsimd/types.h>

namespace ashvardanian {
namespace simsimd {

/**
 *  @brief  Generic matrix structure, with row-major layout and loop-unrolled tile loading and unloading operations.
 */
template <typename scalar_at> struct matrix_gt {
    using scalar_t = scalar_at;
    using mutable_scalar_t = std::remove_const_t<scalar_t>;

    scalar_at* data_{};
    std::size_t rows_{};
    std::size_t cols_{};
    std::size_t stride_bytes_{};

    matrix_gt() noexcept = default;
    matrix_gt(scalar_at* data, std::size_t rows, std::size_t cols, std::size_t stride_bytes) noexcept
        : data_(data), rows_(rows), cols_(cols), stride_bytes_(stride_bytes) {}

    template <int rows_ak, int cols_ak>
    matrix_gt(scalar_at (&data)[rows_ak][cols_ak]) noexcept
        : data_(reinterpret_cast<scalar_at*>(data)), rows_(rows_ak), cols_(cols_ak), stride_bytes_(cols_ak * sizeof(scalar_at)) {}

    matrix_gt(matrix_gt const&) = default;
    matrix_gt& operator=(matrix_gt const&) = default;

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    scalar_t* row_data(std::size_t row) noexcept { return reinterpret_cast<scalar_t*>(reinterpret_cast<char*>(data_) + row * stride_bytes_); }
    scalar_t const* row_data(std::size_t row) const noexcept {
        return reinterpret_cast<scalar_t const*>(reinterpret_cast<char const*>(data_) + row * stride_bytes_);
    }

    matrix_gt submatrix(std::size_t row_offset, std::size_t col_offset, std::size_t rows, std::size_t cols) const noexcept {
        return {row_data(row_offset), rows, cols, stride_bytes_};
    }

    scalar_t& operator()(std::size_t row, std::size_t col) noexcept { return row_data(row)[col]; }
    scalar_t operator()(std::size_t row, std::size_t col) const noexcept { return row_data(row)[col]; }
    scalar_t& at(std::size_t row, std::size_t col) noexcept { return row_data(row)[col]; }
    scalar_t at(std::size_t row, std::size_t col) const noexcept { return row_data(row)[col]; }

    void fill(scalar_t value) noexcept {
        for (std::size_t i = 0; i < rows_; i++)
            for (std::size_t j = 0; j < cols_; j++)
                at(i, j) = value;
    }

    /**
     *  @brief Fills the diagonal of the matrix with a given value, similar to NumPy.
     *  https://numpy.org/doc/2.0/reference/generated/numpy.fill_diagonal.html
     */
    void fill_diagonal(scalar_t value) noexcept {
        for (std::size_t i = 0; i < (std::min)(rows_, cols_); i++)
            at(i, i) = value;
    }

    void fill_random(std::int64_t min = -1, std::int64_t max = 1) noexcept {
        for (std::size_t i = 0; i < rows_; i++)
            for (std::size_t j = 0; j < cols_; j++)
                at(i, j) = std::rand() % (max - min + 1) + min;
    }

    void print() const noexcept {
        for (std::size_t i = 0; i < rows_; i++) {
            for (std::size_t j = 0; j < cols_; j++) {
                if constexpr (std::is_unsigned_v<scalar_t>)
                    std::printf("%8llu ", (unsigned long long)at(i, j));
                else if constexpr (std::is_signed_v<scalar_t>)
                    std::printf("%8lld ", (long long)at(i, j));
                else
                    std::printf("%8.2f ", (float)at(i, j));
            }
            std::printf("\n");
        }
    }

    template <std::size_t tile_height_ak, std::size_t tile_width_ak>
    void export_internal_tile(std::size_t tile_row, std::size_t tile_col, mutable_scalar_t* tile) const noexcept {
#pragma omp unroll
        for (std::size_t i = 0; i != tile_height_ak; ++i) {
            auto data_row = row_data(tile_row + i);
            for (std::size_t j = 0; j != tile_width_ak; ++j)
                tile[i * tile_width_ak + j] = data_row[tile_col + j];
        }
    }

    template <std::size_t tile_height_ak, std::size_t tile_width_ak>
    void export_bounding_tile(std::size_t tile_row, std::size_t tile_col, mutable_scalar_t* tile) const noexcept {
        std::memset(tile, 0, tile_height_ak * tile_width_ak * sizeof(mutable_scalar_t));
        std::size_t tile_rows = (std::min)(tile_height_ak, rows_ - tile_row);
        std::size_t tile_cols = (std::min)(tile_width_ak, cols_ - tile_col);
        for (std::size_t i = 0; i != tile_rows; ++i) {
            auto data_row = row_data(tile_row + i);
            for (std::size_t j = 0; j != tile_cols; ++j)
                tile[i * tile_width_ak + j] = data_row[tile_col + j];
        }
    }

    template <std::size_t tile_height_ak, std::size_t tile_width_ak>
    void import_internal_tile(std::size_t tile_row, std::size_t tile_col, mutable_scalar_t const* tile) noexcept {
#pragma omp unroll
        for (std::size_t i = 0; i != tile_height_ak; ++i) {
            auto data_row = row_data(tile_row + i);
            for (std::size_t j = 0; j != tile_width_ak; ++j)
                data_row[tile_col + j] = tile[i * tile_width_ak + j];
        }
    }

    template <std::size_t tile_height_ak, std::size_t tile_width_ak>
    void import_bounding_tile(std::size_t tile_row, std::size_t tile_col, mutable_scalar_t const* tile) noexcept {
        std::size_t tile_rows = (std::min)(tile_height_ak, rows_ - tile_row);
        std::size_t tile_cols = (std::min)(tile_width_ak, cols_ - tile_col);
        for (std::size_t i = 0; i != tile_rows; ++i) {
            auto data_row = row_data(tile_row + i);
            for (std::size_t j = 0; j != tile_cols; ++j)
                data_row[tile_col + j] = tile[i * tile_width_ak + j];
        }
    }
};

/**
 *  @brief  Baseline serial Cross-Corellation (nxcor) implementation for dense matrices.
 */
template <typename first_at, typename second_at, typename result_at>
void nxcor(matrix_gt<first_at> const& a, matrix_gt<second_at> const& b, matrix_gt<result_at>& c) {

    using scalar_result_t = std::remove_const_t<first_at>;
    using scalar_first_t = std::remove_const_t<second_at>;
    using scalar_second_t = std::remove_const_t<result_at>;

    constexpr std::size_t a_tile_rows_k = 16; // Mostly influenced by CPU cache size.
    constexpr std::size_t b_tile_rows_k = 16; // Mostly influenced by CPU cache size and cache line width.
    constexpr std::size_t tile_depth_k = 16;  // Mostly influenced by CPU register width.

    struct tiles_t {
        alignas(64) scalar_first_t a[a_tile_rows_k][tile_depth_k];
        alignas(64) scalar_second_t b[b_tile_rows_k][tile_depth_k];
        alignas(64) scalar_result_t c[a_tile_rows_k][b_tile_rows_k];
    };

    auto multiply_tile = [](tiles_t& tiles) noexcept {
        for (std::size_t i = 0; i < a_tile_rows_k; ++i)
            for (std::size_t j = 0; j < b_tile_rows_k; ++j)
                for (std::size_t k = 0; k < tile_depth_k; ++k)
                    tiles.c[i][j] += tiles.a[i][k] * tiles.b[j][k];
    };

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t i_tile_offset = 0; i_tile_offset < a.rows(); i_tile_offset += a_tile_rows_k) {
        for (std::size_t j_tile_offset = 0; j_tile_offset < b.cols(); j_tile_offset += b_tile_rows_k) {
            bool is_last_in_a = i_tile_offset + a_tile_rows_k >= a.rows();
            bool is_last_in_b = j_tile_offset + b_tile_rows_k >= b.cols();
            bool is_bounding_tile = is_last_in_a || is_last_in_b;

            // Load a tile of C.
            tiles_t tiles;
            c.template export_bounding_tile<a_tile_rows_k, b_tile_rows_k>(i_tile_offset, j_tile_offset, &tiles.c[0][0]);

            // Progress through columns of A and B.
            std::size_t k_tile_offset = 0;
            if (is_bounding_tile) {
                for (; k_tile_offset + tile_depth_k <= a.cols(); k_tile_offset += tile_depth_k) {
                    a.template export_bounding_tile<a_tile_rows_k, tile_depth_k>( //
                        i_tile_offset, k_tile_offset, &tiles.a[0][0]);
                    b.template export_bounding_tile<b_tile_rows_k, tile_depth_k>( //
                        j_tile_offset, k_tile_offset, &tiles.b[0][0]);
                    multiply_tile(tiles);
                }
            } else {
                for (; k_tile_offset + tile_depth_k <= a.cols(); k_tile_offset += tile_depth_k) {
                    a.template export_internal_tile<a_tile_rows_k, tile_depth_k>( //
                        i_tile_offset, k_tile_offset, &tiles.a[0][0]);
                    b.template export_internal_tile<b_tile_rows_k, tile_depth_k>( //
                        j_tile_offset, k_tile_offset, &tiles.b[0][0]);
                    multiply_tile(tiles);
                }
            }

            // Don't forget the tail of each row, if the number of columns is not divisible by the `tile_depth_k`.
            if (k_tile_offset < a.cols()) {
                a.template export_bounding_tile<a_tile_rows_k, tile_depth_k>( //
                    i_tile_offset, k_tile_offset, &tiles.a[0][0]);
                b.template export_bounding_tile<b_tile_rows_k, tile_depth_k>( //
                    j_tile_offset, k_tile_offset, &tiles.b[0][0]);
                multiply_tile(tiles);
            }

            // Store C back.
            c.template import_bounding_tile<a_tile_rows_k, b_tile_rows_k>(i_tile_offset, j_tile_offset, &tiles.c[0][0]);
        }
    }
}

/**
 *  @brief Multiplies two matrices using the Intel AMX instruction set.
 *  @tparam input_type The type of the input matrices ("bf16" or "i8" or "u8").
 *  @tparam output_type The type of the output matrix ("f32" or "i32").
 *  @tparam tile_m The number of rows in the first matrix (default is 16).
 *  @tparam tile_k The number of columns in the first matrix (default is 32).
 *  @tparam tile_n The number of columns in the second matrix (default is 16).
 */
template <typename input_type, typename output_type, int tile_m = 16, int tile_k = 32, int tile_n = 16>
void nxcor_amx(                                  //
    input_type (&matrix_a)[tile_m][tile_k],      // 16 rows * 32 cols * 2 bytes/scalar = 1024 bytes
    input_type (&matrix_b)[tile_n][tile_k],      // transposed(32 rows * 16 cols) * 2 bytes/scalar = 1024 bytes
    output_type (&result_matrix)[tile_m][tile_n] // 16 rows * 16 cols * 4 bytes/scalar = 1024 bytes
) {

    static_assert(                                          //
        sizeof(input_type) * tile_m * tile_k == 1024 &&     //
            sizeof(input_type) * tile_k * tile_n == 1024 && //
            sizeof(output_type) * tile_m * tile_n == 1024,
        "Choose a simple tile size and thank me later");

    // Set up the tile configuration structure
    // There are 8 tile registers from TMM0 to TMM7.
    // Each is 16 rows by 64 bytes, fitting up to 1 KB of data.
    // The actual dimensions can be different and are controlled
    // by `rows` and `colsb` - the width in bytes.
    alignas(64) std::uint8_t tilecfg[64];
    std::memset(tilecfg, 0, sizeof(tilecfg));
    std::uint8_t* palette_id_ptr = &tilecfg[0];
    std::uint16_t* tiles_colsb_ptr = (std::uint16_t*)(&tilecfg[16]);
    std::uint8_t* tiles_rows_ptr = &tilecfg[48];

    *palette_id_ptr = 1; // The only palette currently supported

    // Important to note, AMX doesn't care about the real shape of our matrix,
    // it only cares about it's own tile shape. Keep it simple, otherwise
    // the next person reading this will be painting the walls with their brains.
    tiles_rows_ptr[0] = 16;
    tiles_rows_ptr[1] = 16;
    tiles_rows_ptr[2] = 16;
    tiles_colsb_ptr[0] = 64;
    tiles_colsb_ptr[1] = 64;
    tiles_colsb_ptr[2] = 64;

    _tile_loadconfig(&tilecfg);
    _tile_zero(2);
    _tile_loadd(0, &matrix_a[0][0], 64);

    // The second matrix must be reordered to fit the tile shape
    constexpr int tile_k_pack = (4 / sizeof(input_type));                     // Vertical K packing into Dword
    input_type matrix_b_reordered[tile_k / tile_k_pack][tile_n][tile_k_pack]; // Re-laid B matrix
    for (int k = 0; k < tile_k; ++k)
        for (int n = 0; n < tile_n; ++n)
            // We are shrinking the number of rows in the second matrix by 2x for `bf16` and by 4x for `i8` and `u8`.
            // We are also practically making the rows longer by 2x for `bf16` and by 4x for `i8` and `u8`.
            matrix_b_reordered[k / tile_k_pack][n][k % tile_k_pack] = matrix_b[n][k];
    _tile_loadd(1, &matrix_b_reordered[0][0][0], 64);

    // Here are the shape constraints:
    //
    //      • #UD if srcdest.colbytes mod 4 ≠ 0.
    //      • #UD if src1.colbytes mod 4 ≠ 0.
    //      • #UD if src2.colbytes mod 4 ≠ 0.
    //      • #UD if srcdest.colbytes ≠ src2.colbytes -
    //              why the hell the row width of `f32` destination should
    //              be equal to the row width of `bfloat16` source?!
    //      • #UD if src1.colbytes / 4 ≠ src2.rows.
    //              so this practically means that the second matrix must have 2x
    //              fewer rows than the first one, meaning the number of columns in the
    //              first matrix must be 2x smaller than the number of rows in it!
    //      • #UD if srcdest.rows ≠ src1.rows.
    if constexpr (sizeof(input_type) == 2) {
        _tile_dpbf16ps(2, 0, 1);
    } else {
        _tile_dpbssd(2, 0, 1);
    }

    // Store the result back into the result matrix
    _tile_stored(2, result_matrix, 64);

    // Zero out the tile registers
    _tile_release();
}

template <typename input_type, typename output_type, int tile_m = 16, int tile_k = 32, int tile_n = 16> void try_amx() {
    std::printf("\n\n\n");
    std::printf("Trying AMX with %d x %d x %d matrix multiplication of type %s -> %s\n", tile_m, tile_k, tile_n, typeid(input_type).name(),
                typeid(output_type).name());

    input_type buffer_a[tile_m][tile_k];
    input_type buffer_b[tile_n][tile_k];
    output_type buffer_c[tile_m][tile_n] = {0};

    matrix_gt<input_type> matrix_a{&buffer_a[0][0], tile_m, tile_k, tile_k * sizeof(input_type)};
    matrix_gt<input_type> matrix_b{&buffer_b[0][0], tile_n, tile_k, tile_k * sizeof(input_type)};
    matrix_gt<output_type> matrix_c{&buffer_c[0][0], tile_m, tile_n, tile_n * sizeof(output_type)};

    // Initialize the matrices with values
    // std::iota(&buffer_a[0][0], &buffer_a[tile_m - 1][tile_k - 1] + 1, 1);
    // std::iota(&buffer_b[0][0], &buffer_b[tile_n - 1][tile_k - 1] + 1, 1);
    for (std::size_t row = 0; row != tile_m; ++row)
        for (std::size_t col = 0; col != tile_k; ++col)
            buffer_a[row][col] = row;
    for (std::size_t row = 0; row != tile_n; ++row)
        for (std::size_t col = 0; col != tile_k; ++col)
            buffer_b[row][col] = -(__bf16)row;

    // Perform matrix multiplication using AMX-BF16 inline assembly
    nxcor_amx<input_type, output_type, tile_m, tile_k, tile_n>(buffer_a, buffer_b, buffer_c);
    std::printf("Resulting 16x16 matrix with AMX:\n");
    matrix_c.print();
    (void)matrix_a;
    (void)matrix_b;

    // Compare this to naive multiplication
    output_type buffer_c_serial[tile_m][tile_n] = {0};
    matrix_gt<output_type> matrix_c_serial{&buffer_c_serial[0][0], tile_m, tile_n, tile_n * sizeof(output_type)};
    for (int i = 0; i < tile_m; i++)
        for (int j = 0; j < tile_n; j++)
            for (int k = 0; k < tile_k; k++)
                buffer_c_serial[i][j] += (output_type)buffer_a[i][k] * (output_type)buffer_b[j][k];
    std::printf("Resulting 16x16 matrix after naive multiplication:\n");
    matrix_c_serial.print();
}

#if 0
void check_tiled_amx() {
    simsimd_bf16_t buffer_a[SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE][SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE];
    simsimd_bf16_t buffer_b[SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE][SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE];
    simsimd_bf16_t buffer_c[SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE][SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE];
    simsimd_bf16_t buffer_c_serial[SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE][SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE];

    // The `bf16` resolution can accurately represent integers between -256 and 256, which makes it problematic
    // for testing purposes.
    av::matrix_gt<__bf16> a(buffer_a);
    av::matrix_gt<__bf16> b(buffer_b);
    av::matrix_gt<__bf16> c(buffer_c);
    av::matrix_gt<__bf16> c_serial(buffer_c_serial);

    // a.fill(1.0f);
    // b.fill(1.0f);
    // for (std::size_t row = 0; row != SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE; ++row)
    //     for (std::size_t col = 0; col != SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE; ++col)
    //         buffer_a[row][col] = row, buffer_b[row][col] = -(__bf16)row;
    a.fill_random();
    b.fill_random();
    av::nxcor(a, b, c_serial);
    c_serial.print();

    simsimd_nxcor_bf16_sapphire( //
        SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE, SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE,
        SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE,                                           //
        &buffer_a[0][0], SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE * sizeof(simsimd_bf16_t), //
        &buffer_b[0][0], SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE * sizeof(simsimd_bf16_t), //
        &buffer_c[0][0], SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE * sizeof(simsimd_bf16_t));
    c.print();

    // Find the first mismatch position
    std::size_t row = 0, col = 0;
    for (; row != SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE; ++row)
        for (col = 0; col != SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE; ++col)
            if (buffer_c[row][col] != buffer_c_serial[row][col])
                break;
    if (row != SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE && col != SIMSIMD_NXCOR_BF16_AMX_L1_TILE_SIZE)
        std::printf("Mismatch at row %zu, col %zu\n", row, col);
}
#endif

} // namespace simsimd
} // namespace ashvardanian

#endif
