/**
 *  @brief NumKong packed_matrix type for efficient GEMM.
 *  @file include/numkong/matrix.hpp
 *  @author Ash Vardanian
 *  @date March 2026
 *
 *  Provides a pre-packed matrix type that wraps `dots_pack` / `dots_packed` for
 *  cache-efficient matrix multiplication.
 *
 *  @code
 *  auto b = nk::tensor<nk::f32_t>::try_zeros({256, 512});
 *  auto packed = nk::packed_matrix<nk::f32_t>::try_pack(b.view());
 *  // multiply many times with different A matrices
 *  @endcode
 */

#ifndef NK_MATRIX_HPP
#define NK_MATRIX_HPP

#include "tensor.hpp"
#include "dots.hpp"

namespace ashvardanian::numkong {

/**
 *  @brief Owning, move-only, pre-packed matrix for efficient GEMM.
 *  @tparam value_type_ Element type (e.g., f32_t, bf16_t).
 *  @tparam allocator_type_ Allocator for the packed buffer (default: aligned_allocator<char>).
 *
 *  Wraps `dots_pack` to pre-arrange a matrix B into a cache-friendly layout.
 *  Use `try_pack()` to create from a 2D tensor_view, then call `multiply()` to
 *  compute C = A × B^T.
 */
template <typename value_type_, typename allocator_type_ = aligned_allocator<char>>
struct packed_matrix {
    using value_type = value_type_;
    using result_type = typename value_type_::dot_result_t;
    using allocator_type = allocator_type_;
    using alloc_traits = std::allocator_traits<allocator_type_>;
    using size_type = std::size_t;

  private:
    char *data_ = nullptr;
    size_type size_bytes_ = 0;
    size_type rows_ = 0;  // n (number of rows in B)
    size_type depth_ = 0; // k (number of columns in B)
    [[no_unique_address]] allocator_type_ alloc_;

  public:
    packed_matrix() noexcept = default;

    explicit packed_matrix(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~packed_matrix() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
    }

    packed_matrix(packed_matrix &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_bytes_(std::exchange(other.size_bytes_, 0)),
          rows_(std::exchange(other.rows_, 0)), depth_(std::exchange(other.depth_, 0)),
          alloc_(std::move(other.alloc_)) {}

    packed_matrix &operator=(packed_matrix &&other) noexcept {
        if (this != &other) {
            if (data_) alloc_traits::deallocate(alloc_, data_, size_bytes_);
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(other.alloc_);
            data_ = std::exchange(other.data_, nullptr);
            size_bytes_ = std::exchange(other.size_bytes_, 0);
            rows_ = std::exchange(other.rows_, 0);
            depth_ = std::exchange(other.depth_, 0);
        }
        return *this;
    }

    packed_matrix(packed_matrix const &) = delete;
    packed_matrix &operator=(packed_matrix const &) = delete;

    /**
     *  @brief Pack a 2D tensor_view into cache-efficient layout.
     *  @param b 2D tensor view with rank >= 2. Uses extents[0] as rows, extents[1] as depth.
     *  @param alloc Allocator instance.
     *  @return Non-empty packed_matrix on success, empty on failure.
     */
    template <std::size_t max_rank_>
    [[nodiscard]] static packed_matrix try_pack(tensor_view<value_type_, max_rank_> b,
                                                allocator_type_ alloc = {}) noexcept {
        packed_matrix pm(alloc);
        if (b.rank() < 2) return pm;

        pm.rows_ = b.extent(0);
        pm.depth_ = b.extent(1);
        pm.size_bytes_ = dots_packed_size<value_type_>(pm.rows_, pm.depth_);
        if (pm.size_bytes_ == 0) return pm;

        pm.data_ = alloc_traits::allocate(pm.alloc_, pm.size_bytes_);
        if (!pm.data_) {
            pm.size_bytes_ = 0;
            return pm;
        }

        dots_pack<value_type_>(b.data(), pm.rows_, pm.depth_, static_cast<size_type>(b.stride_bytes(0)), pm.data_);
        return pm;
    }

    /**
     *  @brief Compute C = A × B^T using this pre-packed B.
     *  @tparam max_rank_ Max rank of the tensor types.
     *  @param a 2D tensor_view for matrix A [m x k].
     *  @param c 2D tensor_span for result C [m x n].
     *
     *  Requires: a.extent(1) == depth(), c.extent(0) == a.extent(0), c.extent(1) == rows().
     */
    template <std::size_t max_rank_>
    void multiply(tensor_view<value_type_, max_rank_> a, tensor_span<result_type, max_rank_> c) const noexcept {
        if (!data_ || a.rank() < 2 || c.rank() < 2) return;
        auto m = a.extent(0);
        dots_packed<value_type_>(a.data(), data_, c.data(), m, rows_, depth_, static_cast<size_type>(a.stride_bytes(0)),
                                 static_cast<size_type>(c.stride_bytes(0)));
    }

    /** @brief Number of rows in the packed matrix (n). */
    constexpr size_type rows() const noexcept { return rows_; }

    /** @brief Number of columns / depth (k). */
    constexpr size_type depth() const noexcept { return depth_; }

    /** @brief Size of the packed buffer in bytes. */
    constexpr size_type size_bytes() const noexcept { return size_bytes_; }

    /** @brief True if no matrix is packed. */
    constexpr bool empty() const noexcept { return data_ == nullptr; }

    /** @brief Raw pointer to the packed data. */
    constexpr void const *data() const noexcept { return data_; }
};

} // namespace ashvardanian::numkong

#endif // NK_MATRIX_HPP
