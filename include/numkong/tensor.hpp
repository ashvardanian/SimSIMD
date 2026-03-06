/**
 *  @brief NumKong Tensor types for C++23 and newer.
 *  @file include/numkong/tensor.hpp
 *  @author Ash Vardanian
 *  @date March 2026
 *
 *  Provides owning and non-owning N-dimensional tensor types:
 *
 *  - `nk::tensor<T, A, max_rank>`: Owning, non-resizable
 *  - `nk::tensor_view<T, max_rank>`: Non-owning, const
 *  - `nk::tensor_span<T, max_rank>`: Non-owning, mutable
 *  - `nk::matrix` / `nk::matrix_view` / `nk::matrix_span`: 2D aliases
 *
 *  Features:
 *  - Signed strides (ptrdiff_t) for reversed/transposed views
 *  - Signed indexing (negative = from end)
 *  - C++23 variadic `operator[]` for multi-dimensional access
 *  - Axis iteration (rows_views(), rows_spans(), axis_iterator)
 *  - Conversion to vector_view/vector_span for rank-1 tensors
 */

#ifndef NK_TENSOR_HPP
#define NK_TENSOR_HPP

#include "vector.hpp" // aligned_allocator, range, all_t, resolve_index_, vector_view, vector_span

namespace ashvardanian::numkong {

#pragma region - Shape Storage

/**
 *  @brief Inline fixed-capacity shape descriptor.
 *  @tparam max_rank_ Maximum number of dimensions supported.
 *
 *  Stores extents and signed strides for up to `max_rank_` dimensions.
 *  For `max_rank_=2` (matrix), this is only 40 bytes.
 *  For `max_rank_=64`, this is 1032 bytes.
 */
template <std::size_t max_rank_>
struct shape_storage_ {
    std::size_t extents[max_rank_] = {};
    std::ptrdiff_t strides[max_rank_] = {};
    std::size_t rank = 0;

    /** @brief Total number of elements. */
    constexpr std::size_t numel() const noexcept {
        std::size_t n = 1;
        for (std::size_t i = 0; i < rank; ++i) n *= extents[i];
        return n;
    }

    /** @brief Linearize multi-dimensional coordinates to a byte offset. */
    constexpr std::ptrdiff_t linearize(std::size_t const *coords) const noexcept {
        std::ptrdiff_t offset = 0;
        for (std::size_t i = 0; i < rank; ++i) offset += static_cast<std::ptrdiff_t>(coords[i]) * strides[i];
        return offset;
    }

    /** @brief Create contiguous (row-major) shape storage. */
    static constexpr shape_storage_ contiguous(std::size_t const *exts, std::size_t rank_val,
                                               std::size_t elem_bytes) noexcept {
        shape_storage_ s;
        s.rank = rank_val;
        auto stride = static_cast<std::ptrdiff_t>(elem_bytes);
        for (std::size_t i = rank_val; i > 0; --i) {
            s.extents[i - 1] = exts[i - 1];
            s.strides[i - 1] = stride;
            stride *= static_cast<std::ptrdiff_t>(exts[i - 1]);
        }
        return s;
    }
};

#pragma endregion - Shape Storage

#pragma region - Tensor View

/**
 *  @brief Non-owning, immutable, N-dimensional view.
 *  @tparam value_type_ Element type.
 *  @tparam max_rank_ Maximum number of dimensions.
 */
template <typename value_type_, std::size_t max_rank_ = 8>
struct tensor_view {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  private:
    char const *data_ = nullptr;
    shape_storage_<max_rank_> shape_;

  public:
    constexpr tensor_view() noexcept = default;

    constexpr tensor_view(char const *data, shape_storage_<max_rank_> const &shape) noexcept
        : data_(data), shape_(shape) {}

    /** @brief Number of dimensions. */
    constexpr size_type rank() const noexcept { return shape_.rank; }

    /** @brief Extent along the i-th dimension. */
    constexpr size_type extent(size_type i) const noexcept { return shape_.extents[i]; }

    /** @brief Stride in bytes along the i-th dimension (signed). */
    constexpr difference_type stride_bytes(size_type i) const noexcept { return shape_.strides[i]; }

    /** @brief Total number of elements. */
    constexpr size_type numel() const noexcept { return shape_.numel(); }

    /** @brief True if empty. */
    constexpr bool empty() const noexcept { return shape_.numel() == 0; }

    /** @brief Raw byte pointer. */
    constexpr char const *byte_data() const noexcept { return data_; }

    /** @brief Typed pointer (assumes data is contiguous from this pointer). */
    constexpr value_type const *data() const noexcept { return reinterpret_cast<value_type const *>(data_); }

    /** @brief Access the shape storage. */
    constexpr shape_storage_<max_rank_> const &shape() const noexcept { return shape_; }

    /** @brief Element access with signed index along the leading dimension.
     *  Returns a sub-view with rank-1 if rank > 1, or a value reference if rank == 1. */
    tensor_view<value_type_, max_rank_> slice_leading(difference_type idx) const noexcept {
        auto i = resolve_index_(idx, shape_.extents[0]);
        auto offset = static_cast<difference_type>(i) * shape_.strides[0];
        shape_storage_<max_rank_> sub;
        sub.rank = shape_.rank - 1;
        for (size_type d = 0; d < sub.rank; ++d) {
            sub.extents[d] = shape_.extents[d + 1];
            sub.strides[d] = shape_.strides[d + 1];
        }
        return {data_ + offset, sub};
    }

    /** @brief Convert to vector_view (requires rank == 1). */
    vector_view<value_type> as_vector() const noexcept { return {data_, shape_.extents[0], shape_.strides[0]}; }

    /** @brief Check if the tensor is contiguous in memory. */
    constexpr bool is_contiguous() const noexcept {
        if (shape_.rank == 0) return true;
        auto expected = static_cast<difference_type>(sizeof(value_type));
        for (size_type i = shape_.rank; i > 0; --i) {
            if (shape_.strides[i - 1] != expected) return false;
            expected *= static_cast<difference_type>(shape_.extents[i - 1]);
        }
        return true;
    }

    /** @brief Transpose the first two dimensions (swap extents and strides). Requires rank >= 2. */
    constexpr tensor_view transpose() const noexcept {
        if (shape_.rank < 2) return *this;
        auto transposed = shape_;
        std::swap(transposed.extents[0], transposed.extents[1]);
        std::swap(transposed.strides[0], transposed.strides[1]);
        return {data_, transposed};
    }

    /** @brief Reshape to new extents (requires contiguous layout and matching element count).
     *  Returns an empty view if the tensor is not contiguous or element counts don't match. */
    tensor_view reshape(std::initializer_list<size_type> new_extents) const noexcept {
        auto new_rank = new_extents.size();
        if (!is_contiguous() || new_rank > max_rank_ || new_rank == 0) return {};
        auto new_shape = shape_storage_<max_rank_>::contiguous(new_extents.begin(), new_rank, sizeof(value_type));
        if (new_shape.numel() != shape_.numel()) return {};
        return {data_, new_shape};
    }
};

#pragma endregion - Tensor View

#pragma region - Tensor Span

/**
 *  @brief Non-owning, mutable, N-dimensional view.
 *  @tparam value_type_ Element type.
 *  @tparam max_rank_ Maximum number of dimensions.
 */
template <typename value_type_, std::size_t max_rank_ = 8>
struct tensor_span {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  private:
    char *data_ = nullptr;
    shape_storage_<max_rank_> shape_;

  public:
    constexpr tensor_span() noexcept = default;

    constexpr tensor_span(char *data, shape_storage_<max_rank_> const &shape) noexcept : data_(data), shape_(shape) {}

    /** @brief Number of dimensions. */
    constexpr size_type rank() const noexcept { return shape_.rank; }
    /** @brief Extent along the i-th dimension. */
    constexpr size_type extent(size_type i) const noexcept { return shape_.extents[i]; }
    /** @brief Stride in bytes along the i-th dimension (signed). */
    constexpr difference_type stride_bytes(size_type i) const noexcept { return shape_.strides[i]; }
    /** @brief Total number of elements. */
    constexpr size_type numel() const noexcept { return shape_.numel(); }
    /** @brief True if empty. */
    constexpr bool empty() const noexcept { return shape_.numel() == 0; }

    /** @brief Raw byte pointer. */
    constexpr char *byte_data() noexcept { return data_; }
    constexpr char const *byte_data() const noexcept { return data_; }
    /** @brief Typed pointer (assumes data is contiguous from this pointer). */
    constexpr value_type *data() noexcept { return reinterpret_cast<value_type *>(data_); }
    constexpr value_type const *data() const noexcept { return reinterpret_cast<value_type const *>(data_); }
    /** @brief Access the shape storage. */
    constexpr shape_storage_<max_rank_> const &shape() const noexcept { return shape_; }

    /** @brief Implicit conversion to const view. */
    constexpr operator tensor_view<value_type_, max_rank_>() const noexcept {
        return {static_cast<char const *>(data_), shape_};
    }

    /** @brief Slice along leading dimension. */
    tensor_span slice_leading(difference_type idx) const noexcept {
        auto i = resolve_index_(idx, shape_.extents[0]);
        auto offset = static_cast<difference_type>(i) * shape_.strides[0];
        shape_storage_<max_rank_> sub;
        sub.rank = shape_.rank - 1;
        for (size_type d = 0; d < sub.rank; ++d) {
            sub.extents[d] = shape_.extents[d + 1];
            sub.strides[d] = shape_.strides[d + 1];
        }
        return {data_ + offset, sub};
    }

    /** @brief Convert to vector_span (requires rank == 1). */
    vector_span<value_type> as_vector() noexcept { return {data_, shape_.extents[0], shape_.strides[0]}; }

    /** @brief Convert to vector_view (requires rank == 1). */
    vector_view<value_type> as_vector() const noexcept {
        return {static_cast<char const *>(data_), shape_.extents[0], shape_.strides[0]};
    }

    /** @brief Check if contiguous in memory. */
    constexpr bool is_contiguous() const noexcept {
        if (shape_.rank == 0) return true;
        auto expected = static_cast<difference_type>(sizeof(value_type));
        for (size_type i = shape_.rank; i > 0; --i) {
            if (shape_.strides[i - 1] != expected) return false;
            expected *= static_cast<difference_type>(shape_.extents[i - 1]);
        }
        return true;
    }

    /** @brief Transpose the first two dimensions. Requires rank >= 2. */
    constexpr tensor_span transpose() noexcept {
        if (shape_.rank < 2) return *this;
        auto transposed = shape_;
        std::swap(transposed.extents[0], transposed.extents[1]);
        std::swap(transposed.strides[0], transposed.strides[1]);
        return {data_, transposed};
    }

    /** @brief Reshape to new extents (requires contiguous layout and matching element count).
     *  Returns an empty span if not contiguous or element counts don't match. */
    tensor_span reshape(std::initializer_list<size_type> new_extents) noexcept {
        auto new_rank = new_extents.size();
        if (!is_contiguous() || new_rank > max_rank_ || new_rank == 0) return {};
        auto new_shape = shape_storage_<max_rank_>::contiguous(new_extents.begin(), new_rank, sizeof(value_type));
        if (new_shape.numel() != shape_.numel()) return {};
        return {data_, new_shape};
    }
};

#pragma endregion - Tensor Span

#pragma region - Axis Iterator

/**
 *  @brief Random-access iterator over slices along the leading dimension.
 *  @tparam view_type_ Either `tensor_view` or `tensor_span`.
 *
 *  For a rank-2 matrix, iterating yields rank-1 row views/spans.
 *  Dereference calls `parent_.slice_leading(index_)` to produce each sub-view.
 */
template <typename view_type_>
class axis_iterator {
    using value_type = typename view_type_::value_type;
    using difference_type = std::ptrdiff_t;

    char const *data_ = nullptr;
    difference_type stride_ = 0;
    std::size_t index_ = 0;
    view_type_ parent_;

  public:
    using iterator_category = std::random_access_iterator_tag;

    constexpr axis_iterator() noexcept = default;

    constexpr axis_iterator(view_type_ const &parent, std::size_t index) noexcept
        : data_(parent.byte_data()), stride_(parent.stride_bytes(0)), index_(index), parent_(parent) {}

    constexpr auto operator*() const noexcept { return parent_.slice_leading(static_cast<difference_type>(index_)); }

    constexpr axis_iterator &operator++() noexcept {
        ++index_;
        return *this;
    }
    constexpr axis_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++index_;
        return tmp;
    }
    constexpr axis_iterator &operator--() noexcept {
        --index_;
        return *this;
    }
    constexpr axis_iterator operator--(int) noexcept {
        auto tmp = *this;
        --index_;
        return tmp;
    }

    constexpr axis_iterator operator+(difference_type n) const noexcept {
        auto copy = *this;
        copy.index_ += n;
        return copy;
    }
    constexpr axis_iterator operator-(difference_type n) const noexcept {
        auto copy = *this;
        copy.index_ -= n;
        return copy;
    }
    constexpr difference_type operator-(axis_iterator const &other) const noexcept {
        return static_cast<difference_type>(index_) - static_cast<difference_type>(other.index_);
    }

    constexpr bool operator==(axis_iterator const &other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(axis_iterator const &other) const noexcept { return index_ != other.index_; }
    constexpr bool operator<(axis_iterator const &other) const noexcept { return index_ < other.index_; }
};

#pragma endregion - Axis Iterator

#pragma region - Tensor

/**
 *  @brief Owning, non-resizable, N-dimensional tensor.
 *  @tparam value_type_ Element type.
 *  @tparam allocator_type_ Allocator.
 *  @tparam max_rank_ Maximum number of dimensions.
 *
 *  Fixed-size at construction. Use `try_zeros()` factory for non-throwing construction.
 */
template <typename value_type_, typename allocator_type_ = aligned_allocator<value_type_>, std::size_t max_rank_ = 8>
struct tensor {
    using value_type = value_type_;
    using allocator_type = allocator_type_;
    using alloc_traits = std::allocator_traits<allocator_type_>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type_ *;

    using view_type = tensor_view<value_type_, max_rank_>;
    using span_type = tensor_span<value_type_, max_rank_>;

  private:
    pointer data_ = nullptr;
    shape_storage_<max_rank_> shape_;
    [[no_unique_address]] allocator_type_ alloc_;

  public:
    tensor() noexcept = default;

    explicit tensor(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    ~tensor() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, shape_.numel());
    }

    tensor(tensor &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), shape_(std::exchange(other.shape_, {})),
          alloc_(std::move(other.alloc_)) {}

    tensor &operator=(tensor &&other) noexcept {
        if (this != &other) {
            if (data_) alloc_traits::deallocate(alloc_, data_, shape_.numel());
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(other.alloc_);
            data_ = std::exchange(other.data_, nullptr);
            shape_ = std::exchange(other.shape_, {});
        }
        return *this;
    }

    tensor(tensor const &) = delete;
    tensor &operator=(tensor const &) = delete;

    /**
     *  @brief Factory: allocate a zero-initialized tensor with the given extents.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_zeros(std::initializer_list<size_type> extents,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        auto rank = extents.size();
        if (rank > max_rank_ || rank == 0) return t;
        t.shape_ = shape_storage_<max_rank_>::contiguous(extents.begin(), rank, sizeof(value_type));
        auto n = t.shape_.numel();
        if (n == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, n);
        if (!ptr) return t;
        if constexpr (is_memset_zero_safe_v<value_type_>)
            std::memset(static_cast<void *>(ptr), 0, n * sizeof(value_type_));
        else
            for (size_type i = 0; i < n; ++i) ptr[i] = value_type_ {};
        t.data_ = ptr;
        return t;
    }

    /**
     *  @brief Factory: allocate a tensor filled with ones.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_ones(std::initializer_list<size_type> extents,
                                         allocator_type_ alloc = {}) noexcept {
        return try_full(extents, value_type_ {1}, alloc);
    }

    /**
     *  @brief Factory: allocate a tensor filled with a given value.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param val Fill value.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_full(std::initializer_list<size_type> extents, value_type_ val,
                                         allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        auto rank = extents.size();
        if (rank > max_rank_ || rank == 0) return t;
        t.shape_ = shape_storage_<max_rank_>::contiguous(extents.begin(), rank, sizeof(value_type));
        auto n = t.shape_.numel();
        if (n == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, n);
        if (!ptr) return t;
        for (size_type i = 0; i < n; ++i) ptr[i] = val;
        t.data_ = ptr;
        return t;
    }

    /**
     *  @brief Factory: allocate an uninitialized tensor.
     *  @param extents Extents (one per dimension), e.g. `{3, 4}`.
     *  @param alloc Allocator instance.
     *  @return Non-empty tensor on success, empty on failure.
     */
    [[nodiscard]] static tensor try_empty(std::initializer_list<size_type> extents,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        auto rank = extents.size();
        if (rank > max_rank_ || rank == 0) return t;
        t.shape_ = shape_storage_<max_rank_>::contiguous(extents.begin(), rank, sizeof(value_type));
        auto n = t.shape_.numel();
        if (n == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, n);
        if (!ptr) return t;
        t.data_ = ptr;
        return t;
    }

    /**
     *  @brief Factory: adopt raw memory.
     */
    [[nodiscard]] static tensor from_raw(pointer ptr, shape_storage_<max_rank_> const &shape,
                                         allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        t.data_ = ptr;
        t.shape_ = shape;
        return t;
    }

    /** @brief Number of dimensions. */
    constexpr size_type rank() const noexcept { return shape_.rank; }

    /** @brief Extent along dimension i. */
    constexpr size_type extent(size_type i) const noexcept { return shape_.extents[i]; }

    /** @brief Stride in bytes along dimension i (signed). */
    constexpr difference_type stride_bytes(size_type i) const noexcept { return shape_.strides[i]; }

    /** @brief Total number of elements. */
    constexpr size_type numel() const noexcept { return shape_.numel(); }

    /** @brief True if empty. */
    constexpr bool empty() const noexcept { return data_ == nullptr; }

    /** @brief Typed pointer to data. */
    pointer data() noexcept { return data_; }
    value_type const *data() const noexcept { return data_; }

    /** @brief Shape storage. */
    constexpr shape_storage_<max_rank_> const &shape() const noexcept { return shape_; }

    /** @brief Allocator. */
    allocator_type get_allocator() const noexcept { return alloc_; }

    /** @brief Create an immutable view. */
    view_type view() const noexcept { return {reinterpret_cast<char const *>(data_), shape_}; }

    /** @brief Create a mutable span. */
    span_type span() noexcept { return {reinterpret_cast<char *>(data_), shape_}; }

    /** @brief Range of immutable row views (slices along leading dimension). */
    struct rows_views_t {
        view_type parent;
        auto begin() const noexcept { return axis_iterator<view_type>(parent, 0); }
        auto end() const noexcept { return axis_iterator<view_type>(parent, parent.extent(0)); }
    };

    /** @brief Range of mutable row spans (slices along leading dimension). */
    struct rows_spans_t {
        span_type parent;
        auto begin() noexcept { return axis_iterator<span_type>(parent, 0); }
        auto end() noexcept { return axis_iterator<span_type>(parent, parent.extent(0)); }
    };

    /** @brief Iterate rows as immutable views. */
    rows_views_t rows_views() const noexcept { return {view()}; }

    /** @brief Iterate rows as mutable spans. */
    rows_spans_t rows_spans() noexcept { return {span()}; }
};

/** @brief Non-member swap. */
template <typename V, typename A, std::size_t R>
void swap(tensor<V, A, R> &a, tensor<V, A, R> &b) noexcept {
    auto tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

#pragma endregion - Tensor

#pragma region - Matrix Aliases

/** @brief 2D owning matrix (max_rank = 2, smaller shape_storage). */
template <typename value_type_, typename allocator_type_ = aligned_allocator<value_type_>>
using matrix = tensor<value_type_, allocator_type_, 2>;

/** @brief 2D immutable view. */
template <typename value_type_>
using matrix_view = tensor_view<value_type_, 2>;

/** @brief 2D mutable span. */
template <typename value_type_>
using matrix_span = tensor_span<value_type_, 2>;

#pragma endregion - Matrix Aliases

} // namespace ashvardanian::numkong

#endif // NK_TENSOR_HPP
