/**
 *  @brief NumKong Tensor types and tensor-level operations for C++23 and newer.
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
 *  Tensor-level free functions:
 *  - Non-allocating scalar results: `sum(view)`, `min(view)`, `max(view)`, etc.
 *  - Allocating ops: `try_add(a, b)`, `try_sum(view, axis)`, etc.
 *  - In-place into pre-allocated output: `add_into(a, b, out)`, etc.
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

#include <cstring> // `std::memset`

#include "vector.hpp" // `aligned_allocator`

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

    /** @brief Reinterpret as a 2D matrix view. Requires rank >= 2. */
    tensor_view<value_type_, 2> as_matrix() const noexcept {
        shape_storage_<2> matrix_shape;
        matrix_shape.rank = 2;
        matrix_shape.extents[0] = shape_.extents[0];
        matrix_shape.extents[1] = shape_.extents[1];
        matrix_shape.strides[0] = shape_.strides[0];
        matrix_shape.strides[1] = shape_.strides[1];
        return {data_, matrix_shape};
    }

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

    /** @brief Reinterpret as a 2D matrix span. Requires rank >= 2. */
    tensor_span<value_type_, 2> as_matrix() noexcept {
        shape_storage_<2> matrix_shape;
        matrix_shape.rank = 2;
        matrix_shape.extents[0] = shape_.extents[0];
        matrix_shape.extents[1] = shape_.extents[1];
        matrix_shape.strides[0] = shape_.strides[0];
        matrix_shape.strides[1] = shape_.strides[1];
        return {data_, matrix_shape};
    }

    /** @brief Reinterpret as a 2D const matrix view. Requires rank >= 2. */
    tensor_view<value_type_, 2> as_matrix() const noexcept {
        shape_storage_<2> matrix_shape;
        matrix_shape.rank = 2;
        matrix_shape.extents[0] = shape_.extents[0];
        matrix_shape.extents[1] = shape_.extents[1];
        matrix_shape.strides[0] = shape_.strides[0];
        matrix_shape.strides[1] = shape_.strides[1];
        return {static_cast<char const *>(data_), matrix_shape};
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

    /** @brief Factory: zero-initialized tensor from pointer + rank. */
    [[nodiscard]] static tensor try_zeros(size_type const *extents, size_type rank,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        if (rank > max_rank_ || rank == 0) return t;
        t.shape_ = shape_storage_<max_rank_>::contiguous(extents, rank, sizeof(value_type));
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

    /** @brief Factory: uninitialized tensor from pointer + rank. */
    [[nodiscard]] static tensor try_empty(size_type const *extents, size_type rank,
                                          allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        if (rank > max_rank_ || rank == 0) return t;
        t.shape_ = shape_storage_<max_rank_>::contiguous(extents, rank, sizeof(value_type));
        auto n = t.shape_.numel();
        if (n == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, n);
        if (!ptr) return t;
        t.data_ = ptr;
        return t;
    }

    /** @brief Factory: filled tensor from pointer + rank. */
    [[nodiscard]] static tensor try_full(size_type const *extents, size_type rank, value_type_ val,
                                         allocator_type_ alloc = {}) noexcept {
        tensor t(alloc);
        if (rank > max_rank_ || rank == 0) return t;
        t.shape_ = shape_storage_<max_rank_>::contiguous(extents, rank, sizeof(value_type));
        auto n = t.shape_.numel();
        if (n == 0) return t;
        pointer ptr = alloc_traits::allocate(t.alloc_, n);
        if (!ptr) return t;
        for (size_type i = 0; i < n; ++i) ptr[i] = val;
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

    /** @brief Reinterpret as a 2D immutable matrix view. Requires rank >= 2. */
    tensor_view<value_type_, 2> as_matrix_view() const noexcept { return view().as_matrix(); }

    /** @brief Reinterpret as a 2D mutable matrix span. Requires rank >= 2. */
    tensor_span<value_type_, 2> as_matrix_span() noexcept { return span().as_matrix(); }
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

#include "numkong/reduce.hpp"
#include "numkong/each.hpp"
#include "numkong/dots.hpp"
#include "numkong/spatials.hpp"
#include "numkong/trigonometry.hpp"

namespace ashvardanian::numkong {

#pragma region - Enums and Result Types

/** @brief Controls whether reduction collapses or preserves the reduced axis. */
enum keep_dims_t : bool { collapse_dims_k = false, keep_dims_k = true };

/** @brief Result of moments(): Σxᵢ and Σxᵢ². */
template <typename sum_type_, typename sumsq_type_>
struct moments_result {
    sum_type_ sum {};
    sumsq_type_ sumsq {};
};

/** @brief Result of minmax(): min/max values with their indices. */
template <typename minmax_value_type_>
struct minmax_result {
    minmax_value_type_ min_value {};
    std::size_t min_index = 0;
    minmax_value_type_ max_value {};
    std::size_t max_index = 0;
};

#pragma endregion - Enums and Result Types

#pragma region - Helpers

/** @brief Compute output shape with one axis removed (or set to 1 if keep_dims). */
template <std::size_t max_rank_>
shape_storage_<max_rank_> reduced_shape_(shape_storage_<max_rank_> const &in, std::size_t axis, keep_dims_t keep_dims,
                                         std::size_t elem_bytes) noexcept {
    std::size_t out_extents[max_rank_];
    std::size_t out_rank = 0;
    for (std::size_t i = 0; i < in.rank; ++i) {
        if (i == axis) {
            if (keep_dims) out_extents[out_rank++] = 1;
        }
        else { out_extents[out_rank++] = in.extents[i]; }
    }
    return shape_storage_<max_rank_>::contiguous(out_extents, out_rank, elem_bytes);
}

/** @brief Validate that two views have matching shapes. */
template <typename value_type_, std::size_t max_rank_>
bool shapes_match_(tensor_view<value_type_, max_rank_> a, tensor_view<value_type_, max_rank_> b) noexcept {
    if (a.rank() != b.rank()) return false;
    for (std::size_t i = 0; i < a.rank(); ++i)
        if (a.extent(i) != b.extent(i)) return false;
    return true;
}

/** @brief Validate shape match between view and span. */
template <typename in_type_, typename out_type_, std::size_t max_rank_>
bool shapes_match_out_(tensor_view<in_type_, max_rank_> a, tensor_span<out_type_, max_rank_> out) noexcept {
    if (a.rank() != out.rank()) return false;
    for (std::size_t i = 0; i < a.rank(); ++i)
        if (a.extent(i) != out.extent(i)) return false;
    return true;
}

#pragma endregion - Helpers

#pragma region - Scalar Reductions

/** @brief Compute Σxᵢ and Σxᵢ² in a single pass. Returns zeroed result for empty tensors. */
template <typename value_type_, std::size_t max_rank_ = 8>
moments_result<typename value_type_::reduce_moments_sum_t, typename value_type_::reduce_moments_sumsq_t> moments(
    tensor_view<value_type_, max_rank_> input) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    moments_result<sum_t, sumsq_t> result {};
    if (input.empty() || input.numel() == 0) return result;
    if (input.is_contiguous()) {
        numkong::reduce_moments<value_type_>(input.data(), input.numel(), sizeof(value_type_), &result.sum,
                                             &result.sumsq);
        return result;
    }
    if (input.rank() == 1) {
        numkong::reduce_moments<value_type_>(
            input.data(), input.extent(0), static_cast<std::size_t>(input.stride_bytes(0)), &result.sum, &result.sumsq);
        return result;
    }
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        auto slice_result = moments<value_type_, max_rank_>(input.slice_leading(static_cast<std::ptrdiff_t>(i)));
        result.sum = saturating_add(result.sum, slice_result.sum);
        result.sumsq = saturating_add(result.sumsq, slice_result.sumsq);
    }
    return result;
}

/** @brief Find min and max values with their flat indices. */
template <typename value_type_, std::size_t max_rank_ = 8>
minmax_result<typename value_type_::reduce_minmax_value_t> minmax(tensor_view<value_type_, max_rank_> input) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    minmax_result<minmax_t> result {};
    if (input.empty() || input.numel() == 0) return result;
    if (input.is_contiguous()) {
        numkong::reduce_minmax<value_type_>(input.data(), input.numel(), sizeof(value_type_), &result.min_value,
                                            &result.min_index, &result.max_value, &result.max_index);
        return result;
    }
    if (input.rank() == 1) {
        numkong::reduce_minmax<value_type_>(input.data(), input.extent(0),
                                            static_cast<std::size_t>(input.stride_bytes(0)), &result.min_value,
                                            &result.min_index, &result.max_value, &result.max_index);
        return result;
    }
    result.min_value = minmax_t(value_type_::finite_max());
    result.max_value = minmax_t(value_type_::finite_min());
    std::size_t base = 0;
    for (std::size_t i = 0; i < input.extent(0); ++i) {
        auto slice = input.slice_leading(static_cast<std::ptrdiff_t>(i));
        auto slice_result = minmax<value_type_, max_rank_>(slice);
        if (slice_result.min_value < result.min_value) {
            result.min_value = slice_result.min_value;
            result.min_index = base + slice_result.min_index;
        }
        if (slice_result.max_value > result.max_value) {
            result.max_value = slice_result.max_value;
            result.max_index = base + slice_result.max_index;
        }
        base += slice.numel();
    }
    return result;
}

/** @brief Σ of all elements. */
template <typename value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_moments_sum_t sum(tensor_view<value_type_, max_rank_> input) noexcept {
    return moments(input).sum;
}

/** @brief Find the minimum element value. */
template <typename value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_minmax_value_t min(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).min_value;
}

/** @brief Find the maximum element value. */
template <typename value_type_, std::size_t max_rank_ = 8>
typename value_type_::reduce_minmax_value_t max(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).max_value;
}

/** @brief Index of the minimum element (flat). */
template <typename value_type_, std::size_t max_rank_ = 8>
std::size_t argmin(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).min_index;
}

/** @brief Index of the maximum element (flat). */
template <typename value_type_, std::size_t max_rank_ = 8>
std::size_t argmax(tensor_view<value_type_, max_rank_> input) noexcept {
    return minmax(input).max_index;
}

#pragma endregion - Scalar Reductions

#pragma region - Axis Reductions

/** @brief Σ along a single axis. Returns empty tensor on failure. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<typename value_type_::reduce_moments_sum_t, aligned_allocator<typename value_type_::reduce_moments_sum_t>,
       max_rank_>
try_sum(tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using out_tensor_t = tensor<sum_t, aligned_allocator<sum_t>, max_rank_>;
    if (input.empty() || axis >= input.rank()) return out_tensor_t {};

    auto out_shape = reduced_shape_(input.shape(), axis, keep_dims, sizeof(sum_t));
    auto result = out_tensor_t::try_zeros(out_shape.extents, out_shape.rank);
    if (result.empty()) return result;

    std::size_t reduction_len = input.extent(axis);
    auto out_span = result.span();

    if (axis == input.rank() - 1 && input.rank() >= 2) {
        auto output_data = out_span.data();
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto slice = input.slice_leading(static_cast<std::ptrdiff_t>(i));
            if (slice.rank() == 1) {
                using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
                sumsq_t discard {};
                numkong::reduce_moments<value_type_>(slice.data(), slice.extent(0),
                                                     static_cast<std::size_t>(slice.stride_bytes(0)), output_data,
                                                     &discard);
                ++output_data;
            }
            else {
                auto slice_result = try_sum<value_type_, max_rank_>(slice, axis - 1, keep_dims);
                if (!slice_result.empty()) {
                    std::size_t count = slice_result.numel();
                    auto src = slice_result.data();
                    for (std::size_t j = 0; j < count; ++j) output_data[j] = src[j];
                    output_data += count;
                }
            }
        }
    }
    else if (axis == 0) {
        std::size_t inner = input.numel() / reduction_len;
        auto output_data = out_span.data();
        for (std::size_t i = 0; i < reduction_len; ++i) {
            auto slice = input.slice_leading(static_cast<std::ptrdiff_t>(i));
            if (slice.is_contiguous() && slice.numel() == inner) {
                auto src = slice.data();
                for (std::size_t j = 0; j < inner; ++j) output_data[j] = saturating_add(output_data[j], sum_t(src[j]));
            }
            else {
                for (std::size_t j = 0; j < inner; ++j) {
                    if (slice.rank() == 1) {
                        auto *ptr = reinterpret_cast<value_type_ const *>(
                            slice.byte_data() + static_cast<std::ptrdiff_t>(j) * slice.stride_bytes(0));
                        output_data[j] = saturating_add(output_data[j], sum_t(*ptr));
                    }
                }
            }
        }
    }
    else {
        auto output_data = out_span.data();
        if (input.is_contiguous() && input.rank() == 2) {
            std::size_t rows = input.extent(0);
            std::size_t cols = input.extent(1);
            for (std::size_t r = 0; r < rows; ++r) {
                auto row = input.slice_leading(static_cast<std::ptrdiff_t>(r));
                using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
                sumsq_t discard {};
                sum_t s {};
                numkong::reduce_moments<value_type_>(row.data(), cols, static_cast<std::size_t>(row.stride_bytes(0)),
                                                     &s, &discard);
                output_data[r] = s;
            }
        }
    }

    return result;
}

/** @brief Moments along an axis (Σxᵢ and Σxᵢ² per slice). Returns pair of tensors; either may be empty on failure. */
template <typename value_type_, std::size_t max_rank_ = 8>
moments_result<tensor<typename value_type_::reduce_moments_sum_t,
                      aligned_allocator<typename value_type_::reduce_moments_sum_t>, max_rank_>,
               tensor<typename value_type_::reduce_moments_sumsq_t,
                      aligned_allocator<typename value_type_::reduce_moments_sumsq_t>, max_rank_>>
try_moments(tensor_view<value_type_, max_rank_> input, std::size_t axis,
            keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using sum_t = typename value_type_::reduce_moments_sum_t;
    using sumsq_t = typename value_type_::reduce_moments_sumsq_t;
    using sum_tensor_t = tensor<sum_t, aligned_allocator<sum_t>, max_rank_>;
    using sumsq_tensor_t = tensor<sumsq_t, aligned_allocator<sumsq_t>, max_rank_>;

    if (input.empty() || axis >= input.rank()) return {sum_tensor_t {}, sumsq_tensor_t {}};

    auto out_shape_sum = reduced_shape_(input.shape(), axis, keep_dims, sizeof(sum_t));
    auto out_shape_sq = reduced_shape_(input.shape(), axis, keep_dims, sizeof(sumsq_t));

    auto sums = sum_tensor_t::try_zeros(out_shape_sum.extents, out_shape_sum.rank);
    auto sumsqs = sumsq_tensor_t::try_zeros(out_shape_sq.extents, out_shape_sq.rank);
    if (sums.empty() || sumsqs.empty()) return {sum_tensor_t {}, sumsq_tensor_t {}};

    if (input.rank() == 2 && axis == 1) {
        auto s_out = sums.data();
        auto sq_out = sumsqs.data();
        for (std::size_t r = 0; r < input.extent(0); ++r) {
            auto row = input.slice_leading(static_cast<std::ptrdiff_t>(r));
            numkong::reduce_moments<value_type_>(row.data(), row.extent(0),
                                                 static_cast<std::size_t>(row.stride_bytes(0)), &s_out[r], &sq_out[r]);
        }
    }

    return {std::move(sums), std::move(sumsqs)};
}

/** @brief Min along an axis. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<typename value_type_::reduce_minmax_value_t, aligned_allocator<typename value_type_::reduce_minmax_value_t>,
       max_rank_>
try_min(tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    using out_tensor_t = tensor<minmax_t, aligned_allocator<minmax_t>, max_rank_>;
    if (input.empty() || axis >= input.rank()) return out_tensor_t {};

    auto out_shape = reduced_shape_(input.shape(), axis, keep_dims, sizeof(minmax_t));
    auto result = out_tensor_t::try_full(out_shape.extents, out_shape.rank, minmax_t(value_type_::finite_max()));
    if (result.empty()) return result;

    if (input.rank() == 2 && axis == 1) {
        auto output_data = result.data();
        for (std::size_t r = 0; r < input.extent(0); ++r) {
            auto row = input.slice_leading(static_cast<std::ptrdiff_t>(r));
            output_data[r] = min<value_type_, max_rank_>(row);
        }
    }
    else if (input.rank() == 2 && axis == 0) {
        auto output_data = result.data();
        for (std::size_t r = 0; r < input.extent(0); ++r) {
            auto row = input.slice_leading(static_cast<std::ptrdiff_t>(r));
            if (row.is_contiguous()) {
                auto src = row.data();
                for (std::size_t c = 0; c < input.extent(1); ++c) {
                    minmax_t v = minmax_t(src[c]);
                    if (v < output_data[c]) output_data[c] = v;
                }
            }
        }
    }
    return result;
}

/** @brief Max along an axis. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<typename value_type_::reduce_minmax_value_t, aligned_allocator<typename value_type_::reduce_minmax_value_t>,
       max_rank_>
try_max(tensor_view<value_type_, max_rank_> input, std::size_t axis, keep_dims_t keep_dims = collapse_dims_k) noexcept {
    using minmax_t = typename value_type_::reduce_minmax_value_t;
    using out_tensor_t = tensor<minmax_t, aligned_allocator<minmax_t>, max_rank_>;
    if (input.empty() || axis >= input.rank()) return out_tensor_t {};

    auto out_shape = reduced_shape_(input.shape(), axis, keep_dims, sizeof(minmax_t));
    auto result = out_tensor_t::try_full(out_shape.extents, out_shape.rank, minmax_t(value_type_::finite_min()));
    if (result.empty()) return result;

    if (input.rank() == 2 && axis == 1) {
        auto output_data = result.data();
        for (std::size_t r = 0; r < input.extent(0); ++r) {
            auto row = input.slice_leading(static_cast<std::ptrdiff_t>(r));
            output_data[r] = max<value_type_, max_rank_>(row);
        }
    }
    else if (input.rank() == 2 && axis == 0) {
        auto output_data = result.data();
        for (std::size_t r = 0; r < input.extent(0); ++r) {
            auto row = input.slice_leading(static_cast<std::ptrdiff_t>(r));
            if (row.is_contiguous()) {
                auto src = row.data();
                for (std::size_t c = 0; c < input.extent(1); ++c) {
                    minmax_t v = minmax_t(src[c]);
                    if (v > output_data[c]) output_data[c] = v;
                }
            }
        }
    }
    return result;
}

#pragma endregion - Axis Reductions

#pragma region - Elementwise Binary

/** @brief Elementwise addition: output[i] = lhs[i] + rhs[i]. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool add_into(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
              tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_(lhs, rhs) || !shapes_match_out_(lhs, output)) return false;
    if (lhs.empty()) return true;
    if (lhs.rank() >= 2) {
        for (std::size_t i = 0; i < lhs.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!add_into<value_type_, max_rank_>(lhs.slice_leading(row_index), rhs.slice_leading(row_index),
                                                  output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    numkong::sum<value_type_>(lhs.data(), rhs.data(), lhs.extent(0), output.data());
    return true;
}

/** @brief Allocating elementwise add: result = lhs + rhs. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_add(
    tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!add_into<value_type_, max_rank_>(lhs, rhs, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise add scalar: output[i] = input[i] + scalar. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool add_into(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
              tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_out_(input, output)) return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!add_into<value_type_, max_rank_>(input.slice_leading(row_index), scalar,
                                                  output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    typename value_type_::scale_t one = 1;
    numkong::scale<value_type_>(input.data(), input.extent(0), &one, &scalar, output.data());
    return true;
}

/** @brief Allocating add scalar. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_add(tensor_view<value_type_, max_rank_> input,
                                                                       typename value_type_::scale_t scalar) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!add_into<value_type_, max_rank_>(input, scalar, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise subtraction: output[i] = lhs[i] − rhs[i]. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool sub_into(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
              tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_(lhs, rhs) || !shapes_match_out_(lhs, output)) return false;
    if (lhs.empty()) return true;
    if (lhs.rank() >= 2) {
        for (std::size_t i = 0; i < lhs.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!sub_into<value_type_, max_rank_>(lhs.slice_leading(row_index), rhs.slice_leading(row_index),
                                                  output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    typename value_type_::scale_t alpha = 1;
    typename value_type_::scale_t beta = -1;
    numkong::blend<value_type_>(lhs.data(), rhs.data(), lhs.extent(0), &alpha, &beta, output.data());
    return true;
}

/** @brief Allocating elementwise sub. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_sub(
    tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!sub_into<value_type_, max_rank_>(lhs, rhs, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise sub scalar: output[i] = input[i] − scalar. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool sub_into(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
              tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_out_(input, output)) return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!sub_into<value_type_, max_rank_>(input.slice_leading(row_index), scalar,
                                                  output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    typename value_type_::scale_t one = 1;
    typename value_type_::scale_t neg_scalar = -scalar;
    numkong::scale<value_type_>(input.data(), input.extent(0), &one, &neg_scalar, output.data());
    return true;
}

/** @brief Allocating sub scalar. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_sub(tensor_view<value_type_, max_rank_> input,
                                                                       typename value_type_::scale_t scalar) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!sub_into<value_type_, max_rank_>(input, scalar, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise multiplication: output[i] = lhs[i] × rhs[i]. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool mul_into(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
              tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_(lhs, rhs) || !shapes_match_out_(lhs, output)) return false;
    if (lhs.empty()) return true;
    if (lhs.rank() >= 2) {
        for (std::size_t i = 0; i < lhs.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!mul_into<value_type_, max_rank_>(lhs.slice_leading(row_index), rhs.slice_leading(row_index),
                                                  output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    typename value_type_::scale_t alpha = 1;
    typename value_type_::scale_t beta = 0;
    numkong::fma<value_type_>(lhs.data(), rhs.data(), lhs.extent(0), output.data(), &alpha, &beta, output.data());
    return true;
}

/** @brief Allocating elementwise multiply. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_mul(
    tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_zeros(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!mul_into<value_type_, max_rank_>(lhs, rhs, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise multiply by scalar: output[i] = input[i] × scalar. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool mul_into(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t scalar,
              tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_out_(input, output)) return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!mul_into<value_type_, max_rank_>(input.slice_leading(row_index), scalar,
                                                  output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    typename value_type_::scale_t zero = 0;
    numkong::scale<value_type_>(input.data(), input.extent(0), &scalar, &zero, output.data());
    return true;
}

/** @brief Allocating multiply by scalar. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_mul(tensor_view<value_type_, max_rank_> input,
                                                                       typename value_type_::scale_t scalar) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!mul_into<value_type_, max_rank_>(input, scalar, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion - Elementwise Binary

#pragma region - Elementwise Affine

/** @brief Scale: output[i] = α × input[i] + β. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool scale_into(tensor_view<value_type_, max_rank_> input, typename value_type_::scale_t alpha,
                typename value_type_::scale_t beta, tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_out_(input, output)) return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!scale_into<value_type_, max_rank_>(input.slice_leading(row_index), alpha, beta,
                                                    output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    numkong::scale<value_type_>(input.data(), input.extent(0), &alpha, &beta, output.data());
    return true;
}

/** @brief Allocating scale: result[i] = α × input[i] + β. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_scale(tensor_view<value_type_, max_rank_> input,
                                                                         typename value_type_::scale_t alpha,
                                                                         typename value_type_::scale_t beta) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!scale_into<value_type_, max_rank_>(input, alpha, beta, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Blend: output[i] = α × lhs[i] + β × rhs[i]. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool blend_into(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
                typename value_type_::scale_t alpha, typename value_type_::scale_t beta,
                tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_(lhs, rhs) || !shapes_match_out_(lhs, output)) return false;
    if (lhs.empty()) return true;
    if (lhs.rank() >= 2) {
        for (std::size_t i = 0; i < lhs.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!blend_into<value_type_, max_rank_>(lhs.slice_leading(row_index), rhs.slice_leading(row_index), alpha,
                                                    beta, output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    numkong::blend<value_type_>(lhs.data(), rhs.data(), lhs.extent(0), &alpha, &beta, output.data());
    return true;
}

/** @brief Allocating blend: result[i] = α × lhs[i] + β × rhs[i]. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_blend(tensor_view<value_type_, max_rank_> lhs,
                                                                         tensor_view<value_type_, max_rank_> rhs,
                                                                         typename value_type_::scale_t alpha,
                                                                         typename value_type_::scale_t beta) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (!shapes_match_(lhs, rhs) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!blend_into<value_type_, max_rank_>(lhs, rhs, alpha, beta, result.span())) return out_tensor_t {};
    return result;
}

/** @brief FMA: output[i] = α × lhs[i] × rhs[i] + β × addend[i]. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool fma_into(tensor_view<value_type_, max_rank_> lhs, tensor_view<value_type_, max_rank_> rhs,
              tensor_view<value_type_, max_rank_> addend, typename value_type_::scale_t alpha,
              typename value_type_::scale_t beta, tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_(lhs, rhs) || !shapes_match_(lhs, addend) || !shapes_match_out_(lhs, output)) return false;
    if (lhs.empty()) return true;
    if (lhs.rank() >= 2) {
        for (std::size_t i = 0; i < lhs.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!fma_into<value_type_, max_rank_>(lhs.slice_leading(row_index), rhs.slice_leading(row_index),
                                                  addend.slice_leading(row_index), alpha, beta,
                                                  output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    numkong::fma<value_type_>(lhs.data(), rhs.data(), lhs.extent(0), addend.data(), &alpha, &beta, output.data());
    return true;
}

/** @brief Allocating FMA: result[i] = α × lhs[i] × rhs[i] + β × addend[i]. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_fma(tensor_view<value_type_, max_rank_> lhs,
                                                                       tensor_view<value_type_, max_rank_> rhs,
                                                                       tensor_view<value_type_, max_rank_> addend,
                                                                       typename value_type_::scale_t alpha,
                                                                       typename value_type_::scale_t beta) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (!shapes_match_(lhs, rhs) || !shapes_match_(lhs, addend) || lhs.empty()) return out_tensor_t {};
    auto &input_shape = lhs.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!fma_into<value_type_, max_rank_>(lhs, rhs, addend, alpha, beta, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion - Elementwise Affine

#pragma region - Trigonometric

/** @brief Elementwise sin into pre-allocated output. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool sin_into(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_out_(input, output)) return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!sin_into<value_type_, max_rank_>(input.slice_leading(row_index), output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    numkong::sin<value_type_>(input.data(), input.extent(0), output.data());
    return true;
}

/** @brief Allocating sin. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_sin(
    tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!sin_into<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise cos into pre-allocated output. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool cos_into(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_out_(input, output)) return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!cos_into<value_type_, max_rank_>(input.slice_leading(row_index), output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    numkong::cos<value_type_>(input.data(), input.extent(0), output.data());
    return true;
}

/** @brief Allocating cos. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_cos(
    tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!cos_into<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Elementwise atan into pre-allocated output. */
template <typename value_type_, std::size_t max_rank_ = 8>
bool atan_into(tensor_view<value_type_, max_rank_> input, tensor_span<value_type_, max_rank_> output) noexcept {
    if (!shapes_match_out_(input, output)) return false;
    if (input.empty()) return true;
    if (input.rank() >= 2) {
        for (std::size_t i = 0; i < input.extent(0); ++i) {
            auto row_index = static_cast<std::ptrdiff_t>(i);
            if (!atan_into<value_type_, max_rank_>(input.slice_leading(row_index), output.slice_leading(row_index)))
                return false;
        }
        return true;
    }
    numkong::atan<value_type_>(input.data(), input.extent(0), output.data());
    return true;
}

/** @brief Allocating atan. */
template <typename value_type_, std::size_t max_rank_ = 8>
tensor<value_type_, aligned_allocator<value_type_>, max_rank_> try_atan(
    tensor_view<value_type_, max_rank_> input) noexcept {
    using out_tensor_t = tensor<value_type_, aligned_allocator<value_type_>, max_rank_>;
    if (input.empty()) return out_tensor_t {};
    auto &input_shape = input.shape();
    auto result = out_tensor_t::try_empty(input_shape.extents, input_shape.rank);
    if (result.empty()) return result;
    if (!atan_into<value_type_, max_rank_>(input, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion - Trigonometric

#pragma region - Symmetric Distance Matrices

/** @brief Symmetric dot products: C[i,j] = ⟨A[i], A[j]⟩. */
template <typename value_type_>
bool dots_symmetric_into(matrix_view<value_type_> input,
                         matrix_span<typename value_type_::dot_result_t> output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::dots_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                         static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                         static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** @brief Allocating symmetric dot products: C = A × Aᵀ. */
template <typename value_type_>
matrix<typename value_type_::dot_result_t> try_dots_symmetric(matrix_view<value_type_> input) noexcept {
    using result_t = typename value_type_::dot_result_t;
    using out_tensor_t = matrix<result_t>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!dots_symmetric_into<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Symmetric angular distances: C[i,j] = angular(A[i], A[j]). */
template <typename value_type_>
bool angulars_symmetric_into(matrix_view<value_type_> input,
                             matrix_span<typename value_type_::angular_result_t> output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::angulars_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                             static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                             static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** @brief Allocating symmetric angular distances. */
template <typename value_type_>
matrix<typename value_type_::angular_result_t> try_angulars_symmetric(matrix_view<value_type_> input) noexcept {
    using result_t = typename value_type_::angular_result_t;
    using out_tensor_t = matrix<result_t>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!angulars_symmetric_into<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Symmetric Euclidean distances: C[i,j] = √Σ(A[i]ₖ − A[j]ₖ)². */
template <typename value_type_>
bool euclideans_symmetric_into(matrix_view<value_type_> input,
                               matrix_span<typename value_type_::euclidean_result_t> output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::euclideans_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                               static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                               static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** @brief Allocating symmetric Euclidean distances. */
template <typename value_type_>
matrix<typename value_type_::euclidean_result_t> try_euclideans_symmetric(matrix_view<value_type_> input) noexcept {
    using result_t = typename value_type_::euclidean_result_t;
    using out_tensor_t = matrix<result_t>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!euclideans_symmetric_into<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Symmetric Hamming distances: C[i,j] = hamming(A[i], A[j]). */
template <typename value_type_>
bool hammings_symmetric_into(matrix_view<value_type_> input,
                             matrix_span<typename value_type_::hamming_result_t> output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::hammings_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                             static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                             static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** @brief Allocating symmetric Hamming distances. */
template <typename value_type_>
matrix<typename value_type_::hamming_result_t> try_hammings_symmetric(matrix_view<value_type_> input) noexcept {
    using result_t = typename value_type_::hamming_result_t;
    using out_tensor_t = matrix<result_t>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!hammings_symmetric_into<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

/** @brief Symmetric Jaccard distances: C[i,j] = jaccard(A[i], A[j]). */
template <typename value_type_>
bool jaccards_symmetric_into(matrix_view<value_type_> input,
                             matrix_span<typename value_type_::jaccard_result_t> output) noexcept {
    std::size_t num_vectors = input.extent(0);
    if (output.extent(0) != num_vectors || output.extent(1) != num_vectors) return false;
    numkong::jaccards_symmetric<value_type_>(input.data(), num_vectors, input.extent(1),
                                             static_cast<std::size_t>(input.stride_bytes(0)), output.data(),
                                             static_cast<std::size_t>(output.stride_bytes(0)));
    return true;
}

/** @brief Allocating symmetric Jaccard distances. */
template <typename value_type_>
matrix<typename value_type_::jaccard_result_t> try_jaccards_symmetric(matrix_view<value_type_> input) noexcept {
    using result_t = typename value_type_::jaccard_result_t;
    using out_tensor_t = matrix<result_t>;
    if (input.empty()) return out_tensor_t {};
    std::size_t num_vectors = input.extent(0);
    auto result = out_tensor_t::try_zeros({num_vectors, num_vectors});
    if (result.empty()) return result;
    if (!jaccards_symmetric_into<value_type_>(input, result.span())) return out_tensor_t {};
    return result;
}

#pragma endregion - Symmetric Distance Matrices

} // namespace ashvardanian::numkong

#endif // NK_TENSOR_HPP
