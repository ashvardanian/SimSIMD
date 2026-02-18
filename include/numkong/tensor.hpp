/**
 *  @brief NumKong Tensor class for C++23 and newer.
 *  @file include/numkong/tensor.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2026
 *
 *  Similar to `std::mdspan` in its purpose, this file provides logic for addressing
 *  and operating on high-dimensional arrays.
 *
 *  @section vector_terminology Terminology: Dimensions vs Values
 *
 *  The `nk::vector<value_type_>` container uses two distinct counts:
 *
 *  - `size()`: Number of logical dimensions (what kernels see).
 *    For `vector<i4x2_t>` with 100 dimensions, you have 100 nibbles.
 *
 *  - `size_values()`: Number of C++ container elements (`value_type_` instances).
 *    For `vector<i4x2_t>` with 100 dimensions, you have 50 values (2 dims/value).
 *
 *  @code
 *  nk::vector<nk::i4x2_t> v;
 *  v.resize(100);           // 100 dimensions (nibbles)
 *  v.size();                // 100
 *  v.size_values();         // 50 (each i4x2_t holds 2 nibbles)
 *  v[42];                   // Access dimension 42 via proxy reference
 *  @endcode
 *
 *  See `types.hpp` for the full terminology reference.
 */

#ifndef NK_TENSOR_HPP
#define NK_TENSOR_HPP

#include <cstdlib>     // `std::aligned_alloc`, `std::free`
#include <cstring>     // `std::memcpy`
#include <iterator>    // `std::random_access_iterator_tag`
#include <memory>      // `std::allocator_traits`
#include <stdexcept>   // `std::out_of_range`
#include <type_traits> // `std::conditional_t`
#include <utility>     // `std::exchange`, `std::swap`

#include "types.hpp"

namespace ashvardanian::numkong {

#pragma region - Aligned Allocator

/**
 *  @brief Cache-aligned allocator with non-throwing allocation.
 *  @tparam value_type_ Value type to allocate.
 *  @tparam alignment_ Alignment in bytes (default: 64 for cache line).
 *
 *  This allocator uses `std::aligned_alloc` and returns `nullptr` on failure
 *  instead of throwing. It is stateless and always compares equal.
 */
template <typename value_type_, std::size_t alignment_ = 64>
struct aligned_allocator {
    using value_type = value_type_;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    static constexpr std::size_t alignment = alignment_;

    constexpr aligned_allocator() noexcept = default;

    template <typename other_type_>
    constexpr aligned_allocator(aligned_allocator<other_type_, alignment_> const &) noexcept {}

    [[nodiscard]] value_type *allocate(std::size_t n) noexcept {
        if (n == 0) return nullptr;
        std::size_t bytes = n * sizeof(value_type);
        // Round up to alignment boundary (required by aligned_alloc)
        std::size_t aligned_bytes = ((bytes + alignment_ - 1) / alignment_) * alignment_;
        return static_cast<value_type *>(std::aligned_alloc(alignment_, aligned_bytes));
    }

    void deallocate(value_type *p, std::size_t) noexcept {
        if (p) std::free(p);
    }

    template <typename other_type_>
    constexpr bool operator==(aligned_allocator<other_type_, alignment_> const &) const noexcept {
        return true;
    }
};

#pragma endregion - Aligned Allocator

#pragma region - Vector

template <typename value_type_, typename allocator_type_>
struct vector;

/**
 *  @brief Random-access iterator over logical dimensions.
 *
 *  For sub-byte types (i4x2_t, u1x8_t), dereference returns a proxy reference.
 *  For normal types, dereference returns a direct reference.
 */
template <typename container_type_>
class dimension_iterator {
    static constexpr bool is_const_ = std::is_const<container_type_>::value;
    using container_t = typename std::remove_reference<container_type_>::type;
    using container_ptr_t = std::conditional_t<is_const_, container_t const *, container_t *>;

    container_ptr_t container_;
    std::size_t index_;

  public:
    using container_type = container_type_;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using size_type = std::size_t;

    constexpr dimension_iterator() noexcept : container_(nullptr), index_(0) {}
    constexpr dimension_iterator(container_type &c, size_type i) noexcept : container_(&c), index_(i) {}

    constexpr decltype(auto) operator*() const noexcept { return (*container_)[index_]; }

    constexpr auto operator->() const noexcept {
        if constexpr (is_const_) return &(*container_)[index_];
        else return &(*container_)[index_];
    }

    constexpr decltype(auto) operator[](difference_type n) const noexcept { return (*container_)[index_ + n]; }

    constexpr dimension_iterator &operator++() noexcept {
        ++index_;
        return *this;
    }
    constexpr dimension_iterator operator++(int) noexcept {
        auto tmp = *this;
        ++index_;
        return tmp;
    }
    constexpr dimension_iterator &operator--() noexcept {
        --index_;
        return *this;
    }
    constexpr dimension_iterator operator--(int) noexcept {
        auto tmp = *this;
        --index_;
        return tmp;
    }

    constexpr dimension_iterator &operator+=(difference_type n) noexcept {
        index_ += n;
        return *this;
    }
    constexpr dimension_iterator &operator-=(difference_type n) noexcept {
        index_ -= n;
        return *this;
    }
    constexpr dimension_iterator operator+(difference_type n) const noexcept {
        return {*container_, index_ + static_cast<size_type>(n)};
    }
    constexpr dimension_iterator operator-(difference_type n) const noexcept {
        return {*container_, index_ - static_cast<size_type>(n)};
    }
    constexpr difference_type operator-(dimension_iterator const &other) const noexcept {
        return static_cast<difference_type>(index_) - static_cast<difference_type>(other.index_);
    }

    constexpr bool operator==(dimension_iterator const &other) const noexcept { return index_ == other.index_; }
    constexpr bool operator!=(dimension_iterator const &other) const noexcept { return index_ != other.index_; }
    constexpr bool operator<(dimension_iterator const &other) const noexcept { return index_ < other.index_; }
    constexpr bool operator<=(dimension_iterator const &other) const noexcept { return index_ <= other.index_; }
    constexpr bool operator>(dimension_iterator const &other) const noexcept { return index_ > other.index_; }
    constexpr bool operator>=(dimension_iterator const &other) const noexcept { return index_ >= other.index_; }

    friend constexpr dimension_iterator operator+(difference_type n, dimension_iterator const &it) noexcept {
        return it + n;
    }
};

/**
 *  @brief Cache-aligned owning vector with non-throwing allocation semantics.
 *
 *  All allocation operations return `bool` on failure instead of throwing.
 *  Supports custom allocators including stateful ones with proper propagation.
 *
 *  @tparam value_type_ The element type stored in the vector.
 *  @tparam allocator_type_ Allocator type (default: aligned_allocator<value_type_>).
 *
 *  @section vector_primary Primary API (Dimension-based)
 *  - `resize(d)` - Resize to `d` dimensions
 *  - `reserve(d)` - Reserve capacity for `d` dimensions
 *  - `size()` - Number of dimensions
 *  - `capacity()` - Capacity in dimensions
 *  - `at(i)` - Dimension access with bounds checking
 *  - `operator[](i)` - Dimension access (proxy for sub-byte)
 *  - `begin()`, `end()` - Dimension iterators
 *  - `data()` - Raw pointer for C API calls
 *
 *  @section vector_secondary Secondary API (Value-based)
 *  - `resize_values(n)` - Resize to `n` values
 *  - `reserve_values(n)` - Reserve capacity for `n` values
 *  - `size_values()` - Number of values
 *  - `capacity_values()` - Capacity in values
 *  - `at_value(i)` - Value access with bounds checking
 *  - `value(i)` - Value access without bounds checking
 *  - `begin_values()`, `end_values()` - Value iterators
 *  - `data_values()` - Pointer to value array
 *
 *  @section vector_subbyte Sub-byte Dimension Ordering
 *  For packed types like `i4x2_t` and `u1x8_t`, dimensions are stored LSB-first:
 *  - `i4x2_t`: dimension 0 is low nibble (bits 0-3), dimension 1 is high nibble (bits 4-7)
 *  - `u1x8_t`: dimension 0 is bit 0, dimension 7 is bit 7
 */
template <typename value_type_, typename allocator_type_ = aligned_allocator<value_type_>>
struct vector {
    using value_type = value_type_;
    using raw_value_type = typename raw_pod_type<value_type>::type;

    using allocator_type = allocator_type_;
    using alloc_traits = std::allocator_traits<allocator_type_>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type_ *;
    using const_pointer = value_type_ const *;
    using reference = value_type_ &;
    using const_reference = value_type_ const &;

    using iterator = dimension_iterator<vector>;
    using const_iterator = dimension_iterator<vector const>;

    using value_iterator = pointer;
    using const_value_iterator = const_pointer;

  private:
    pointer data_ = nullptr;
    size_type dimensions_ = 0;
    size_type capacity_values_ = 0;
    [[no_unique_address]] allocator_type_ alloc_;

    /** @brief Convert dimension count to value count. */
    static constexpr size_type dims_to_values(size_type dims) noexcept {
        return divide_round_up(dims, dimensions_per_value<value_type>());
    }

    /** @brief Convert value count to dimension count. */
    static constexpr size_type values_to_dims(size_type values) noexcept {
        return values * dimensions_per_value<value_type>();
    }

  public:
    /** @brief Default constructor - empty vector with default allocator. */
    vector() noexcept = default;

    /** @brief Construct with custom allocator. */
    explicit vector(allocator_type_ const &alloc) noexcept : alloc_(alloc) {}

    /** @brief Destructor - deallocates memory. */
    ~vector() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, capacity_values_);
    }

    /** @brief Move constructor. */
    vector(vector &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), dimensions_(std::exchange(other.dimensions_, 0)),
          capacity_values_(std::exchange(other.capacity_values_, 0)), alloc_(std::move(other.alloc_)) {}

    /** @brief Move assignment with allocator propagation. */
    vector &operator=(vector &&other) noexcept {
        if (this != &other) {
            if (data_) alloc_traits::deallocate(alloc_, data_, capacity_values_);
            if constexpr (alloc_traits::propagate_on_container_move_assignment::value) alloc_ = std::move(other.alloc_);

            data_ = std::exchange(other.data_, nullptr);
            dimensions_ = std::exchange(other.dimensions_, 0);
            capacity_values_ = std::exchange(other.capacity_values_, 0);
        }
        return *this;
    }

    /** @brief Copy constructor - deleted. */
    vector(vector const &) = delete;
    /** @brief Copy assignment - deleted. */
    vector &operator=(vector const &) = delete;

    /** @brief Swap with another vector. */
    void swap(vector &other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(dimensions_, other.dimensions_);
        swap(capacity_values_, other.capacity_values_);
        if constexpr (alloc_traits::propagate_on_container_swap::value) swap(alloc_, other.alloc_);
    }

    /**
     *  @brief Resize to hold `d` dimensions.
     *  @return `true` on success, `false` on allocation failure.
     */
    [[nodiscard]] bool resize(size_type d) noexcept {
        size_type values_needed = dims_to_values(d);
        if (values_needed <= capacity_values_) {
            dimensions_ = d;
            return true;
        }
        pointer ptr = alloc_traits::allocate(alloc_, values_needed);
        if (!ptr) return false;
        if (data_) alloc_traits::deallocate(alloc_, data_, capacity_values_);
        data_ = ptr;
        dimensions_ = d;
        capacity_values_ = values_needed;
        return true;
    }

    /**
     *  @brief Reserve capacity for at least `d` dimensions.
     *  @return `true` on success, `false` on allocation failure.
     */
    [[nodiscard]] bool reserve(size_type d) noexcept {
        size_type values_needed = dims_to_values(d);
        if (values_needed <= capacity_values_) return true;
        pointer ptr = alloc_traits::allocate(alloc_, values_needed);
        if (!ptr) return false;
        size_type current_values = dims_to_values(dimensions_);
        if (data_ && current_values > 0) std::memcpy(ptr, data_, current_values * sizeof(value_type_));
        if (data_) alloc_traits::deallocate(alloc_, data_, capacity_values_);
        data_ = ptr;
        capacity_values_ = values_needed;
        return true;
    }

    /** @brief Number of dimensions. */
    size_type size() const noexcept { return dimensions_; }

    /** @brief Capacity in dimensions. */
    size_type capacity() const noexcept { return values_to_dims(capacity_values_); }

    /** @brief Check if empty. */
    bool empty() const noexcept { return dimensions_ == 0; }

    /** @brief Clear without deallocating. */
    void clear() noexcept { dimensions_ = 0; }

    /** @brief Release all memory. */
    void reset() noexcept {
        if (data_) alloc_traits::deallocate(alloc_, data_, capacity_values_);
        data_ = nullptr;
        dimensions_ = 0;
        capacity_values_ = 0;
    }

    /**
     *  @brief Dimension access without bounds checking.
     *  @retval For sub-byte types, returns proxy reference.
     *  @retval For normal types, returns direct reference.
     */
    decltype(auto) operator[](size_type i) noexcept {
        if constexpr (dimensions_per_value<value_type>() > 1)
            return sub_byte_ref<value_type>(reinterpret_cast<raw_value_type *>(data_), i);
        else return data_[i];
    }

    decltype(auto) operator[](size_type i) const noexcept {
        if constexpr (dimensions_per_value<value_type>() > 1)
            return sub_byte_ref<value_type>(reinterpret_cast<raw_value_type *>(data_), i).get();
        else return data_[i];
    }

    /** @brief Dimension iterator to beginning. */
    iterator begin() noexcept { return {*this, 0}; }
    const_iterator begin() const noexcept { return {*this, 0}; }
    const_iterator cbegin() const noexcept { return {*this, 0}; }

    /** @brief Dimension iterator to end. */
    iterator end() noexcept { return {*this, dimensions_}; }
    const_iterator end() const noexcept { return {*this, dimensions_}; }
    const_iterator cend() const noexcept { return {*this, dimensions_}; }

    /**
     *  @brief Resize to hold `n` values.
     *  @return `true` on success, `false` on allocation failure.
     */
    [[nodiscard]] bool resize_values(size_type n) noexcept { return resize(values_to_dims(n)); }

    /**
     *  @brief Reserve capacity for at least `n` values.
     *  @return `true` on success, `false` on allocation failure.
     */
    [[nodiscard]] bool reserve_values(size_type n) noexcept {
        if (n <= capacity_values_) return true;
        pointer ptr = alloc_traits::allocate(alloc_, n);
        if (!ptr) return false;
        size_type current_values = dims_to_values(dimensions_);
        if (data_ && current_values > 0) std::memcpy(ptr, data_, current_values * sizeof(value_type_));
        if (data_) alloc_traits::deallocate(alloc_, data_, capacity_values_);
        data_ = ptr;
        capacity_values_ = n;
        return true;
    }

    /** @brief Number of values. */
    size_type size_values() const noexcept { return dims_to_values(dimensions_); }

    /** @brief Capacity in values. */
    size_type capacity_values() const noexcept { return capacity_values_; }

    value_type *values_data() noexcept { return data_; }

    value_type const *values_data() const noexcept { return data_; }

    raw_value_type *raw_values_data() noexcept { return reinterpret_cast<raw_value_type *>(data_); }

    raw_value_type const *raw_values_data() const noexcept { return reinterpret_cast<raw_value_type const *>(data_); }

    /** @brief Size in bytes. */
    size_type size_bytes() const noexcept { return dims_to_values(dimensions_) * sizeof(value_type_); }

    /** @brief Get a copy of the allocator. */
    allocator_type get_allocator() const noexcept { return alloc_; }
};

/** @brief Non-member swap. */
template <typename value_type_, typename allocator_type_>
void swap(vector<value_type_, allocator_type_> &a, vector<value_type_, allocator_type_> &b) noexcept {
    a.swap(b);
}

#pragma endregion - Vector

} // namespace ashvardanian::numkong

#endif // NK_TENSOR_HPP
