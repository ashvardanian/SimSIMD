/**
 *  @brief NumKong Tensor class for C++ 23 and newer.
 *  @file include/tensor.hpp
 *  @author Ash Vardanian
 *  @date January 7, 2026
 *
 *  Similar to `std::mdspan` in its purpose, this file provides logic for addressing
 *  and operating on high-dimensional arrays
 */

#ifndef NK_TENSOR_HPP
#define NK_TENSOR_HPP

#include <cstdlib> // `std::aligned_alloc`
#include <cstring> // `std::memcpy`
#include <utility> // `std::exchange`

#include "types.hpp"

namespace ashvardanian::numkong {

#pragma region Vector

/**
 *  @brief Cache-aligned owning vector with non-throwing allocation semantics.
 *
 *  All allocation operations return `bool` on failure instead of throwing.
 *  This is the foundational storage type for future `matrix<>` and `tensor<>`.
 *
 *  @tparam scalar_type_ The element type stored in the vector.
 */
template <typename scalar_type_>
struct vector {
    static constexpr std::size_t alignment = 64;

    using value_type = scalar_type_;
    using pointer = scalar_type_ *;
    using const_pointer = scalar_type_ const *;
    using reference = scalar_type_ &;
    using const_reference = scalar_type_ const &;
    using size_type = std::size_t;
    using iterator = pointer;
    using const_iterator = const_pointer;

  private:
    scalar_type_ *data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;

  public:
    /** @brief Default construction only - no throwing constructors */
    vector() noexcept = default;

    ~vector() noexcept {
        if (data_) std::free(data_);
    }

    /** @brief Move constructor - non-throwing */
    vector(vector &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)) {}

    /** @brief Move assignment - non-throwing */
    vector &operator=(vector &&other) noexcept {
        if (this != &other) {
            if (data_) std::free(data_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
        }
        return *this;
    }

    /** @brief Delete copy operations */
    vector(vector const &) = delete;
    vector &operator=(vector const &) = delete;

    /**
     *  @brief Resize the vector to hold `n` elements.
     *  @return `true` on success, `false` on allocation failure.
     *  @note If `n <= capacity()`, no allocation occurs.
     */
    [[nodiscard]] bool resize(std::size_t n) noexcept {
        if (n <= capacity_) {
            size_ = n;
            return true;
        }
        std::size_t bytes = align_size(n * sizeof(scalar_type_));
        void *ptr = std::aligned_alloc(alignment, bytes);
        if (!ptr) return false;
        if (data_) std::free(data_);
        data_ = static_cast<scalar_type_ *>(ptr);
        size_ = n;
        capacity_ = n;
        return true;
    }

    /**
     *  @brief Reserve capacity for at least `n` elements without changing size.
     *  @return `true` on success, `false` on allocation failure.
     *  @note Preserves existing data up to `size()`.
     */
    [[nodiscard]] bool reserve(std::size_t n) noexcept {
        if (n <= capacity_) return true;
        std::size_t bytes = align_size(n * sizeof(scalar_type_));
        void *ptr = std::aligned_alloc(alignment, bytes);
        if (!ptr) return false;
        if (data_ && size_ > 0) std::memcpy(ptr, data_, size_ * sizeof(scalar_type_));
        if (data_) std::free(data_);
        data_ = static_cast<scalar_type_ *>(ptr);
        capacity_ = n;
        return true;
    }

    /** @brief Clear the vector without deallocating */
    void clear() noexcept { size_ = 0; }

    /** @brief Release all memory and reset to empty state */
    void release() noexcept {
        if (data_) std::free(data_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    /** @brief Accessors - all noexcept */
    scalar_type_ *data() noexcept { return data_; }
    scalar_type_ const *data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t size_bytes() const noexcept { return size_ * sizeof(scalar_type_); }
    bool empty() const noexcept { return size_ == 0; }

    /** @brief Element access - no bounds checking */
    scalar_type_ &operator[](std::size_t i) noexcept { return data_[i]; }
    scalar_type_ const &operator[](std::size_t i) const noexcept { return data_[i]; }

    /** @brief Iterator support */
    scalar_type_ *begin() noexcept { return data_; }
    scalar_type_ *end() noexcept { return data_ + size_; }
    scalar_type_ const *begin() const noexcept { return data_; }
    scalar_type_ const *end() const noexcept { return data_ + size_; }
    scalar_type_ const *cbegin() const noexcept { return data_; }
    scalar_type_ const *cend() const noexcept { return data_ + size_; }

  private:
    /** @brief Round up size to alignment boundary */
    static constexpr std::size_t align_size(std::size_t bytes) noexcept {
        return ((bytes + alignment - 1) / alignment) * alignment;
    }
};

#pragma endregion Vector

} // namespace ashvardanian::numkong

#endif // NK_TENSOR_HPP
