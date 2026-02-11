/**
 *  @brief JavaScript bindings for NumKong.
 *  @file javascript/numkong.c
 *  @author Ash Vardanian
 *  @date October 18, 2023
 *
 *  @see NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <string.h>          // `strcmp` function
#include <node_api.h>        // `napi_*` functions
#include <numkong/numkong.h> // `nk_*` functions

/** @brief  Global variable that caches the CPU capabilities, and is computed just once, when the module is loaded. */
nk_capability_t static_capabilities = nk_cap_serial_k;

#pragma region Helpers

/** @brief  Parses a dtype string (e.g. "f32", "f16", "bf16") into a nk_dtype_t enum value. */
static nk_dtype_t parse_dtype_string(const char *str) {
    if (strcmp(str, "f64") == 0) return nk_f64_k;
    else if (strcmp(str, "f32") == 0) return nk_f32_k;
    else if (strcmp(str, "f16") == 0) return nk_f16_k;
    else if (strcmp(str, "bf16") == 0) return nk_bf16_k;
    else if (strcmp(str, "e4m3") == 0) return nk_e4m3_k;
    else if (strcmp(str, "e5m2") == 0) return nk_e5m2_k;
    else if (strcmp(str, "e2m3") == 0) return nk_e2m3_k;
    else if (strcmp(str, "e3m2") == 0) return nk_e3m2_k;
    else if (strcmp(str, "i8") == 0) return nk_i8_k;
    else if (strcmp(str, "u8") == 0) return nk_u8_k;
    else if (strcmp(str, "u1") == 0) return nk_u1_k;
    return nk_dtype_unknown_k;
}

/** @brief  Validates that the N-API TypedArray type is compatible with the claimed dtype. */
static int is_compatible_napi_type(napi_typedarray_type napi_type, nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return napi_type == napi_float64_array;
    case nk_f32_k: return napi_type == napi_float32_array;
    case nk_f16_k:
    case nk_bf16_k: return napi_type == napi_uint16_array;
    case nk_e4m3_k:
    case nk_e5m2_k:
    case nk_e2m3_k:
    case nk_e3m2_k:
    case nk_u8_k:
    case nk_u1_k: return napi_type == napi_uint8_array;
    case nk_i8_k: return napi_type == napi_int8_array;
    default: return 0;
    }
}

/** @brief  Returns the output dtype for a given metric kind and input dtype. */
static nk_dtype_t kernel_output_dtype(nk_kernel_kind_t kind, nk_dtype_t input) {
    switch (kind) {
    case nk_kernel_dot_k: return nk_dot_output_dtype(input);
    case nk_kernel_angular_k: return nk_angular_output_dtype(input);
    case nk_kernel_sqeuclidean_k: return nk_sqeuclidean_output_dtype(input);
    case nk_kernel_euclidean_k: return nk_euclidean_output_dtype(input);
    case nk_kernel_kld_k: return nk_probability_output_dtype(input);
    case nk_kernel_jsd_k: return nk_probability_output_dtype(input);
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Converts an nk_scalar_buffer_t result to a JavaScript number.
 *  @param  env       N-API environment.
 *  @param  result    The scalar buffer containing the result.
 *  @param  out_dtype The dtype of the value stored in the buffer.
 *  @return napi_value containing the result as a JavaScript Number, or NULL on error.
 */
static napi_value scalar_to_js_number(napi_env env, nk_scalar_buffer_t const *result, nk_dtype_t out_dtype) {
    double result_f64;
    switch (out_dtype) {
    case nk_f64_k: result_f64 = (double)result->f64; break;
    case nk_f32_k: result_f64 = (double)result->f32; break;
    case nk_f16_k: {
        nk_f32_t t;
        nk_f16_to_f32(&result->f16, &t);
        result_f64 = (double)t;
        break;
    }
    case nk_bf16_k: {
        nk_f32_t t;
        nk_bf16_to_f32(&result->bf16, &t);
        result_f64 = (double)t;
        break;
    }
    case nk_i8_k: result_f64 = (double)result->i8; break;
    case nk_u8_k: result_f64 = (double)result->u8; break;
    case nk_i32_k: result_f64 = (double)result->i32; break;
    case nk_u32_k: result_f64 = (double)result->u32; break;
    default: napi_throw_error(env, NULL, "Unexpected output dtype in result conversion"); return NULL;
    }
    napi_value js_result;
    if (napi_create_double(env, result_f64, &js_result) != napi_ok) return NULL;
    return js_result;
}

#pragma endregion Helpers

#pragma region Distance API

/** @brief  Core distance computation — resolves dtype, dispatches kernel, converts result. */
static napi_value dense(napi_env env, napi_callback_info info, nk_kernel_kind_t kernel_kind, nk_dtype_t dtype) {
    size_t argc = 3;
    napi_value args[3];
    napi_status status;

    // Get callback info and ensure the argument count is correct (2 or 3 args)
    status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (status != napi_ok || argc < 2 || argc > 3) {
        napi_throw_error(env, NULL, "Expected 2 or 3 arguments: (a, b[, dtype])");
        return NULL;
    }

    // Obtain the typed arrays from the arguments
    void *data_a, *data_b;
    size_t length_a, length_b;
    napi_typedarray_type type_a, type_b;
    napi_status status_a, status_b;
    status_a = napi_get_typedarray_info(env, args[0], &type_a, &length_a, &data_a, NULL, NULL);
    status_b = napi_get_typedarray_info(env, args[1], &type_b, &length_b, &data_b, NULL, NULL);
    if (status_a != napi_ok || status_b != napi_ok || type_a != type_b || length_a != length_b) {
        napi_throw_error(env, NULL, "Both arguments must be typed arrays of matching types and dimensionality");
        return NULL;
    }

    // When dtype is unknown, try to resolve from optional 3rd argument or auto-detect
    if (dtype == nk_dtype_unknown_k) {
        if (argc == 3) {
            // Parse explicit dtype string from 3rd argument
            char dtype_str[16];
            size_t str_len;
            if (napi_get_value_string_utf8(env, args[2], dtype_str, sizeof(dtype_str), &str_len) != napi_ok) {
                napi_throw_error(env, NULL, "Third argument must be a dtype string");
                return NULL;
            }
            dtype = parse_dtype_string(dtype_str);
            if (dtype == nk_dtype_unknown_k) {
                napi_throw_error(env, NULL, "Unsupported dtype string");
                return NULL;
            }
            if (!is_compatible_napi_type(type_a, dtype)) {
                napi_throw_error(env, NULL, "TypedArray type is not compatible with the specified dtype");
                return NULL;
            }
        }
        else {
            // Auto-detect from N-API TypedArray type (backward-compatible 4-type whitelist)
            if (type_a != napi_float64_array && type_a != napi_float32_array && type_a != napi_int8_array &&
                type_a != napi_uint8_array) {
                napi_throw_error(
                    env, NULL,
                    "Only f64, f32, i8, u8 arrays are auto-detected; pass dtype string as 3rd argument " "for other " "types");
                return NULL;
            }
            switch (type_a) {
            case napi_float64_array: dtype = nk_f64_k; break;
            case napi_float32_array: dtype = nk_f32_k; break;
            case napi_int8_array: dtype = nk_i8_k; break;
            case napi_uint8_array: dtype = nk_u8_k; break;
            default: break;
            }
        }
    }

    nk_metric_dense_punned_t metric = NULL;
    nk_capability_t capability = nk_cap_serial_k;
    nk_find_kernel_punned(kernel_kind, dtype, static_capabilities, nk_cap_any_k, (nk_kernel_punned_t *)&metric,
                          &capability);
    if (metric == NULL) {
        napi_throw_error(env, NULL, "Unsupported dtype for given metric");
        return NULL;
    }

    nk_dtype_t out_dtype = kernel_output_dtype(kernel_kind, dtype);
    if (out_dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported output dtype for given metric/input combination");
        return NULL;
    }

    nk_scalar_buffer_t result;
    metric(data_a, data_b, length_a, &result);

    return scalar_to_js_number(env, &result, out_dtype);
}

/** @brief  N-API entry for inner product (dot).  */
napi_value api_ip(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_dot_k, nk_dtype_unknown_k);
}
/** @brief  N-API entry for angular distance.  */
napi_value api_angular(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_angular_k, nk_dtype_unknown_k);
}
/** @brief  N-API entry for squared Euclidean distance.  */
napi_value api_sqeuclidean(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_sqeuclidean_k, nk_dtype_unknown_k);
}
/** @brief  N-API entry for Euclidean distance.  */
napi_value api_euclidean(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_euclidean_k, nk_dtype_unknown_k);
}
/** @brief  N-API entry for Kullback-Leibler divergence.  */
napi_value api_kld(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_kld_k, nk_dtype_unknown_k);
}
/** @brief  N-API entry for Jensen-Shannon divergence.  */
napi_value api_jsd(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_jsd_k, nk_dtype_unknown_k);
}
/** @brief  N-API entry for Hamming distance.  */
napi_value api_hamming(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_hamming_k, nk_u1_k); }
/** @brief  N-API entry for Jaccard distance.  */
napi_value api_jaccard(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_jaccard_k, nk_u1_k); }

#pragma endregion Distance API

#pragma region Capabilities API

/**
 *  @brief  Returns the runtime-detected SIMD capabilities as a bitmask.
 *  @return BigInt bitmask of nk_capability_t flags (33 flags from NEON to SME2P1)
 *
 *  This function exposes the cached capability bitmask to JavaScript users,
 *  allowing them to query what SIMD extensions are available at runtime.
 *  The capabilities are detected once at module load time and cached in static_capabilities.
 */
napi_value api_get_capabilities(napi_env env, napi_callback_info info) {
    napi_value result;
    // Use cached capabilities from module load (static_capabilities set in Init())
    napi_create_bigint_uint64(env, (uint64_t)static_capabilities, &result);
    return result;
}

#pragma endregion Capabilities API

#pragma region Cast API

/** @brief  Converts a single value from a narrow type to f32. Reads uint32 bits, returns double. */
static napi_value cast_to_f32(napi_env env, napi_callback_info info, nk_dtype_t src_dtype) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc != 1) {
        napi_throw_error(env, NULL, "Expected 1 argument");
        return NULL;
    }

    uint32_t bits;
    if (napi_get_value_uint32(env, args[0], &bits) != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a number");
        return NULL;
    }

    nk_f32_t f32_val;
    nk_cast(&bits, src_dtype, 1, &f32_val, nk_f32_k);

    napi_value result;
    napi_create_double(env, (double)f32_val, &result);
    return result;
}

/** @brief  Converts a single f32 value to a narrow type. Reads double, returns uint32 bits. */
static napi_value cast_from_f32(napi_env env, napi_callback_info info, nk_dtype_t dst_dtype) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (argc != 1) {
        napi_throw_error(env, NULL, "Expected 1 argument");
        return NULL;
    }

    double f32_dbl;
    if (napi_get_value_double(env, args[0], &f32_dbl) != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a number");
        return NULL;
    }

    nk_f32_t f32_val = (nk_f32_t)f32_dbl;
    uint32_t bits = 0;
    nk_cast(&f32_val, nk_f32_k, 1, &bits, dst_dtype);

    napi_value result;
    napi_create_uint32(env, bits, &result);
    return result;
}

/** @brief  N-API entry for scalar f16-to-f32 conversion.  */
napi_value api_cast_f16_to_f32(napi_env e, napi_callback_info i) { return cast_to_f32(e, i, nk_f16_k); }
/** @brief  N-API entry for scalar f32-to-f16 conversion.  */
napi_value api_cast_f32_to_f16(napi_env e, napi_callback_info i) { return cast_from_f32(e, i, nk_f16_k); }
/** @brief  N-API entry for scalar bf16-to-f32 conversion.  */
napi_value api_cast_bf16_to_f32(napi_env e, napi_callback_info i) { return cast_to_f32(e, i, nk_bf16_k); }
/** @brief  N-API entry for scalar f32-to-bf16 conversion.  */
napi_value api_cast_f32_to_bf16(napi_env e, napi_callback_info i) { return cast_from_f32(e, i, nk_bf16_k); }
/** @brief  N-API entry for scalar e4m3-to-f32 conversion.  */
napi_value api_cast_e4m3_to_f32(napi_env e, napi_callback_info i) { return cast_to_f32(e, i, nk_e4m3_k); }
/** @brief  N-API entry for scalar f32-to-e4m3 conversion.  */
napi_value api_cast_f32_to_e4m3(napi_env e, napi_callback_info i) { return cast_from_f32(e, i, nk_e4m3_k); }
/** @brief  N-API entry for scalar e5m2-to-f32 conversion.  */
napi_value api_cast_e5m2_to_f32(napi_env e, napi_callback_info i) { return cast_to_f32(e, i, nk_e5m2_k); }
/** @brief  N-API entry for scalar f32-to-e5m2 conversion.  */
napi_value api_cast_f32_to_e5m2(napi_env e, napi_callback_info i) { return cast_from_f32(e, i, nk_e5m2_k); }

/**
 *  @brief Buffer casting function using nk_cast.
 *  @param env N-API environment
 *  @param info Callback info containing 4 arguments:
 *              - src: source TypedArray
 *              - srcType: source dtype string
 *              - dst: destination TypedArray
 *              - dstType: destination dtype string
 *  @return null (modifies dst in place)
 */
napi_value api_cast(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value args[4];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc != 4) {
        napi_throw_error(env, NULL, "cast requires 4 arguments: (src, srcType, dst, dstType)");
        return NULL;
    }

    // Get source and destination arrays
    void *src_data, *dst_data;
    size_t src_len, dst_len;
    napi_typedarray_type src_type, dst_type;

    napi_get_typedarray_info(env, args[0], &src_type, &src_len, &src_data, NULL, NULL);
    napi_get_typedarray_info(env, args[2], &dst_type, &dst_len, &dst_data, NULL, NULL);

    // Get dtype strings
    char src_dtype_str[16], dst_dtype_str[16];
    size_t str_len;
    napi_get_value_string_utf8(env, args[1], src_dtype_str, sizeof(src_dtype_str), &str_len);
    napi_get_value_string_utf8(env, args[3], dst_dtype_str, sizeof(dst_dtype_str), &str_len);

    // Map dtype strings to nk_dtype_t
    nk_dtype_t src_dtype = parse_dtype_string(src_dtype_str);
    nk_dtype_t dst_dtype = parse_dtype_string(dst_dtype_str);

    if (src_dtype == nk_dtype_unknown_k || dst_dtype == nk_dtype_unknown_k) {
        napi_throw_error(env, NULL, "Unsupported dtype string");
        return NULL;
    }

    // Perform conversion using nk_cast
    nk_cast(src_data, src_dtype, src_len, dst_data, dst_dtype);

    return NULL; // Modifies dst_data in place
}

#pragma endregion Cast API

#pragma region Module Init

/** @brief  Registers a C function as a named JavaScript export. */
static napi_status export_function(napi_env env, napi_value exports, char const *name, napi_callback func) {
    napi_value fn;
    napi_status status = napi_create_function(env, name, NAPI_AUTO_LENGTH, func, NULL, &fn);
    if (status != napi_ok) return status;
    return napi_set_named_property(env, exports, name, fn);
}

/** @brief  Module initialization — exports all functions, detects CPU capabilities.  */
napi_value Init(napi_env env, napi_value exports) {
    if (export_function(env, exports, "dot", api_ip) != napi_ok ||
        export_function(env, exports, "inner", api_ip) != napi_ok ||
        export_function(env, exports, "sqeuclidean", api_sqeuclidean) != napi_ok ||
        export_function(env, exports, "euclidean", api_euclidean) != napi_ok ||
        export_function(env, exports, "angular", api_angular) != napi_ok ||
        export_function(env, exports, "hamming", api_hamming) != napi_ok ||
        export_function(env, exports, "jaccard", api_jaccard) != napi_ok ||
        export_function(env, exports, "kullbackleibler", api_kld) != napi_ok ||
        export_function(env, exports, "jensenshannon", api_jsd) != napi_ok ||
        export_function(env, exports, "getCapabilities", api_get_capabilities) != napi_ok ||
        export_function(env, exports, "castF16ToF32", api_cast_f16_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToF16", api_cast_f32_to_f16) != napi_ok ||
        export_function(env, exports, "castBF16ToF32", api_cast_bf16_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToBF16", api_cast_f32_to_bf16) != napi_ok ||
        export_function(env, exports, "castE4M3ToF32", api_cast_e4m3_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToE4M3", api_cast_f32_to_e4m3) != napi_ok ||
        export_function(env, exports, "castE5M2ToF32", api_cast_e5m2_to_f32) != napi_ok ||
        export_function(env, exports, "castF32ToE5M2", api_cast_f32_to_e5m2) != napi_ok ||
        export_function(env, exports, "cast", api_cast) != napi_ok) {
        return NULL;
    }
    static_capabilities = nk_capabilities();
    return exports;
}

#pragma endregion Module Init

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
