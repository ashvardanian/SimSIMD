/**
 *  @brief JavaScript bindings for NumKong.
 *  @file javascript/numkong.c
 *  @author Ash Vardanian
 *  @date October 18, 2023
 *
 *  @see NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <node_api.h>        // `napi_*` functions
#include <numkong/numkong.h> // `nk_*` functions

/// @brief  Global variable that caches the CPU capabilities, and is computed just once, when the module is loaded.
nk_capability_t static_capabilities = nk_cap_serial_k;

/// @brief  Returns the output dtype for a given metric kind and input dtype.
static nk_dtype_t kernel_output_dtype(nk_kernel_kind_t kind, nk_dtype_t input) {
    switch (kind) {
    case nk_kernel_dot_k: return nk_dot_output_dtype(input);
    case nk_kernel_angular_k: return nk_angular_output_dtype(input);
    case nk_kernel_l2sq_k: return nk_l2sq_output_dtype(input);
    case nk_kernel_l2_k: return nk_l2_output_dtype(input);
    default: return nk_f64_k;
    }
}

napi_value dense(napi_env env, napi_callback_info info, nk_kernel_kind_t kernel_kind, nk_dtype_t dtype) {
    size_t argc = 2;
    napi_value args[2];
    napi_status status;

    // Get callback info and ensure the argument count is correct
    status = napi_get_cb_info(env, info, &argc, args, NULL, NULL);
    if (status != napi_ok || argc != 2) {
        napi_throw_error(env, NULL, "Wrong number of arguments");
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
    if (type_a != napi_float64_array && type_a != napi_float32_array && //
        type_a != napi_int8_array && type_a != napi_uint8_array) {
        napi_throw_error(env, NULL,
                         "Only `float64`, `float32`, `int8` and `uint8` arrays are supported in JavaScript bindings");
        return NULL;
    }

    if (dtype == nk_dtype_unknown_k) switch (type_a) {
        case napi_float64_array: dtype = nk_f64_k; break;
        case napi_float32_array: dtype = nk_f32_k; break;
        case napi_int8_array: dtype = nk_i8_k; break;
        case napi_uint8_array: dtype = nk_u8_k; break;
        default: break;
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
    nk_scalar_buffer_t result;
    metric(data_a, data_b, length_a, &result);

    // Convert the result to a JavaScript number
    napi_value js_result;
    status = napi_create_double(env, nk_scalar_buffer_get_f64(&result, out_dtype), &js_result);
    if (status != napi_ok) return NULL;

    return js_result;
}

napi_value api_ip(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_dot_k, nk_dtype_unknown_k);
}
napi_value api_angular(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_angular_k, nk_dtype_unknown_k);
}
napi_value api_l2sq(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_l2sq_k, nk_dtype_unknown_k);
}
napi_value api_l2(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_l2_k, nk_dtype_unknown_k);
}
napi_value api_kld(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_kld_k, nk_dtype_unknown_k);
}
napi_value api_jsd(napi_env env, napi_callback_info info) {
    return dense(env, info, nk_kernel_jsd_k, nk_dtype_unknown_k);
}
napi_value api_hamming(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_hamming_k, nk_u1_k); }
napi_value api_jaccard(napi_env env, napi_callback_info info) { return dense(env, info, nk_kernel_jaccard_k, nk_u1_k); }

napi_value Init(napi_env env, napi_value exports) {

    // Define an array of property descriptors
    napi_property_descriptor dot_descriptor = {"dot", 0, api_ip, 0, 0, 0, napi_default, 0};
    napi_property_descriptor inner_descriptor = {"inner", 0, api_ip, 0, 0, 0, napi_default, 0};
    napi_property_descriptor sqeuclidean_descriptor = {"sqeuclidean", 0, api_l2sq, 0, 0, 0, napi_default, 0};
    napi_property_descriptor euclidean_descriptor = {"euclidean", 0, api_l2, 0, 0, 0, napi_default, 0};
    napi_property_descriptor angular_descriptor = {"angular", 0, api_angular, 0, 0, 0, napi_default, 0};
    napi_property_descriptor hamming_descriptor = {"hamming", 0, api_hamming, 0, 0, 0, napi_default, 0};
    napi_property_descriptor jaccard_descriptor = {"jaccard", 0, api_jaccard, 0, 0, 0, napi_default, 0};
    napi_property_descriptor kld_descriptor = {"kullbackleibler", 0, api_kld, 0, 0, 0, napi_default, 0};
    napi_property_descriptor jsd_descriptor = {"jensenshannon", 0, api_jsd, 0, 0, 0, napi_default, 0};
    napi_property_descriptor properties[] = {
        dot_descriptor,     inner_descriptor,   sqeuclidean_descriptor, euclidean_descriptor, angular_descriptor,
        hamming_descriptor, jaccard_descriptor, kld_descriptor,         jsd_descriptor,
    };

    // Define the properties on the `exports` object
    size_t property_count = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, property_count, properties);

    static_capabilities = nk_capabilities();
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
