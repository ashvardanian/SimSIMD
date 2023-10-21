/**
 *  @file       lib.c
 *  @brief      JavaScript bindings for SimSIMD.
 *  @author     Ash Vardanian
 *  @date       October 18, 2023
 *
 *  @copyright  Copyright (c) 2023
 *  @see        NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <node_api.h>        // `napi_*` functions
#include <simsimd/simsimd.h> // `simsimd_*` functions

napi_value runAPI(napi_env env, napi_callback_info info, simsimd_metric_kind_t metric_kind) {
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
    if (type_a != napi_float32_array && type_a != napi_int8_array && type_a != napi_uint8_array) {
        napi_throw_error(env, NULL, "Only `float32`, `int8` and `uint8` arrays are supported in JavaScript bindings");
        return NULL;
    }

    simsimd_datatype_t datatype = simsimd_datatype_unknown_k;
    switch (type_a) {
    case napi_float32_array: datatype = simsimd_datatype_f32_k; break;
    case napi_int8_array: datatype = simsimd_datatype_i8_k; break;
    case napi_uint8_array: datatype = simsimd_datatype_b8_k; break;
    }

    simsimd_metric_punned_t metric = simsimd_metric_punned(metric_kind, datatype, simsimd_cap_any_k);
    if (metric == NULL) {
        napi_throw_error(env, NULL, "Unsupported datatype");
        return NULL;
    }

    simsimd_f32_t result = metric(data_a, data_b, length_a, length_b);

    // Convert the result to a JavaScript number
    napi_value js_result;
    status = napi_create_double(env, result, &js_result);
    if (status != napi_ok)
        return NULL;

    return js_result;
}

napi_value l2sqAPI(napi_env env, napi_callback_info info) { return runAPI(env, info, simsimd_metric_sqeuclidean_k); }
napi_value cosAPI(napi_env env, napi_callback_info info) { return runAPI(env, info, simsimd_metric_cosine_k); }
napi_value ipAPI(napi_env env, napi_callback_info info) { return runAPI(env, info, simsimd_metric_inner_k); }
napi_value hammingAPI(napi_env env, napi_callback_info info) { return runAPI(env, info, simsimd_metric_hamming_k); }
napi_value jaccardAPI(napi_env env, napi_callback_info info) { return runAPI(env, info, simsimd_metric_jaccard_k); }

napi_value Init(napi_env env, napi_value exports) {

    // Define an array of property descriptors
    napi_property_descriptor sqeuclideanDesc = {"sqeuclidean", 0, l2sqAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor innerDesc = {"inner", 0, ipAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor cosineDesc = {"cosine", 0, cosAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor hammingDesc = {"hamming", 0, hammingAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor jaccardDesc = {"jaccard", 0, jaccardAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor properties[] = {sqeuclideanDesc, innerDesc, cosineDesc, hammingDesc, jaccardDesc};

    // Define the properties on the `exports` object
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)