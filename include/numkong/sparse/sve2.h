/**
 *  @brief SVE2-accelerated Sparse Vector Operations.
 *  @file include/numkong/sparse/sve2.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/sparse.h
 */
#ifndef NK_SPARSE_SVE2_H
#define NK_SPARSE_SVE2_H

#if NK_TARGET_ARM_

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/*  SVE2 introduces many new integer-oriented instructions, extending some of the NEON functionality
 *  to variable-length SVE registers. Those include "compare multiple" intrinsics:
 *
 *  - `svmatch[_u16]` that matches each scalar in first vector against all members of a 128-bit lane in the second.
 *  - `svhistcnt[_s32]_z` does something similar, performing an inclusive prefix scan.
 *  - `svtbx[_u16]` does extended table lookup
 *
 *  Other notable instructions:
 *
 *  - `DUP`: Broadcast indexed predicate element
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/DUP--predicate---Broadcast-indexed-predicate-element-?lang=en
 *  - `SCLAMP` and `UCLAMP`: clamp values, i.e. combined min+max
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/SCLAMP--Signed-clamp-to-minimum-maximum-vector-?lang=en
 *    https://developer.arm.com/documentation/ddi0602/2021-06/SVE-Instructions/UCLAMP--Unsigned-clamp-to-minimum-maximum-vector-?lang=en
 *  - `TBLQ`: Table lookup quadword
 *    https://developer.arm.com/documentation/ddi0602/2022-12/SVE-Instructions/TBLQ--Programmable-table-lookup-within-each-quadword-vector-segment--zeroing--?lang=en
 *
 *  Great resources for SVE2 intrinsics:
 *
 *  > ARM's Scalable Vector Extensions: A Critical Look at SVE2 For Integer Workloads
 *    https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd
 */
#if NK_TARGET_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+sve2")
#endif

NK_PUBLIC void nk_sparse_intersect_u16_sve2( //
    nk_u16_t const *a, nk_u16_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u16_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 8 values.
    nk_size_t const register_size = svcnth();
    nk_size_t const lanes_count = register_size / 8;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b16_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b16_u64(b_idx, b_length);
        svuint16_t a_vec = svld1_u16(a_progress, a + a_idx);
        svuint16_t b_vec = svld1_u16(b_progress, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = svlastb(a_progress, a_vec);
        nk_u16_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u16_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b16_u64(a_idx, a_length);
            a_vec = svld1_u16(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b16_u64(b_idx, b_length);
            b_vec = svld1_u16(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Before we evaluate the intersection size, obfurscating the order in `b_vec`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint16_t a_last_broadcasted =  svdup_n_u16(a_max);
        //      svuint16_t b_last_broadcasted =  svdup_n_u16(b_max);
        svbool_t a_mask = svcmple_n_u16(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u16(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b16(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b16(b_progress, b_mask);

        // Compare `a_vec` with each lane of `b_vec`
        svbool_t equal_mask = svmatch_u16(a_progress, a_vec, b_vec);
        for (nk_size_t i = 1; i < lanes_count; i++) {
            b_vec = svext_u16(b_vec, b_vec, 8);
            equal_mask = svorr_z(svptrue_b16(), equal_mask, svmatch_u16(a_progress, a_vec, b_vec));
        }
        nk_size_t equal_count = svcntp_b16(svptrue_b16(), equal_mask);

        // Manually compact and store matching elements (svcompact_u16 is not defined)
        if (result) {
            nk_u16_t a_data[16];
            nk_u16_t mask_data[16];

            svst1_u16(svptrue_b16(), a_data, a_vec);
            svst1_u16(svptrue_b16(), mask_data, svdup_n_u16_z(equal_mask, 1));

            for (nk_size_t i = 0; i < svcnth(); i++)
                if (mask_data[i]) result[c++] = a_data[i];
            c -= equal_count;
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_intersect_u32_sve2( //
    nk_u32_t const *a, nk_u32_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u32_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 4 values.
    nk_size_t const register_size = svcntw();
    nk_size_t const lanes_count = register_size / 4;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b32_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b32_u64(b_idx, b_length);
        svuint32_t a_vec = svld1_u32(a_progress, a + a_idx);
        svuint32_t b_vec = svld1_u32(b_progress, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u32_t a_min;
        nk_u32_t a_max = svlastb(a_progress, a_vec);
        nk_u32_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u32_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b32_u64(a_idx, a_length);
            a_vec = svld1_u32(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b32_u64(b_idx, b_length);
            b_vec = svld1_u32(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Before we evaluate the intersection size, obfurscating the order in `b_vec`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint32_t a_last_broadcasted =  svdup_n_u32(a_max);
        //      svuint32_t b_last_broadcasted =  svdup_n_u32(b_max);
        svbool_t a_mask = svcmple_n_u32(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u32(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b32(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b32(b_progress, b_mask);

        // Comparing `a_vec` with each lane of `b_vec` can't be done with `svmatch`,
        // the same way as in `nk_sparse_intersect_u16_sve2`, as that instruction is only
        // available for 8-bit and 16-bit integers.
        //
        //      svbool_t equal_mask = svpfalse_b();
        //      for (nk_size_t i = 0; i < register_size; i++) {
        //          equal_mask = svorr_z(svptrue_b32(), equal_mask, svcmpeq_u32(a_progress, a_vec, b_vec));
        //          b_vec = svext_u32(b_vec, b_vec, 1);
        //      }
        //      nk_size_t equal_count = svcntp_b32(a_progress, equal_mask);
        //
        // Alternatively, one can use histogram instructions, like `svhistcnt_u32_z`.
        // They practically compute the prefix-matching count, which is equivalent to
        // the lower triangle of the row-major intersection matrix.
        // To compute the upper triangle, we can reverse (with `svrev_b32`) the order of
        // elements and repeat the operation, accumulating the results for top and bottom.
        // Let's look at 4x element registers as an example:
        //
        //      ⊐ α = {A, B, C, D}, β = {X, Y, Z, W}:
        //
        //      hist(α, β):           hist(α_rev, β_rev):
        //
        //        X Y Z W               W Z Y X
        //      A 1 0 0 0             D 1 0 0 0
        //      B 1 1 0 0             C 1 1 0 0
        //      C 1 1 1 0             B 1 1 1 0
        //      D 1 1 1 1             A 1 1 1 1
        //
        svuint32_t hist_lower = svhistcnt_u32_z(a_progress, a_vec, b_vec);
        svuint32_t a_rev_vec = svrev_u32(a_vec);
        svuint32_t b_rev_vec = svrev_u32(b_vec);
        svuint32_t hist_upper = svrev_u32(svhistcnt_u32_z(svptrue_b32(), a_rev_vec, b_rev_vec));
        svuint32_t hist = svorr_u32_x(a_progress, hist_lower, hist_upper);
        svbool_t equal_mask = svcmpne_n_u32(a_progress, hist, 0);
        nk_size_t equal_count = svcntp_b32(a_progress, equal_mask);

        // Use SVE2 svcompact to compress matching elements and store to result buffer
        if (result) {
            svuint32_t compacted = svcompact_u32(equal_mask, a_vec);
            svbool_t store_predicate = svwhilelt_b32_u64(0, equal_count);
            svst1_u32(store_predicate, result + c, compacted);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_intersect_u64_sve2( //
    nk_u64_t const *a, nk_u64_t const *b,    //
    nk_size_t a_length, nk_size_t b_length,  //
    nk_u64_t *result, nk_size_t *count) {

    // A single SVE lane is 128 bits wide, so one lane fits 2 values.
    nk_size_t const register_size = svcntd();
    nk_size_t const lanes_count = register_size / 2;
    nk_size_t a_idx = 0, b_idx = 0;
    nk_size_t c = 0;

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b64_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b64_u64(b_idx, b_length);
        svuint64_t a_vec = svld1_u64(a_progress, a + a_idx);
        svuint64_t b_vec = svld1_u64(b_progress, b + b_idx);

        // Intersecting registers involves comparisons,
        // so we want to avoid it if the slices don't overlap at all.
        nk_u64_t a_min;
        nk_u64_t a_max = svlastb(a_progress, a_vec);
        nk_u64_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u64_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b64_u64(a_idx, a_length);
            a_vec = svld1_u64(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b64_u64(b_idx, b_length);
            b_vec = svld1_u64(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Estimate how much we will need to advance the pointers afterwards.
        svbool_t a_mask = svcmple_n_u64(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u64(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b64(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b64(b_progress, b_mask);

        // Use histogram instructions like `svhistcnt_u64_z` to compute intersection.
        // They compute the prefix-matching count, equivalent to the lower triangle
        // of the row-major intersection matrix.
        svuint64_t hist_lower = svhistcnt_u64_z(a_progress, a_vec, b_vec);
        svuint64_t a_rev_vec = svrev_u64(a_vec);
        svuint64_t b_rev_vec = svrev_u64(b_vec);
        svuint64_t hist_upper = svrev_u64(svhistcnt_u64_z(svptrue_b64(), a_rev_vec, b_rev_vec));
        svuint64_t hist = svorr_u64_x(a_progress, hist_lower, hist_upper);
        svbool_t equal_mask = svcmpne_n_u64(a_progress, hist, 0);
        nk_size_t equal_count = svcntp_b64(a_progress, equal_mask);

        // Use SVE2 svcompact to compress matching elements and store to result buffer
        if (result) {
            svuint64_t compacted = svcompact_u64(equal_mask, a_vec);
            svbool_t store_predicate = svwhilelt_b64_u64(0, equal_count);
            svst1_u64(store_predicate, result + c, compacted);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
        c += equal_count;
    }
    *count = c;
}

NK_PUBLIC void nk_sparse_dot_u32f32_sve2(                 //
    nk_u32_t const *a, nk_u32_t const *b,                 //
    nk_f32_t const *a_weights, nk_f32_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,               //
    nk_f32_t *product) {

    // A single SVE lane is 128 bits wide, so one lane fits 4 values.
    nk_size_t const register_size = svcntw();
    nk_size_t const lanes_count = register_size / 4;
    nk_size_t a_idx = 0, b_idx = 0;
    svfloat32_t product_f32_sve = svdup_f32(0.f);

    while (a_idx < a_length && b_idx < b_length) {
        // Load indices with progress predicates
        svbool_t a_progress = svwhilelt_b32_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b32_u64(b_idx, b_length);
        svuint32_t a_u32_sve = svld1_u32(a_progress, a + a_idx);
        svuint32_t b_u32_sve = svld1_u32(b_progress, b + b_idx);

        // Avoid expensive intersection if slices don't overlap at all
        nk_u32_t a_min;
        nk_u32_t a_max = svlastb(a_progress, a_u32_sve);
        nk_u32_t b_min = svlasta(svpfalse_b(), b_u32_sve);
        nk_u32_t b_max = svlastb(b_progress, b_u32_sve);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b32_u64(a_idx, a_length);
            a_u32_sve = svld1_u32(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_u32_sve);
        }
        a_min = svlasta(svpfalse_b(), a_u32_sve);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b32_u64(b_idx, b_length);
            b_u32_sve = svld1_u32(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_u32_sve);
        }
        b_min = svlasta(svpfalse_b(), b_u32_sve);

        // Calculate step sizes before modifying vectors
        svbool_t a_mask = svcmple_n_u32(a_progress, a_u32_sve, b_max);
        svbool_t b_mask = svcmple_n_u32(b_progress, b_u32_sve, a_max);
        nk_u64_t a_step = svcntp_b32(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b32(b_progress, b_mask);

        // Use histogram-based intersection (svmatch_u32 doesn't exist)
        svuint32_t hist_lower = svhistcnt_u32_z(a_progress, a_u32_sve, b_u32_sve);
        svuint32_t a_rev_u32_sve = svrev_u32(a_u32_sve);
        svuint32_t b_rev_u32_sve = svrev_u32(b_u32_sve);
        svuint32_t hist_upper = svrev_u32(svhistcnt_u32_z(svptrue_b32(), a_rev_u32_sve, b_rev_u32_sve));
        svuint32_t hist = svorr_u32_x(a_progress, hist_lower, hist_upper);
        svbool_t a_equal_mask = svcmpne_n_u32(a_progress, hist, 0);

        // Load weights and mask by intersection
        svfloat32_t a_weights_f32_sve = svld1_f32(a_progress, a_weights + a_idx);
        svfloat32_t b_weights_f32_sve = svld1_f32(b_progress, b_weights + b_idx);

        // For each position in a that matches something in b, we need the corresponding b weight.
        // Use lane-by-lane matching for dot product.
        for (nk_size_t i = 0; i < lanes_count; i++) {
            // Check which elements of a match the current rotation of b
            svbool_t equal_lane = svcmpeq_u32(a_progress, a_u32_sve, b_u32_sve);
            // Multiply matching weights and accumulate
            svfloat32_t b_equal_weights = svsel_f32(equal_lane, b_weights_f32_sve, svdup_f32(0.f));
            product_f32_sve = svmla_f32_x(a_progress, product_f32_sve, a_weights_f32_sve, b_equal_weights);
            // Rotate b vectors
            b_u32_sve = svext_u32(b_u32_sve, b_u32_sve, 4);
            b_weights_f32_sve = svext_f32(b_weights_f32_sve, b_weights_f32_sve, 4);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
    }
    *product = svaddv_f32(svptrue_b32(), product_f32_sve);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVE2

#if NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+sve+sve2+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+sve+sve2+bf16")
#endif

NK_PUBLIC void nk_sparse_dot_u16bf16_sve2(                  //
    nk_u16_t const *a, nk_u16_t const *b,                   //
    nk_bf16_t const *a_weights, nk_bf16_t const *b_weights, //
    nk_size_t a_length, nk_size_t b_length,                 //
    nk_f32_t *product) {

    // A single SVE lane is 128 bits wide, so one lane fits 8 values.
    nk_size_t const register_size = svcnth();
    nk_size_t const lanes_count = register_size / 8;
    nk_size_t a_idx = 0, b_idx = 0;
    svfloat32_t product_vec = svdupq_n_f32(0.f, 0.f, 0.f, 0.f);

    while (a_idx < a_length && b_idx < b_length) {
        // Load `a_member` and broadcast it, load `b_members_vec` from memory
        svbool_t a_progress = svwhilelt_b16_u64(a_idx, a_length);
        svbool_t b_progress = svwhilelt_b16_u64(b_idx, b_length);
        svuint16_t a_vec = svld1_u16(a_progress, a + a_idx);
        svuint16_t b_vec = svld1_u16(b_progress, b + b_idx);

        // Intersecting registers with `svmatch_u16` involves a lot of shuffling
        // and comparisons, so we want to avoid it if the slices don't overlap at all..
        nk_u16_t a_min;
        nk_u16_t a_max = svlastb(a_progress, a_vec);
        nk_u16_t b_min = svlasta(svpfalse_b(), b_vec);
        nk_u16_t b_max = svlastb(b_progress, b_vec);

        // If the slices don't overlap, advance the appropriate pointer
        while (a_max < b_min && (a_idx + register_size) <= a_length) {
            a_idx += register_size;
            a_progress = svwhilelt_b16_u64(a_idx, a_length);
            a_vec = svld1_u16(a_progress, a + a_idx);
            a_max = svlastb(a_progress, a_vec);
        }
        a_min = svlasta(svpfalse_b(), a_vec);
        while (b_max < a_min && (b_idx + register_size) <= b_length) {
            b_idx += register_size;
            b_progress = svwhilelt_b16_u64(b_idx, b_length);
            b_vec = svld1_u16(b_progress, b + b_idx);
            b_max = svlastb(b_progress, b_vec);
        }
        b_min = svlasta(svpfalse_b(), b_vec);

        // Before we evaluate the intersection size, obfurscating the order in `b_vec`,
        // let's estimate how much we will need to advance the pointers afterwards.
        // For that, we don't even need to broadcast the values in SVE, as the whole
        // register can be compared against a scalar:
        //
        //      svuint16_t a_last_broadcasted =  svdup_n_u16(a_max);
        //      svuint16_t b_last_broadcasted =  svdup_n_u16(b_max);
        svbool_t a_mask = svcmple_n_u16(a_progress, a_vec, b_max);
        svbool_t b_mask = svcmple_n_u16(b_progress, b_vec, a_max);
        nk_u64_t a_step = svcntp_b16(a_progress, a_mask);
        nk_u64_t b_step = svcntp_b16(b_progress, b_mask);

        // Compare `a_vec` with each lane of `b_vec`
        svbfloat16_t a_weights_vec = svld1_bf16(a_progress, (__bf16 const *)a_weights + a_idx);
        svbfloat16_t b_weights_vec = svld1_bf16(b_progress, (__bf16 const *)b_weights + b_idx);
        for (nk_size_t i = 0; i < lanes_count; i++) {
            svbool_t equal_mask = svmatch_u16(a_progress, a_vec, b_vec);
            //! The `svsel_bf16` intrinsic is broken in many compilers, not returning the correct type.
            //! So we reinterprete floats as integers and apply `svsel_s16`, but the `svreinterpret_s16_bs16`
            //! and `svreinterpret_bf16_s16` are not always properly defined!
            svint16_t b_equal_weights_vec = svsel_s16(equal_mask, svreinterpret_s16_bf16(b_weights_vec),
                                                      svdup_n_s16(0));
            product_vec = svbfdot_f32(product_vec, a_weights_vec, svreinterpret_bf16_s16(b_equal_weights_vec));
            b_vec = svext_u16(b_vec, b_vec, 8);
        }

        // Advance
        a_idx += a_step;
        b_idx += b_step;
    }
    *product = svaddv_f32(svptrue_b32(), product_vec);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ARM_
#endif // NK_SPARSE_SVE2_H
