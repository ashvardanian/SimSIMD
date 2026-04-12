/**
 *  @brief SVE horizontal reduction helpers with MSan unpoisoning.
 *  @file include/numkong/reduce/sve.h
 *  @author Ash Vardanian
 *  @date April 12, 2026
 *
 *  LLVM's MSan does not instrument ARM SVE intrinsics — `svaddv` moves data
 *  from vector to scalar registers via architecture-specific paths invisible
 *  to the compiler, causing false-positive uninitialized-value reports.
 *  These macros wrap the reduction and unpoison the scalar result.
 *
 *  Implemented as statement-expression macros rather than inline functions because
 *  `svaddv` callers span both SVE and SME streaming translation units with
 *  incompatible target attributes — an `always_inline` function compiled for
 *  `+sve` cannot be inlined into a streaming-mode caller compiled for `+sme`,
 *  and vice-versa. Macros expand in the caller's target context, avoiding this.
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_SVE_H
#define NK_REDUCE_SVE_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVE || NK_TARGET_SVE2 || NK_TARGET_SME

#include "numkong/types.h"

#define nk_svaddv_f64_(predicate, vector)                \
    ({                                                   \
        nk_f64_t r_ = svaddv_f64((predicate), (vector)); \
        nk_unpoison_(&r_, sizeof(r_));                    \
        r_;                                              \
    })

#define nk_svaddv_f32_(predicate, vector)                \
    ({                                                   \
        nk_f32_t r_ = svaddv_f32((predicate), (vector)); \
        nk_unpoison_(&r_, sizeof(r_));                    \
        r_;                                              \
    })

#define nk_svaddv_u32_(predicate, vector)                \
    ({                                                   \
        nk_u64_t r_ = svaddv_u32((predicate), (vector)); \
        nk_unpoison_(&r_, sizeof(r_));                    \
        r_;                                              \
    })

#define nk_svaddv_s32_(predicate, vector)                \
    ({                                                   \
        nk_i64_t r_ = svaddv_s32((predicate), (vector)); \
        nk_unpoison_(&r_, sizeof(r_));                    \
        r_;                                              \
    })

#define nk_svaddv_u8_(predicate, vector)                \
    ({                                                  \
        nk_u64_t r_ = svaddv_u8((predicate), (vector)); \
        nk_unpoison_(&r_, sizeof(r_));                   \
        r_;                                             \
    })

#endif // NK_TARGET_SVE || NK_TARGET_SVE2 || NK_TARGET_SME
#endif // NK_TARGET_ARM64_
#endif // NK_REDUCE_SVE_H
