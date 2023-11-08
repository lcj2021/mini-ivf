#ifndef INCLUDE_DISTANCE_HPP
#define INCLUDE_DISTANCE_HPP
// http://koturn.hatenablog.com/entry/2016/07/18/090000
// windows is not supported, but just in case (later someone might implement)
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=590,27,2
#ifdef _MSC_VER
#  include <immintrin.h> 
#else
#  include <x86intrin.h>
#endif
#include <string>

// These fast L2 squared distance codes (SSE and AVX) are from the Faiss library:
// https://github.com/facebookresearch/faiss/blob/master/utils.cpp
//
// Based on them, AVX512 implementation is also prepared.
// But it doesn't seem drastically fast. Only slightly faster than AVX:
// (runtime) REF >> SSE >= AVX ~ AVX512


// ========================= Distance functions ============================

#if defined(__AVX512F__)  
static const std::string g_simd_architecture = "avx512";
// AVX512 implementation by Yusuke
float vec_L2sqr (const float *x, const float *y, size_t d);
float vec_L2sqr(const uint8_t *x, const uint8_t *y, size_t d);

#elif defined (__AVX__)  
static const std::string g_simd_architecture = "avx";
// AVX implementation
float vec_L2sqr (const float *x, const float *y, size_t d);
float vec_L2sqr(const uint8_t *x, const uint8_t *y, size_t d);

#else 
static const std::string g_simd_architecture = "sse";
// SSE implementation. Unroot!
float vec_L2sqr(const float *x, const float *y, size_t d);
float vec_L2sqr(const uint8_t *x, const uint8_t *y, size_t d);


#endif

// namespace Anonymous

#endif // DISTANCE_H
