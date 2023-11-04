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



// From Faiss.
// Reference implementation
float fvec_L2sqr_ref(const float *x, const float *y, size_t d);
float fvec_L2sqr(const uint8_t *x, const uint8_t *y, size_t d);
float fvec_L2sqr(const uint8_t *x, const float *y, size_t d);


// ========================= Reading functions ============================

// Reading function for SSE, AVX, and AVX512
// This function is from Faiss
// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x);

#if defined(__AVX__)

// Reading function for AVX and AVX512
// This function is from Faiss
// reads 0 <= d < 8 floats as __m256
static inline __m256 masked_read_8 (int d, const float *x);

#endif // __AVX__

#if defined(__AVX512F__) 
// Reading function for AVX512
// reads 0 <= d < 16 floats as __m512
static inline __m512 masked_read_16 (int d, const float *x);

#endif // __AVX512F__



// ========================= Distance functions ============================

#if defined(__AVX512F__)  
static const std::string g_simd_architecture = "avx512";

// AVX512 implementation by Yusuke
float fvec_L2sqr (const float *x, const float *y, size_t d);

#elif defined (__AVX__)  
static const std::string g_simd_architecture = "avx";

// This function is from Faiss
// AVX implementation
float fvec_L2sqr (const float *x, const float *y, size_t d);

#else 
static const std::string g_simd_architecture = "sse";


// This function is from Faiss
// SSE implementation. Unroot!
float fvec_L2sqr(const float *x, const float *y, size_t d);


#endif

// namespace Anonymous

#endif // DISTANCE_H
