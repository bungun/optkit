#ifndef OPTKIT_DEFS_H_
#define OPTKIT_DEFS_H_

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gsl_cblas.h"

#ifdef __CUDACC__
#include "cublas_v2.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define OPTKIT_VERSION_MAJOR 0
#define OPTKIT_VERSION_MINOR 0
#define OPTKIT_VERSION_CHANGE 4
#define OPTKIT_VERSION_STATUS 'a' /* 'a' = alpha, 'b' = beta, 0 = release */


#ifdef OK_DEBUG
#define PRINT_DEBUG printf
#else
#define PRINT_DEBUG
#endif


#define OK_NULL 0
#define ok_free(x) free(x); x = OK_NULL


typedef unsigned int uint;
typedef int ok_int;

#ifndef FLOAT
        #define CBLAS(x) cblas_d ## x
        #define MATH(x) x
        typedef double ok_float;
        #define MACHINETOL (ok_float) 10e-10
        #define OK_NAN ((ok_float)0x7ff8000000000000)
        #endif
        #define OK_FLOAT_MAX FLT_MAX
#else
        #define CBLAS(x) cblas_s ## x
        #define MATH(x) x ## f
        typedef float ok_float;
        #define MACHINETOL (ok_float) 10e-5
        #define OK_NAN ((ok_float)0x7fc00000)
        #define OK_FLOAT_MAX DBL_MAX
#endif


#if defined(OPTKIT_ROWMAJOR)
#define OPTKIT_ORDER 101
#elif defined(OPTKIT_COLMAJOR)
#define OPTKIT_ORDER 102
#else
#undef OPTKIT_ORDER
#endif

#define kZero (ok_float) 0
#define kOne (ok_float) 1


#ifndef OK_INFINITY
#define OK_INFINITY OK_NAN
#endif


#ifdef __CUDACC__
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_DEFS_H_ */