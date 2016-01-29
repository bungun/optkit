#ifndef OPTKIT_DEFS_H_GUARD
#define OPTKIT_DEFS_H_GUARD

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
#define OPTKIT_VERSION_CHANGE 3
#define OPTKIT_VERSION_STATUS 'a'
#define OPTKIT_VERSION get_optkit_version

inline void
get_optkit_version(int * maj, int * min, int * change, int * status){
    * maj = OPTKIT_VERSION_MAJOR;
    * min = OPTKIT_VERSION_MINOR;
    * change = OPTKIT_VERSION_CHANGE;
    * status = (int) OPTKIT_VERSION_STATUS;
}

#ifdef OK_DEBUG
#define PRINT_DEBUG printf
#else
#define PRINT_DEBUG
#endif


#define OK_NULL 0
#define ok_free(x) free(x); x = OK_NULL

 
typedef unsigned int uint;
typedef int ok_int;

/* C BLAS enums */
typedef enum CBLAS_ORDER CBLAS_ORDER_t;
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE_t;
typedef enum CBLAS_UPLO CBLAS_UPLO_t;
typedef enum CBLAS_DIAG CBLAS_DIAG_t;
typedef enum CBLAS_SIDE CBLAS_SIDE_t;


#ifndef FLOAT
    #define CBLAS(x) cblas_d ## x
    #define MATH(x) x 
    typedef double ok_float;
    #define MACHINETOL (ok_float) 10e-10
    #ifndef NAN
	    #define NAN ((ok_float)0x7ff8000000000000) 
    #endif
    #define OK_FLOAT_MAX FLT_MAX
#else
    #define CBLAS(x) cblas_s ## x
    #define MATH(x) x ## f
    typedef float ok_float;
    #define MACHINETOL (ok_float) 10e-5
    #ifndef NAN
	    #define NAN ((ok_float)0x7fc00000) 
    #endif
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


#ifndef INFINITY
#define INFINITY NAN
#endif


#ifdef __CUDACC__
#define __DEVICE__ __device__
#else
#define __DEVICE__
#endif


#ifdef __cplusplus
}
#endif

#endif /* GSL_DEFS_H_GUARD */
