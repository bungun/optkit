#ifndef GSL_DEFS_H_GUARD
#define GSL_DEFS_H_GUARD

#include <tgmath.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gsl_cblas.h"


#ifdef __cplusplus
extern "C" {
#endif

#define GSL_NULL 0
#define gsl_free(x) free(x); x=GSL_NULL

typedef enum CBLAS_ORDER CBLAS_ORDER_t;
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE_t;
typedef enum CBLAS_UPLO CBLAS_UPLO_t;
typedef enum CBLAS_DIAG CBLAS_DIAG_t;
typedef enum CBLAS_SIDE CBLAS_SIDE_t;

#ifndef FLOAT
    #define CBLAS(x) cblas_d ## x
    typedef double gsl_float;
#else
    #define CBLAS(x) cblas_s ## x
    typedef float gsl_float;
#endif


#ifdef __cplusplus
}
#endif

#endif /* GSL_DEFS_H_GUARD */
