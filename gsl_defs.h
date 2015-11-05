#ifndef GSL_BLAS_H_GUARD
#define GSL_BLAS_H_GUARD

#include "cblas.h"

#ifdef __cplusplus
extern "C" {
#endif


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

#endif /* GSL_BLAS_H_GUARD */
