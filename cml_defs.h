#ifndef CML_DEFS_H_GUARD
#define CML_DEFS_H_GUARD

#include "gsl_defs.h"
#include "cublas_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

#define cml_free(x) cudaFree(x)

#ifndef FLOAT
    #define CUBLAS(x) cublasD ## x
#else
    #define CUBLAS(x) cublasS ## x
#endif


#ifdef __cplusplus
}
#endif

#endif /* CML_DEFS_H_GUARD */
