#ifndef OPTKIT_DEFS_H_
#define OPTKIT_DEFS_H_

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include "gsl_cblas.h"

#ifdef __CUDACC__
#include "cublas_v2.h"
#include <cusparse.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define OPTKIT_VERSION_MAJOR 0
#define OPTKIT_VERSION_MINOR 0
#define OPTKIT_VERSION_CHANGE 4
#define OPTKIT_VERSION_STATUS 'b' /* 'a' = alpha, 'b' = beta, 0 = release */

#ifdef OK_DEBUG
#define PRINT_DEBUG printf
#else
#define PRINT_DEBUG
#endif

#define OK_NULL 0

#ifndef ok_alloc
#define ok_alloc(x, len) \
	do { \
		x = malloc(len); \
		memset(x, 0, len); \
	} while(0)
#endif

#ifndef ok_free
#define ok_free(x) \
	do { \
		if (x) { \
			free(x); \
			x = OK_NULL; \
		} \
	} while(0)
#endif

typedef unsigned int uint;
typedef int ok_int;
typedef enum optkit_status {
	OPTKIT_SUCCESS = 0,
	OPTKIT_ERROR = 1,
	OPTKIT_ERROR_CUDA = 2,
	OPTKIT_ERROR_CUBLAS = 3,
	OPTKIT_ERROR_CUSPARSE = 4,
	OPTKIT_ERROR_DOMAIN = 10,
	OPTKIT_ERROR_DIVIDE_BY_ZERO = 11,
	OPTKIT_ERROR_LAYOUT_MISMATCH = 100,
	OPTKIT_ERROR_DIMENSION_MISMATCH = 101,
	OPTKIT_ERROR_OUT_OF_BOUNDS = 102,
	OPTKIT_ERROR_OVERWRITE = 1000,
	OPTKIT_ERROR_UNALLOCATED = 1001
} ok_status;

#define OK_SCAN_ERR(err) ok_print_status(err, __FILE__, __LINE__, __func__)

#define OK_CHECK_ERR(err, call) \
	do { \
		if (!err) \
			err = OK_SCAN_ERR( call ); \
	} while(0)

#define OK_MAX_ERR(err, call) \
	do { \
		ok_status newerr = OK_SCAN_ERR( call ); \
		err = err > newerr ? err : newerr; \
	} while(0)

#define OK_RETURNIF_ERR(call) \
	do { \
		ok_status err = OK_SCAN_ERR( call ); \
		if (err) \
			return err; \
	} while(0)

#define OK_CHECK_PTR(ptr) \
	do { \
		if (!ptr) \
			return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED ); \
	} while(0)

enum OPTKIT_TRANSFORM {
	OkTransformScale = 0,
	OkTransformAdd = 1,
	OkTransformIncrement = 2,
	OkTransformDecrement = 3
};


#ifndef FLOAT
	#define CBLAS(x) cblas_d ## x
	#define CBLASI(x) cblas_id ## x
	#define MATH(x) x
	typedef double ok_float;
	#define MACHINETOL (double) 10e-10
	#define OK_NAN ((double) 0x7ff8000000000000)
	#define OK_FLOAT_MAX (double) DBL_MAX
#else
	#define CBLAS(x) cblas_s ## x
	#define CBLASI(x) cblas_is ## x
	#define MATH(x) x ## f
	typedef float ok_float;
	#define MACHINETOL (float) 10e-5
	#define OK_NAN ((float) 0x7fc00000)
	#define OK_FLOAT_MAX (float) FLT_MAX
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

void optkit_version(int * maj, int * min, int * change, int * status);
ok_status ok_device_reset(void);

static const char * ok_err2string(const ok_status error) {
	switch(error) {
	case OPTKIT_SUCCESS:
		return "OPTKIT_SUCCESS";
	case OPTKIT_ERROR:
		return "OPTKIT_ERROR";
	case OPTKIT_ERROR_CUDA:
		return "OPTKIT_ERROR_CUDA";
	case OPTKIT_ERROR_CUBLAS:
		return "OPTKIT_ERROR_CUBLAS";
	case OPTKIT_ERROR_CUSPARSE:
		return "OPTKIT_ERROR_CUSPARSE";
	case OPTKIT_ERROR_DOMAIN:
		return "OPTKIT_ERROR_DOMAIN";
	case OPTKIT_ERROR_DIVIDE_BY_ZERO:
		return "OPTKIT_ERROR_DIVIDE_BY_ZERO";
	case OPTKIT_ERROR_LAYOUT_MISMATCH:
		return "OPTKIT_ERROR_LAYOUT_MISMATCH";
	case OPTKIT_ERROR_DIMENSION_MISMATCH:
		return "OPTKIT_ERROR_DIMENSION_MISMATCH";
	case OPTKIT_ERROR_OUT_OF_BOUNDS:
		return "OPTKIT_ERROR_OUT_OF_BOUNDS";
	case OPTKIT_ERROR_OVERWRITE:
		return "OPTKIT_ERROR_OVERWRITE";
	case OPTKIT_ERROR_UNALLOCATED:
		return "OPTKIT_ERROR_UNALLOCATED";
	default:
		return "<unknown error>";
	}
}

static ok_status ok_print_status(ok_status err, const char * file,
	const int line, const char * function)
{
	if (err != OPTKIT_SUCCESS)
		printf("%s:%d:%s\n ERROR_OPTKIT: %s\n", file, line, function,
			ok_err2string(err));
	return err;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_DEFS_H_ */
