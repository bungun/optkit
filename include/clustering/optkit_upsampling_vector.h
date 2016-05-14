#ifndef OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_
#define OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_

#include "optkit_vector.h"
#include "optkit_matrix.h"
#include "optkit_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OK_CHECK_UPSAMPLINGVEC
#define OK_CHECK_UPSAMPLINGVEC(u) \
	do { \
		if (!u || !u->indices) \
			return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED ); \
	} while(0)
#endif

typedef struct upsamplingvec {
	size_t size1, size2, stride;
	size_t * indices;
	indvector vec;
} upsamplingvec;

/* COMMON IMPLEMENTATION */
ok_status upsamplingvec_alloc(upsamplingvec * u, size_t size1, size_t size2);
ok_status upsamplingvec_free(upsamplingvec * u);
ok_status upsamplingvec_check_bounds(const upsamplingvec * u);
ok_status upsamplingvec_update_size(upsamplingvec * u);
ok_status upsamplingvec_subvector(upsamplingvec * usub, upsamplingvec * u,
	size_t offset1, size_t length1, size_t size2);

/* CPU/GPU-SPECIFIC IMPLEMENTATION */
ok_status upsamplingvec_mul_matrix(void * linalg_handle,
	const enum CBLAS_TRANSPOSE transU, const enum CBLAS_TRANSPOSE transI,
	const enum CBLAS_TRANSPOSE transO, const ok_float alpha,
	upsamplingvec * u, matrix * M_in, ok_float beta, matrix * M_out);
ok_status upsamplingvec_count(const upsamplingvec * u, vector * counts);

/* LOCAL UTILITY */
static inline int upsampling_dims_compatible(const int transpose,
	const upsamplingvec * u, const size_t input_dim1,
	const size_t input_dim2, const size_t output_dim1,
	const size_t output_dim2)
{
	if (transpose)
		return (input_dim2 == output_dim2) &&
		((u->size2 <= output_dim1) && (u->size1 == input_dim1));
	else
		return (input_dim2 == output_dim2) &&
		((u->size1 == output_dim1) && (u->size2 <= input_dim1));
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_ */
