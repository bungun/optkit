#ifndef OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_
#define OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_

#include "optkit_vector.h"
#include "optkit_matrix.h"
#include "optkit_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct upsamplingvec {
	size_t size1, size2, stride;
	size_t * indices;
	indvector vec;
} upsamplingvec;

ok_status upsamplingvec_alloc(upsamplingvec * u, size_t size1, size_t size2);
ok_status upsamplingvec_free(upsamplingvec * u);
ok_status upsamplingvec_check_bounds(const upsamplingvec * u);
ok_status upsamplingvec_subvector(upsamplingvec * usub, upsamplingvec * u,
	size_t offset1, size_t length1, size_t size2);
ok_status upsamplingvec_mul_matrix(const enum CBLAS_TRANSPOSE transU,
	const enum CBLAS_TRANSPOSE transI, const enum CBLAS_TRANSPOSE transO,
	const ok_float alpha, upsamplingvec * u, matrix * M_in, ok_float beta,
	matrix * M_out);
ok_status upsamplingvec_count(const upsamplingvec * u, vector * counts);

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_ */
