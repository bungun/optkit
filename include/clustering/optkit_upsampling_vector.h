#ifndef OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_
#define OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_

#include "optkit_vector.h"
#include "optkit_matrix.h"
#include "optkit_blas.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct upsamplicvec {
	size_t size1, size2, stride;
	size_t * indices;
	indvector vec;
} upsamplingvec;

ok_status upsamplicvec_alloc(upsamplingvec * u, size_t size1, size2);
ok_status upsamplingvec_free(upsamplingvec * u);
ok_status upsamplingvec_check_size(const upsamplingvec * u);
ok_status upsamplingvec_mul_matrix(const ok_float alpha, upsamplingvec * u,
	matrix * M_in, ok_float beta, matrix * M_out);
ok_status upsamplingvecT_mul_matrix(const ok_float alpha, upsamplingvec * u,
	matrix * M_in, ok_float beta, matrix * M_out);
ok_status upsampling_vec_count(const upsamplingvec * u, vector * counts);
ok_status upsamplingvec_shift(upsamplingvec * u, const size_t shift,
	const enum OPTKIT_TRANSFORM direction);

#endif /* OPTKIT_CLUSTERING_UPSAMPLING_VECTOR_H_ */