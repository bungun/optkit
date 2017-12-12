#ifndef OPTKIT_ANDERSON_REDUCTIONS_H_
#define OPTKIT_ANDERSON_REDUCTIONS_H_

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

static ok_status anderson_sum_blocks(vector *w, vector *x, size_t n_blocks)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t k;
	vector xsub;
	OK_CHECK_VECTOR(w);
	OK_CHECK_VECTOR(x);
	if (x->size != n_blocks * w->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	OK_CHECK_ERR( err, vector_set_all(w, kZero) );
	for (k = 0; k < n_blocks; ++k) {
		OK_CHECK_ERR( err, vector_subvector(&xsub, x, k * w->size, w->size) );
		OK_CHECK_ERR( err, vector_add(w, &xsub) );
	}
	return err;
}

static ok_status anderson_reduce_null(vector *w, vector *x, size_t n_null)
{
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_ANDERSON_H_ */
