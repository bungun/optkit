#ifndef OPTKIT_ANDERSON_DIFFERENCE_H_
#define OPTKIT_ANDERSON_DIFFERENCE_H_

#include "optkit_dense.h"
#include "optkit_anderson_reductions.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct difference_accelerator{
	size_t vector_dim, lookback_dim;
	matrix *DX, *DF, *DG, *DXDF;
	vector *f, *g, *x, *alpha;
	int_vector *pivot;
	size_t iter;
	void *blas_handle;
	void *lapack_handle;
} difference_accelerator;

typedef struct fused_diff_accelerator{
	size_t vector_dim, lookback_dim, n_blocks;
	matrix *DX, *DF, *DG, *DXDF;
	vector *w, *f, *g, *x, *alpha;
	int_vector *pivot;
	size_t iter;
	void *blas_handle;
	void *lapack_handle;
} fused_diff_accelerator;


/* ANDERSON, DIFFERENCE FORMULATION */
ok_status anderson_difference_solve(void *blas_hdl, void *lapack_hdl, matrix *DX,
	matrix *DF, matrix *DXDF, vector *f, vector *alpha, int_vector *pivot);
ok_status anderson_difference_mix(void * blas_hdl, matrix *DG, vector *alpha,
	vector *x);
ok_status anderson_difference_accelerate_template(void *blas_handle,
	void *lapack_handle, matrix *DX, matrix *DF, matrix *DG, matrix *DXDF,
	vector *f, vector *g, vector *x, vector *alpha, int_vector *pivot,
	size_t *iter, vector *iterate, vector *x_reduced, size_t n_reduction,
	ok_status (* x_reduction)(vector *x_rdx, vector *x, size_t n_rdx));

/* UNMODIFIED STATE VARIABLE */
ok_status anderson_difference_accelerator_init(difference_accelerator *aa,
	size_t vector_dim, size_t lookback_dim);
ok_status anderson_difference_accelerator_free(difference_accelerator *aa);
ok_status anderson_difference_set_x0(difference_accelerator *aa, vector *x_initial);
ok_status anderson_difference_accelerate(difference_accelerator *aa, vector *x);

/* SUMMED STATE VARIABLE BLOCKS */
ok_status anderson_fused_diff_accelerator_init(fused_diff_accelerator *aa,
	size_t vector_dim, size_t lookback_dim, size_t n_blocks);
ok_status anderson_fused_diff_accelerator_free(fused_diff_accelerator *aa);
ok_status anderson_fused_diff_set_x0(fused_diff_accelerator *aa,
	vector *x_initial);
ok_status anderson_fused_diff_accelerate(fused_diff_accelerator *aa,
	vector *x);


#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_ANDERSON_DIFFERENCE_H_ */
