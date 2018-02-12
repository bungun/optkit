#ifndef OPTKIT_IMPL_POGS_DENSE_H_
#define OPTKIT_IMPL_POGS_DENSE_H_

#include "optkit_pogs_datatypes.h"
#include "optkit_equilibration.h"
#include "optkit_projector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OPTKIT_INDIRECT
typedef direct_projector pogs_dense_projector;
#define PROJECTOR(x) direct_projector_ ## x
#else
typedef indirect_projector pogs_dense_projector;
#define PROJECTOR(x) indirect_projector_ ## x
#endif

typedef struct POGSDenseWork {
	matrix *A;
	pogs_dense_projector *P;
	vector *d, *e;
	ok_float normA;
	int skinny, normalized, equilibrated;
	void *blas_handle;
	void *lapack_handle;
} pogs_dense_work;

typedef struct POGSDenseSolverFlags {
	size_t m, n;
	enum CBLAS_ORDER ord;
} pogs_dense_solver_flags;

typedef struct POGSDenseSolverPrivateData {
	ok_float *A_equil, *d, *e;
#ifndef OPTKIT_INDIRECT
	ok_float *ATA_cholesky;
#endif
} pogs_dense_solver_private_data;

ok_status pogs_dense_problem_data_alloc(pogs_dense_work *W, ok_float *A,
	const pogs_dense_solver_flags *flags);
ok_status pogs_dense_problem_data_free(pogs_dense_work *W);
ok_status pogs_dense_get_init_data(ok_float **A,
	const pogs_dense_solver_private_data *data,
	const pogs_dense_solver_flags *flags);

ok_status pogs_dense_apply_matrix(pogs_dense_work *W, ok_float alpha,
	vector *x, ok_float beta, vector *y);
ok_status pogs_dense_apply_adjoint(pogs_dense_work *W, ok_float alpha,
	vector *x, ok_float beta, vector *y);
ok_status pogs_dense_project_graph(pogs_dense_work *W, vector *x_in,
	vector *y_in, vector *x_out, vector *y_out, ok_float tol);

ok_status pogs_dense_equilibrate_matrix(pogs_dense_work *W, ok_float *A,
	const pogs_dense_solver_flags *flags);
ok_status pogs_dense_initalize_graph_projector(pogs_dense_work *W);
ok_status pogs_dense_estimate_norm(pogs_dense_work *W, ok_float *normest);
ok_status pogs_dense_work_get_norm(pogs_dense_work *W);
ok_status pogs_dense_work_normalize(pogs_dense_work *W);

ok_status pogs_dense_save_work(pogs_dense_solver_private_data *data,
	pogs_dense_solver_flags *flags, const pogs_dense_work *W);
ok_status pogs_dense_load_work(pogs_dense_work *W,
	const pogs_dense_solver_private_data *data,
	const pogs_dense_solver_flags *flags);

/* DEFINITIONS */
ok_status pogs_dense_problem_data_alloc(pogs_dense_work *W, ok_float *A,
	const pogs_dense_solver_flags *flags)
{
	ok_status err = OPTKIT_SUCCESS;
	if (!W || !flags)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (W->A)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	/* allocate matrix */
	ok_alloc(W->A, sizeof(*W->A));
	OK_CHECK_ERR( err, matrix_calloc(W->A, flags->m, flags->n, flags->ord) );

	/* allocate projector */
	ok_alloc(W->P, sizeof(pogs_dense_projector));
	OK_CHECK_ERR( err, PROJECTOR(alloc)(W->P, W->A) );
	return err;
}

ok_status pogs_dense_problem_data_free(pogs_dense_work *W)
{
	OK_CHECK_PTR(W);
	ok_status err = OK_SCAN_ERR( PROJECTOR(free)(W->P) );
	ok_free(W->P);
	OK_MAX_ERR( err, matrix_free(W->A) );
	ok_free(W->A);
	return err;
}

ok_status pogs_dense_get_init_data(ok_float **A,
	const pogs_dense_solver_private_data *data,
	const pogs_dense_solver_flags *flags)
{
	OK_CHECK_PTR(data);
	if (*A)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	*A = data->A_equil;
	return OPTKIT_SUCCESS;
}

ok_status pogs_dense_apply_matrix(pogs_dense_work *W, ok_float alpha,
	vector *x, ok_float beta, vector *y)
{
	return OK_SCAN_ERR( blas_gemv(W->blas_handle, CblasNoTrans, alpha,
		W->A, x, beta, y) );
}

ok_status pogs_dense_apply_adjoint(pogs_dense_work *W, ok_float alpha,
	vector *x, ok_float beta, vector *y)
{
	return OK_SCAN_ERR( blas_gemv(W->blas_handle, CblasTrans, alpha, W->A,
		x, beta, y) );
}

ok_status pogs_dense_project_graph(pogs_dense_work *W, vector *x_in,
	vector *y_in, vector *x_out, vector *y_out, ok_float tol)
{
	/* ignore arg ``tol``, included for method signature consistency */
	return OK_SCAN_ERR( PROJECTOR(project)(W->blas_handle, W->P, x_in,
		y_in, x_out, y_out) );
}


ok_status pogs_dense_equilibrate_matrix(pogs_dense_work *W, ok_float *A,
	const pogs_dense_solver_flags *flags)
{
	ok_status err = OK_SCAN_ERR( regularized_sinkhorn_knopp(
		W->blas_handle, A, W->A, W->d, W->e, flags->ord) );
	W->equilibrated = (err == OPTKIT_SUCCESS);
	return err;
}

ok_status pogs_dense_initalize_graph_projector(pogs_dense_work *W)
{
	return OK_SCAN_ERR( PROJECTOR(initialize)(W->blas_handle, W->P, 1) );
}

/* STUB */
ok_status pogs_dense_estimate_norm(pogs_dense_work *W, ok_float *normest)
{
	*normest = kOne;
	return OPTKIT_SUCCESS;
}

ok_status pogs_dense_work_get_norm(pogs_dense_work *W)
{
	if (!W || !W->P)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	W->normalized = W->P->normalized;
	W->normA = W->P->normA;
	return OPTKIT_SUCCESS;
}

ok_status pogs_dense_work_normalize(pogs_dense_work *W)
{
	if (!W || !W->A)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	return OK_SCAN_ERR( matrix_scale(W->A, kOne / W->normA) );
}

ok_status pogs_dense_save_work(pogs_dense_solver_private_data *data,
	pogs_dense_solver_flags *flags, const pogs_dense_work *W)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR( err, matrix_memcpy_am(data->A_equil, W->A, flags->ord) );
	#ifndef OPTKIT_INDIRECT
	OK_CHECK_ERR( err, matrix_memcpy_am(data->ATA_cholesky, W->P->L,
		flags->ord) );
	#endif
	OK_CHECK_ERR( err, vector_memcpy_av(data->d, W->d, 1) );
	OK_CHECK_ERR( err, vector_memcpy_av(data->e, W->e, 1) );
	flags->m = W->A->size1;
	flags->n = W->A->size2;
	return err;
}

ok_status pogs_dense_load_work(pogs_dense_work *W,
	const pogs_dense_solver_private_data *data,
	const pogs_dense_solver_flags *flags)
{
	ok_status err = OPTKIT_SUCCESS;
	OK_CHECK_ERR( err, matrix_memcpy_ma(W->A, data->A_equil, flags->ord) );
	#ifndef OPTKIT_INDIRECT
	OK_CHECK_ERR( err, matrix_memcpy_ma(W->P->L, data->ATA_cholesky,
		flags->ord) );
	#endif
	OK_CHECK_ERR( err, vector_memcpy_va(W->d, data->d, 1) );
	OK_CHECK_ERR( err, vector_memcpy_va(W->e, data->e, 1) );
	return err;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_POGS_DENSE_H_ */
