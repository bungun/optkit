#ifndef OPTKIT_POGS_IMPL_ABSTRACT_H_
#define OPTKIT_POGS_IMPL_ABSTRACT_H_

#include "optkit_pogs_datatypes.h"
#include "optkit_operator_dense.h"
#include "optkit_operator_sparse.h"
#include "optkit_operator_typesafe.h"
#include "optkit_equilibration.h"
#include "optkit_projector.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct POGSAbstractWork {
	abstract_operator * A;
	projector * P;
	ok_status (* operator_scale)(abstract_operator * o, const
		ok_float scaling);
	ok_status (* operator_equilibrate)(void * linalg_handle,
		abstract_operator * o, vector * d, vector * e,
		const ok_float pnorm);
	vector * d, * e;
	ok_float normA;
	int skinny, normalized, equilibrated;
	void * linalg_handle;
} pogs_abstract_work;

typedef struct POGSAbstractSolverFlags {
	int direct;
	ok_float equil_norm;
} pogs_abstract_solver_flags;

typedef struct POGSAbstractSolverPrivateData {
	ok_float * d, * e;
} pogs_abstract_solver_private_data;

ok_status pogs_abstract_problem_data_alloc(pogs_abstract_work * W,
	abstract_operator * A, const pogs_abstract_solver_flags * flags);
ok_status pogs_abstract_problem_data_free(pogs_abstract_work * W);
ok_status pogs_abstract_get_init_data(abstract_operator * A,
	const pogs_abstract_solver_private_data * data,
	const pogs_abstract_solver_flags * flags);

ok_status pogs_abstract_apply_matrix(pogs_abstract_work * W, ok_float alpha,
	vector * x, ok_float beta, vector * y);
ok_status pogs_abstract_apply_adjoint(pogs_abstract_work * W, ok_float alpha,
	vector * x, ok_float beta, vector * y);
ok_status pogs_abstract_project_graph(pogs_abstract_work * W, vector * x_in,
	vector * y_in, vector * x_out, vector * y_out, ok_float tol);

ok_status pogs_abstract_equilibrate_matrix(pogs_abstract_work * W,
	abstract_operator * A, const pogs_abstract_solver_flags * flags);
ok_status pogs_abstract_initalize_graph_projector(pogs_abstract_work * W);
ok_status pogs_abstract_estimate_norm(pogs_abstract_work * W, ok_float * normest);
ok_status pogs_abstract_work_get_norm(pogs_abstract_work * W);
ok_status pogs_abstract_work_normalize(pogs_abstract_work * W);

ok_status pogs_abstract_save_work(pogs_abstract_solver_private_data * data,
	pogs_abstract_solver_flags * flags, const pogs_abstract_work * W);
ok_status pogs_abstract_load_work(pogs_abstract_work * W,
	const pogs_abstract_solver_private_data * data,
	const pogs_abstract_solver_flags * flags);

abstract_operator * pogs_dense_operator_gen(const ok_float * A, size_t m,
	size_t n, enum CBLAS_ORDER order);
abstract_operator * pogs_sparse_operator_gen(const ok_float * val,
	const ok_int * ind, const ok_int * ptr, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order);
ok_status pogs_dense_operator_free(abstract_operator * A);
ok_status pogs_sparse_operator_free(abstract_operator * A);

/* DEFINITIONS */
ok_status pogs_abstract_problem_data_alloc(pogs_abstract_work * W,
	abstract_operator * A, const pogs_abstract_solver_flags * flags)
{
	if (!W || !A || !flags)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	OK_CHECK_OPERATOR(A);
	int dense_or_sparse = (A->kind == OkOperatorDense ||
		A->kind == OkOperatorSparseCSC ||
		A->kind == OkOperatorSparseCSR);

	/* bind abstract linear operator */
	W->A = A;

	/* allocate projector */
	if (flags->direct && A->kind == OkOperatorDense)
		W->P = dense_direct_projector_alloc(
			dense_operator_get_matrix_pointer(W->A));
	else
		W->P = indirect_projector_generic_alloc(W->A);

	if (!W->P)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	/* set equilibration methods */
	if  (!dense_or_sparse) {
		W->operator_equilibrate = operator_equilibrate;
		W->operator_scale = typesafe_operator_scale;
	} else {
		W->operator_equilibrate = operator_regularized_sinkhorn;
		if (A->kind == OkOperatorDense)
			W->operator_scale = dense_operator_scale;
		else
			W->operator_scale = sparse_operator_scale;
	}
	return OPTKIT_SUCCESS;
}

ok_status pogs_abstract_problem_data_free(pogs_abstract_work * W)
{
	OK_CHECK_PTR(W);
	ok_status err = OK_SCAN_ERR( W->P->free(W->P->data) );
	ok_free(W->P);
	return err;
}

ok_status pogs_abstract_get_init_data(abstract_operator * A,
	const pogs_abstract_solver_private_data * data,
	const pogs_abstract_solver_flags * flags)
{
	return OK_SCAN_ERR( OPTKIT_ERROR_NOT_IMPLEMENTED );
}

ok_status pogs_abstract_apply_matrix(pogs_abstract_work * W, ok_float alpha,
	vector * x, ok_float beta, vector * y)
{
	return OK_SCAN_ERR( W->A->fused_apply(W->A->data, alpha, x, beta, y) );
}

ok_status pogs_abstract_apply_adjoint(pogs_abstract_work * W, ok_float alpha,
	vector * x, ok_float beta, vector * y)
{
	return OK_SCAN_ERR( W->A->fused_adjoint(W->A->data, alpha, x, beta, y) );
}

ok_status pogs_abstract_project_graph(pogs_abstract_work * W, vector * x_in,
	vector * y_in, vector * x_out, vector * y_out, ok_float tol)
{
	return OK_SCAN_ERR( W->P->project(W->P->data, x_in, y_in, x_out, y_out,
		tol) );
}

ok_status pogs_abstract_equilibrate_matrix(pogs_abstract_work * W,
	abstract_operator * A, const pogs_abstract_solver_flags * flags)
{
	ok_status err = OK_SCAN_ERR( W->operator_equilibrate(W->linalg_handle,
		W->A, W->d, W->e, flags->equil_norm) );
	W->equilibrated = (err == OPTKIT_SUCCESS);
	return err;
}

ok_status pogs_abstract_initalize_graph_projector(pogs_abstract_work * W)
{
	int normalize = (int)(W->P->kind == OkProjectorDenseDirect);
	return OK_SCAN_ERR( W->P->initialize(W->P->data, normalize) );
}

ok_status pogs_abstract_estimate_norm(pogs_abstract_work * W, ok_float * normest)
{
	return OK_SCAN_ERR(
		operator_estimate_norm(W->linalg_handle, W->A, normest) );
}

ok_status pogs_abstract_work_get_norm(pogs_abstract_work * W)
{
	if (!W || !W->P)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_RETURNIF_ERR( projector_normalization(W->P, &W->normalized) );
	OK_RETURNIF_ERR( projector_get_norm(W->P, &W->normA) );
	return OPTKIT_SUCCESS;
}

ok_status pogs_abstract_work_normalize(pogs_abstract_work * W)
{
	if (!W || !W->A)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	return OK_SCAN_ERR( W->operator_scale(W->A, kOne / W->normA) );
}

ok_status pogs_abstract_save_work(pogs_abstract_solver_private_data * data,
	pogs_abstract_solver_flags * flags, const pogs_abstract_work * W)
{
	return OK_SCAN_ERR( OPTKIT_ERROR_NOT_IMPLEMENTED );
	// ok_status err = OPTKIT_SUCCESS;
	// OK_CHECK_ERR( err, matrix_memcpy_am(data->A_equil, M->A, flags->ord) );
	// #ifndef OPTKIT_INDIRECT
	// OK_CHECK_ERR( err, matrix_memcpy_am(data->ATA_cholesky, M->P->L,
	// 	flags->ord) );
	// #endif
	// OK_CHECK_ERR( err, vector_memcpy_av(data->d, M->d, 1) );
	// OK_CHECK_ERR( err, vector_memcpy_av(data->e, M->e, 1) );
	// return err;
}

ok_status pogs_abstract_load_work(pogs_abstract_work * W,
	const pogs_abstract_solver_private_data * data,
	const pogs_abstract_solver_flags * flags)
{
	return OK_SCAN_ERR( OPTKIT_ERROR_NOT_IMPLEMENTED );
	// ok_status err = OPTKIT_SUCCESS;
	// OK_CHECK_ERR( err, matrix_memcpy_ma(M->A, data->A_equil, flags->ord) );
	// #ifndef OPTKIT_INDIRECT
	// OK_CHECK_ERR( err, matrix_memcpy_ma(M->P->L, data->ATA_cholesky,
	// 	flags->ord) );
	// #endif
	// OK_CHECK_ERR( err, vector_memcpy_va(M->d, data->d, 1) );
	// OK_CHECK_ERR( err, vector_memcpy_va(M->e, data->e, 1) );
	// return err;
}

abstract_operator * pogs_dense_operator_gen(const ok_float * A, size_t m, size_t n,
	enum CBLAS_ORDER order)
{
	ok_status err = OPTKIT_SUCCESS;
	abstract_operator * o = OK_NULL;
	matrix * A_mat = OK_NULL;

	if (!A) {
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	} else  {
		ok_alloc(A_mat, sizeof(*A_mat));
		OK_CHECK_ERR( err, matrix_calloc(A_mat, m, n, order) );
		OK_CHECK_ERR( err, matrix_memcpy_ma(A_mat, A, order) );
		if (!err)
			o = dense_operator_alloc(A_mat);
	}
	return o;
}

abstract_operator * pogs_sparse_operator_gen(const ok_float * val,
	const ok_int * ind, const ok_int * ptr, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order)
{
	void * handle = OK_NULL;
	abstract_operator * o = OK_NULL;
	sp_matrix * A = OK_NULL;
	ok_status err = OPTKIT_SUCCESS;

	if (!val || !ind || !ptr)
		err = OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	else {
		ok_alloc(A, sizeof(*A));
		OK_CHECK_ERR( err,
			sp_matrix_calloc(A, m, n, nnz, order) );
		OK_CHECK_ERR( err,
			sp_make_handle(&handle) );
		OK_CHECK_ERR( err,
			sp_matrix_memcpy_ma(handle, A, val, ind, ptr) );
		OK_CHECK_ERR( err,
			sp_destroy_handle(handle) );
		if (!err)
			o = sparse_operator_alloc(A);
	}
	return o;
}

ok_status pogs_dense_operator_free(abstract_operator * A)
{
	OK_CHECK_OPERATOR(A);
	matrix * A_mat = dense_operator_get_matrix_pointer(A);
	ok_status err = A->free(A->data);
	OK_MAX_ERR( err, matrix_free(A_mat) );
	ok_free(A_mat);
	ok_free(A);
	return err;
}

ok_status pogs_sparse_operator_free(abstract_operator * A)
{
	OK_CHECK_OPERATOR(A);
	sp_matrix * A_mat = sparse_operator_get_matrix_pointer(A);
	ok_status err = A->free(A->data);
	OK_MAX_ERR( err, sp_matrix_free(A_mat) );
	ok_free(A_mat);
	ok_free(A);
	return err;
}

#ifdef __cplusplus
}
#endif

#endif /* OPTKIT_POGS_IMPL_ABSTRACT_H_ */