#include "optkit_projector.h"

#ifdef __cplusplus
extern "C" {
#endif

const uint kQuietCG = 1;
const uint kItersCG = 100;

ok_status projector_normalization(projector * P, int * normalized)
{
	OK_CHECK_PROJECTOR(P);
	OK_CHECK_PTR(normalized);

	ok_status err = OPTKIT_SUCCESS;
	dense_direct_projector * Pdd = OK_NULL;
	indirect_projector_generic * Pi = OK_NULL;

	if (P->kind == OkProjectorDenseDirect) {
		Pdd = (dense_direct_projector *) P->data;
		*normalized =  Pdd->normalized;
	} else if (P->kind == OkProjectorIndirect) {
		Pi = (indirect_projector_generic *) P->data;
		*normalized = Pi->normalized;
	} else {
		printf("%s", "projector normalizetion status unretrievable ");
		printf("%s\n", "setting *normalized = 1");
		*normalized = -1;
		err = OK_SCAN_ERR( OPTKIT_ERROR_DOMAIN );
	}
	return err;
}

ok_status projector_get_norm(projector * P, ok_float * norm)
{
	OK_CHECK_PROJECTOR(P);
	OK_CHECK_PTR(norm);

	ok_status err = OPTKIT_SUCCESS;
	dense_direct_projector * Pdd = OK_NULL;
	indirect_projector_generic * Pi = OK_NULL;

	if (P->kind == OkProjectorDenseDirect) {
		Pdd = (dense_direct_projector *) P->data;
		*norm = Pdd->normA;
	} else if (P->kind == OkProjectorIndirect) {
		Pi = (indirect_projector_generic *) P->data;
		*norm =  Pi->normA;
	} else {
		printf("%s", "projector norm unretrievable, ");
		printf("%s\n", "setting *norm = 1.0");
		*norm = kOne;
		err = OK_SCAN_ERR( OPTKIT_ERROR_DOMAIN );
	}
	return err;
}


/* Direct Projector methods */
ok_status direct_projector_alloc(direct_projector * P, matrix * A)
{
	OK_CHECK_PTR(P);
	OK_CHECK_MATRIX(A);

	if (P->A)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	ok_status err = OPTKIT_SUCCESS;
	size_t mindim = (A->size1 < A->size2) ? A->size1 : A->size2;

	P->A = A;
	ok_alloc(P->L, sizeof(*P->L));
	err = OK_SCAN_ERR( matrix_calloc(P->L, mindim, mindim, A->order) );
	P->skinny = (uint) mindim == A->size2;
	P->normalized = 0;
	if (err)
		OK_MAX_ERR( err, direct_projector_free(P) );
	return err;
}

ok_status direct_projector_free(direct_projector * P)
{
	OK_CHECK_PTR(P);
	ok_status err = OK_SCAN_ERR( matrix_free(P->L) );
	ok_free(P->L);
	P->A = OK_NULL;
	return err;
}

ok_status direct_projector_initialize(void * linalg_handle,
	direct_projector * P, int normalize)
{
	if (!P || !P->A || !P->L)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	vector diag;
	ok_float mean_diag = kZero;

	diag.data = OK_NULL;
	diag.size = 0;
	diag.stride = 0;

	if (P->skinny)
		blas_gemm(linalg_handle, CblasTrans, CblasNoTrans, kOne, P->A,
			P->A, kZero, P->L);
	else
		blas_gemm(linalg_handle, CblasNoTrans, CblasTrans, kOne, P->A,
			P->A, kZero, P->L);

	matrix_diagonal(&diag, P->L);
	blas_asum(linalg_handle, &diag, &mean_diag);
	mean_diag /= (ok_float) P->L->size1;
	P->normA =  MATH(sqrt)(mean_diag);

	if (mean_diag == 0)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIVIDE_BY_ZERO );

	if (normalize) {
		matrix_scale(P->L, kOne / mean_diag);
		matrix_scale(P->A, kOne / P->normA);
	}
	P->normalized = normalize;

	vector_add_constant(&diag, kOne);
	return OK_SCAN_ERR( linalg_cholesky_decomp(linalg_handle, P->L) );
}

ok_status direct_projector_project(void * linalg_handle, direct_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out)
{
	if (!P || !P->A || !P->L)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_VECTOR(x_in);
	OK_CHECK_VECTOR(y_in);
	OK_CHECK_VECTOR(x_out);
	OK_CHECK_VECTOR(y_out);

	if (P->skinny) {
 		OK_RETURNIF_ERR(
 			vector_memcpy_vv(x_out, x_in) );
		OK_RETURNIF_ERR(
			blas_gemv(linalg_handle, CblasTrans, kOne, P->A, y_in,
				kOne, x_out) );
		OK_RETURNIF_ERR(
			linalg_cholesky_svx(linalg_handle, P->L, x_out) );
		return OK_SCAN_ERR(
			blas_gemv(linalg_handle, CblasNoTrans, kOne, P->A,
				x_out, kZero, y_out) );

	} else {
		OK_RETURNIF_ERR(
			vector_memcpy_vv(y_out, y_in) );
		OK_RETURNIF_ERR(
			blas_gemv(linalg_handle, CblasNoTrans, kOne, P->A, x_in,
				-kOne, y_out) );
		OK_RETURNIF_ERR(
			linalg_cholesky_svx(linalg_handle, P->L, y_out) );
		OK_RETURNIF_ERR(
			blas_gemv(linalg_handle, CblasTrans, -kOne, P->A, y_out,
				kZero, x_out) );
		OK_RETURNIF_ERR(
			blas_axpy(linalg_handle, kOne, y_in, y_out) );
		return OK_SCAN_ERR(
			blas_axpy(linalg_handle, kOne, x_in, x_out) );
	}
}

#ifndef OPTKIT_NO_INDIRECT_PROJECTOR
/* Indirect Projector methods */
ok_status indirect_projector_alloc(indirect_projector * P, operator * A)
{
	OK_CHECK_PTR(P);
	OK_CHECK_OPERATOR(A);
	if (P->A)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	P->A = A;
	P->cgls_work = cgls_init(A->size1, A->size2);
	if (!P->cgls_work)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	return OPTKIT_SUCCESS;
}

ok_status indirect_projector_initialize(void * linalg_handle,
	indirect_projector * P, int normalize)
{
	/* no-op*/
	if (!P || !P->A || !P->cgls_work)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	else
		return OPTKIT_SUCCESS;
}

/*
 * Projects (x_in, y_in) onto y = Ax by setting
 *
 *	x_out = x_in + argmin_x || Ax - (y_in - Ax_in) ||^2 + ||x||^2,
 *	y_out = Ax_out.
 *
 * Note that the KKT optimality conditions for the minimization problem are
 *
 *	AᵀA(x^\star + x_in) - Aᵀy_in + x^\star == 0,
 *
 * which yields
 *
 *	(I + AᵀA)x^\star = (Aᵀy_in - AᵀAx_in)
 *	x^\star = (I + AᵀA)Aᵀy_in - (I + AᵀA)AᵀAx_in
 *
 * adding x_in to this optimal solution,
 *
 *	x_out 	= (I + AᵀA)⁻¹Aᵀy_in - (I + AᵀA)⁻¹AᵀAx + x_in.
 *
 * Adding 0 to the LHS and (I + AᵀA)⁻¹(x_in - x_in) to the RHS, we obtain
 *
 *	x_out	= (I + AᵀA)⁻¹(x_in + Aᵀy_in).
 *
 * This corresponds exactly to the optimality conditions for the problem
 *
 *	minimize ||x - x_in ||^2 + ||Ax - y_in||^2,
 *
 * hence we obtain the desired projection.
 *
 */
ok_status indirect_projector_project(void * linalg_handle,
	indirect_projector * P, vector * x_in, vector * y_in, vector * x_out,
	vector * y_out)
{
	OK_CHECK_PTR(P);
	OK_CHECK_VECTOR(x_in);
	OK_CHECK_VECTOR(y_in);
	OK_CHECK_VECTOR(x_out);
	OK_CHECK_VECTOR(y_out);

	/* y_out = y_in - Ax_in */
	OK_RETURNIF_ERR(
		vector_memcpy_vv(y_out, y_in) );
	printf("P %p\n", P);
	printf("P.A %p\n", P->A);
	printf("P.A.apply %p\n", P->A->apply);
	printf("P.A.adjoint %p\n", P->A->adjoint);
	printf("P.A.fused_apply %p\n", P->A->fused_apply);
	printf("P.A.fused_adjoint %p\n", P->A->fused_adjoint);
	printf("P.A.free %p\n", P->A->free);
	printf("P.A.data %p\n", P->A->data);
	printf("xin %p\n", x_in->data);
	printf("yin %p\n", y_in->data);
	printf("xout %p\n", x_out->data);
	printf("yout %p\n", y_out->data);
	OK_RETURNIF_ERR(
		P->A->fused_apply(P->A->data, -kOne, x_in, kOne, y_out) );

	/* Minimize ||Ax_out - (y_in - Ax_in) ||_2 + ||x_out||_2*/
	OK_RETURNIF_ERR(
		cgls_solve(P->cgls_work, P->A, y_out, x_out, kOne,
			(ok_float) 1e-12, 100, 1, &P->flag) );

	/* x_out += x0 */
	OK_RETURNIF_ERR(
		blas_axpy(linalg_handle, kOne, x_in, x_out) );

	/* y_out = Ax_ouy */
	printf("P %p\n", P);
	printf("P.A %p\n", P->A);
	printf("P.A.apply %p\n", P->A->apply);
	printf("P.A.adjoint %p\n", P->A->adjoint);
	printf("P.A.fused_apply %p\n", P->A->fused_apply);
	printf("P.A.fused_adjoint %p\n", P->A->fused_adjoint);
	printf("P.A.free %p\n", P->A->free);
	printf("P.A.data %p\n", P->A->data);
	printf("xin %p\n", x_in->data);
	printf("yin %p\n", y_in->data);
	printf("xout %p\n", x_out->data);
	printf("yout %p\n", y_out->data);

	return OK_SCAN_ERR(
		P->A->apply(P->A->data, x_out, y_out) );
}

ok_status indirect_projector_free(indirect_projector * P)
{
	OK_CHECK_PTR(P);
	ok_status err = OK_SCAN_ERR( cgls_finish(P->cgls_work) );
	P->cgls_work = OK_NULL;
	P->A = OK_NULL;
	return err;
}
#endif /* ndef OPTKIT_NO_INDIRECT_PROJECTOR */

void * dense_direct_projector_data_alloc(matrix * A)
{
	ok_status err = OPTKIT_SUCCESS;
	size_t mindim;
	dense_direct_projector * P = OK_NULL;
	if (A && A->data) {
		mindim = (A->size1 < A->size2) ? A->size1 : A->size2;

		ok_alloc(P, sizeof(*P));
		P->A = A;

		ok_alloc(P->L, sizeof(*P->L));
		OK_CHECK_ERR( err,
			matrix_calloc(P->L, mindim, mindim, A->order) );
		P->normA = kOne;
		P->skinny = (uint) mindim == A->size2;
		P->normalized = 0;
		OK_CHECK_ERR( err,
			blas_make_handle(&(P->linalg_handle)) );
		if (err) {
			OK_MAX_ERR( err,
				dense_direct_projector_data_free((void *) P) );
			P = OK_NULL;
		}
	}
	return (void *) P;
}

ok_status dense_direct_projector_data_free(void * data)
{
	OK_CHECK_PTR(data);

	dense_direct_projector * P = (dense_direct_projector *) data;
	ok_status err = OK_SCAN_ERR( blas_destroy_handle(P->linalg_handle) );
	OK_MAX_ERR( err, matrix_free(P->L) );
	ok_free(P->L);
	ok_free(P);
	return err;
}

ok_status dense_direct_projector_initialize(void * data, int normalize)
{
	dense_direct_projector * P = (dense_direct_projector *) data;
	direct_projector DP;
	ok_status err = OPTKIT_SUCCESS;

	if (!P || !P->A || !P->L)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	DP.A = P->A;
	DP.L = P->L;
	DP.normA = P->normA;
	DP.skinny = P->skinny;
	DP.normalized = P->normalized;
	err = OK_SCAN_ERR(
		direct_projector_initialize(P->linalg_handle, &DP, normalize) );
	P->normalized = DP.normalized;
	P->normA = DP.normA;
	return err;
}

ok_status dense_direct_projector_project(void * data, vector * x_in, vector * y_in,
	vector * x_out, vector * y_out, ok_float tol)
{
	dense_direct_projector * P = (dense_direct_projector *) data;
	direct_projector DP;

	if (!P || !P->A || !P->L)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	DP.A = P->A;
	DP.L = P->L;
	DP.normA = P->normA;
	DP.skinny = P->skinny;
	DP.normalized = P->normalized;
	return OK_SCAN_ERR(
		direct_projector_project(P->linalg_handle, &DP, x_in, y_in,
			x_out, y_out) );
}

projector * dense_direct_projector_alloc(matrix * A)
{
	projector * P = OK_NULL;
	P = malloc(sizeof(*P));
	P->kind = OkProjectorDenseDirect;
	P->size1 = A->size1;
	P->size2 = A->size2;
	P->data = dense_direct_projector_data_alloc(A);
	P->initialize = dense_direct_projector_initialize;
	P->project = dense_direct_projector_project;
	P->free = dense_direct_projector_data_free;
	if (!P->data)
		ok_free(P);
	return P;
}

#ifndef OPTKIT_NO_INDIRECT_PROJECTOR
void * indirect_projector_data_alloc(operator * A)
{
	ok_status err;
	indirect_projector_generic * P = OK_NULL;
	ok_alloc(P, sizeof(*P));
	P->A = A;
	P->cgls_work = cgls_init(A->size1, A->size2);
	P->normA = kOne;
	P->normalized = 0;
	err = blas_make_handle(&(P->linalg_handle));
	if (err || !P->A || !P->cgls_work) {
		OK_MAX_ERR( err,
			indirect_projector_data_free((void *) P) );
		P = OK_NULL;
	}

	return (void *) P;
}

ok_status indirect_projector_data_free(void * data)
{
	OK_CHECK_PTR(data);

	indirect_projector_generic * P = (indirect_projector_generic *) data;
	ok_status err = OK_SCAN_ERR( cgls_finish(P->cgls_work) );
	OK_MAX_ERR( err, blas_destroy_handle(P->linalg_handle) );
	ok_free(P);
	return err;
}

/* STUB??? or ok as no-op? */
ok_status indirect_projector_g_initialize(void * data, int normalize)
{
	/* estimate norm */
	/* normalize */
	OK_CHECK_PTR(data);
	return OPTKIT_SUCCESS;

}

ok_status indirect_projector_g_project(void * data, vector * x_in, vector * y_in,
	vector * x_out, vector * y_out, ok_float tol)
{
	indirect_projector_generic * P = (indirect_projector_generic *) data;
	printf("%s\n", "HERE0");
	OK_CHECK_PTR(P);

	printf("%s\n", "HERE1");
	/* y_out = y_in - Ax_in */
	OK_RETURNIF_ERR(
		vector_memcpy_vv(y_out, y_in) );

	printf("%s\n", "HERE2");
	OK_RETURNIF_ERR(
		P->A->fused_apply(P->A->data, -kOne, x_in, kOne, y_out) );

	printf("%s\n", "HERE3");
	/* Minimize ||Ax_out - (y_in - Ax_in) ||_2 + ||x_out||_2*/
	OK_RETURNIF_ERR(
		cgls_solve(P->cgls_work, P->A, y_out, x_out, kOne, tol,
			kItersCG, kQuietCG, &P->flag) );


	printf("%s\n", "HERE4");
	/* x_out += x0 */
	OK_RETURNIF_ERR(
		blas_axpy(P->linalg_handle, kOne, x_in, x_out) );

	printf("%s\n", "HERE5");
	/* y_out = Ax_ouy */
	return OK_SCAN_ERR(
		P->A->apply(P->A->data, x_out, y_out) );
}

projector * indirect_projector_generic_alloc(operator * A)
{
	projector * P = OK_NULL;
	ok_alloc(P, sizeof(*P));
	P->kind = OkProjectorIndirect;
	P->size1 = A->size1;
	P->size2 = A->size2;
	P->data = indirect_projector_data_alloc(A);
	P->initialize = indirect_projector_g_initialize;
	P->project = indirect_projector_g_project;
	P->free = indirect_projector_data_free;
	if (!P->data)
		ok_free(P);
	return P;
}
#endif /* ndef OPTKIT_NO_INDIRECT_PROJECTOR */


#ifdef __cplusplus
}
#endif
