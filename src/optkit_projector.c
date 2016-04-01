#include "optkit_projector.h"

#ifdef __cplusplus
extern "C" {
#endif

const int kQuietCG = 1;
const int kItersCG = 100;

void projectorlib_version(int * maj, int * min, int * change, int * status)
{
	* maj = OPTKIT_VERSION_MAJOR;
	* min = OPTKIT_VERSION_MINOR;
	* change = OPTKIT_VERSION_CHANGE;
	* status = (int) OPTKIT_VERSION_STATUS;
}

int projector_normalization(projector * P)
{
	dense_direct_projector * Pdd = OK_NULL;
	indirect_projector_generic * Pi = OK_NULL;

	if (P->kind == OkProjectorDenseDirect) {
		Pdd = (dense_direct_projector *) P->data;
		return Pdd->normalized;
	} else if (P->kind == OkProjectorIndirect) {
		Pi = (indirect_projector_generic *) P->data;
		return Pi->normalized;
	} else {
		printf("%s", "projector normalizetion status unretrievable");
		printf("%s\n", "returning 1");
		return 0;
	}
}

ok_float projector_get_norm(projector * P)
{
	dense_direct_projector * Pdd = OK_NULL;
	indirect_projector_generic * Pi = OK_NULL;

	if (P->kind == OkProjectorDenseDirect) {
		Pdd = (dense_direct_projector *) P->data;
		return Pdd->normA;
	} else if (P->kind == OkProjectorIndirect) {
		Pi = (indirect_projector_generic *) P->data;
		return Pi->normA;
	} else {
		printf("%s\n", "projector norm unretrievable, returning 1");
		return kOne;
	}
}


/* Direct Projector methods */
void direct_projector_alloc(direct_projector * P, matrix * A)
{
	size_t mindim = (A->size1 < A->size2) ? A->size1 : A->size2;

	P->A = A;
	P->L = (matrix *) malloc( sizeof(matrix) );
	matrix_calloc(P->L, mindim, mindim, A->order);
	P->skinny = (uint) mindim == A->size2;
	P->normalized = 0;
}

void direct_projector_free(direct_projector * P)
{
	matrix_free(P->L);
	P->A = OK_NULL;
}

void direct_projector_initialize(void * linalg_handle, direct_projector * P,
	int normalize)
{
	vector diag = (vector){0, 0, OK_NULL};
	ok_float mean_diag = kZero;

	if (P->skinny)
		blas_gemm(linalg_handle, CblasTrans, CblasNoTrans, kOne, P->A,
			P->A, kZero, P->L);
	else
		blas_gemm(linalg_handle, CblasNoTrans, CblasTrans, kOne, P->A,
			P->A, kZero, P->L);

	matrix_diagonal(&diag, P->L);
	mean_diag = blas_asum(linalg_handle, &diag) / (ok_float) P->L->size1;
	P->normA =  MATH(sqrt)(mean_diag);

	if (normalize) {
		matrix_scale(P->L, kOne / mean_diag);
		matrix_scale(P->A, kOne / P->normA);
	}
	P->normalized = normalize;


	vector_add_constant(&diag, kOne);
	linalg_cholesky_decomp(linalg_handle, P->L);
}

void direct_projector_project(void * linalg_handle, direct_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out)
{
	if (P->skinny) {
 		vector_memcpy_vv(x_out, x_in);
		blas_gemv(linalg_handle, CblasTrans, kOne, P->A, y_in, kOne,
			x_out);
		linalg_cholesky_svx(linalg_handle, P->L, x_out);
		blas_gemv(linalg_handle, CblasNoTrans, kOne, P->A, x_out, kZero,
			y_out);

	} else {
		vector_memcpy_vv(y_out, y_in);
		blas_gemv(linalg_handle, CblasNoTrans, kOne, P->A, x_in, -kOne,
			y_out);
		linalg_cholesky_svx(linalg_handle, P->L, y_out);
		blas_gemv(linalg_handle, CblasTrans, -kOne, P->A, y_out, kZero,
			x_out);
		blas_axpy(linalg_handle, kOne, y_in, y_out);
		blas_axpy(linalg_handle, kOne, x_in, x_out);
	}
}

/* Indirect Projector methods */
void indirect_projector_alloc(indirect_projector * P, operator * A)
{
	P->A = A;
	P->cgls_work = cgls_init(A->size1, A->size2);
}

void indirect_projector_initialize(void * linalg_handle, indirect_projector * P,
	int normalize)
{
	/* no-op*/
	return;
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
void indirect_projector_project(void * linalg_handle, indirect_projector * P,
	vector * x_in, vector * y_in, vector * x_out, vector * y_out)
{
	/* y_out = y_in - Ax_in */
	vector_memcpy_vv(y_out, y_in);
	P->A->fused_apply(P->A->data, -kOne, x_in, kOne, y_out);

	/* Minimize ||Ax_out - (y_in - Ax_in) ||_2 + ||x_out||_2*/
	cgls_solve(P->cgls_work, P->A, y_out, x_out, kOne, 1e-12, 100, 1);

	/* x_out += x0 */
	blas_axpy(linalg_handle, kOne, x_in, x_out);

	/* y_out = Ax_ouy */
	P->A->apply(P->A->data, x_out, y_out);
}

void indirect_projector_free(indirect_projector * P)
{
	P->A = OK_NULL;
	cgls_finish(P->cgls_work);
	P->cgls_work = OK_NULL;
}

void * dense_direct_projector_data_alloc(matrix * A)
{
	size_t mindim = (A->size1 < A->size2) ? A->size1 : A->size2;

	dense_direct_projector * P;
	P = malloc(sizeof(*P));
	P->A = A;
	P->L = malloc(sizeof(matrix));
	matrix_calloc(P->L, mindim, mindim, A->order);
	P->normA = kOne;
	P->skinny = (uint) mindim == A->size2;
	P->normalized = 0;
	blas_make_handle(&(P->linalg_handle));
	return (void *) P;
}

void dense_direct_projector_data_free(void * data)
{
	dense_direct_projector * P = (dense_direct_projector *) data;
	blas_destroy_handle(P->linalg_handle);
	matrix_free(P->L);
	ok_free(P->L);
	ok_free(P);
}

void dense_direct_projector_initialize(void * data, int normalize)
{
	dense_direct_projector * P = (dense_direct_projector *) data;
	direct_projector DP = (direct_projector) {OK_NULL, OK_NULL, kOne, 0, 0};
	DP.A = P->A;
	DP.L = P->L;
	DP.normA = P->normA;
	DP.skinny = P->skinny;
	DP.normalized = P->normalized;
	direct_projector_initialize(P->linalg_handle, &DP, normalize);
	P->normalized = DP.normalized;
	P->normA = DP.normA;
}

void dense_direct_projector_project(void * data, vector * x_in, vector * y_in,
	vector * x_out, vector * y_out, ok_float tol)
{
	dense_direct_projector * P = (dense_direct_projector *) data;
	direct_projector DP = (direct_projector) {OK_NULL, OK_NULL, kOne, 0, 0};
	DP.A = P->A;
	DP.L = P->L;
	DP.normA = P->normA;
	DP.skinny = P->skinny;
	DP.normalized = P->normalized;
	direct_projector_project(P->linalg_handle, &DP, x_in, y_in, x_out,
		y_out);
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
	return P;
}

void * indirect_projector_data_alloc(operator * A)
{
	indirect_projector_generic * P;
	P = malloc(sizeof(*P));
	P->A = A;
	P->cgls_work = cgls_init(A->size1, A->size2);
	P->normA = kOne;
	P->normalized = 0;
	blas_make_handle(&(P->linalg_handle));
	return (void *) P;
}

void indirect_projector_data_free(void * data)
{
	indirect_projector_generic * P = (indirect_projector_generic *) data;
	cgls_finish(P->cgls_work);
	blas_destroy_handle(P->linalg_handle);
	ok_free(P);
}

void indirect_projector_g_initialize(void * data, int normalize)
{
	/* estimate norm */
	/* normalize */
	return;
}

void indirect_projector_g_project(void * data, vector * x_in, vector * y_in,
	vector * x_out, vector * y_out, ok_float tol)
{
	indirect_projector_generic * P = (indirect_projector_generic *) data;

	/* y_out = y_in - Ax_in */
	vector_memcpy_vv(y_out, y_in);
	P->A->fused_apply(P->A->data, -kOne, x_in, kOne, y_out);

	/* Minimize ||Ax_out - (y_in - Ax_in) ||_2 + ||x_out||_2*/
	cgls_solve(P->cgls_work, P->A, y_out, x_out, kOne, tol, kItersCG,
		kQuietCG);

	/* x_out += x0 */
	blas_axpy(P->linalg_handle, kOne, x_in, x_out);

	/* y_out = Ax_ouy */
	P->A->apply(P->A->data, x_out, y_out);
}

projector * indirect_projector_generic_alloc(operator * A)
{
	projector * P = OK_NULL;
	P = malloc(sizeof(*P));
	P->kind = OkProjectorIndirect;
	P->size1 = A->size1;
	P->size2 = A->size2;
	P->data = indirect_projector_data_alloc(A);
	P->initialize = indirect_projector_g_initialize;
	P->project = indirect_projector_g_project;
	P->free = indirect_projector_data_free;
	return P;
}


#ifdef __cplusplus
}
#endif
