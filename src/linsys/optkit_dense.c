#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

inline ok_float __matrix_get_colmajor(const matrix * A, size_t i, size_t j)
{
	return A->data[i + j * A->ld];
}

inline ok_float __matrix_get_rowmajor(const matrix * A, size_t i, size_t j)
{
	return A->data[i * A->ld + j];
}

inline void __matrix_set_rowmajor(matrix * A, size_t i, size_t j, ok_float x)
{
	A->data[i * A->ld + j] = x;
}

inline void __matrix_set_colmajor(matrix * A, size_t i, size_t j, ok_float x)
{
	A->data[i + j * A->ld] = x;
}

/* Non-Block Cholesky. */
static void __linalg_cholesky_decomp_noblk(void * linalg_handle, matrix *A)
{
	ok_float l11;
	matrix l21, a22;
	size_t n = A->size1, i;

	l21.data = OK_NULL;
	a22.data = OK_NULL;

	for (i = 0; i < n; ++i) {
		/* L11 = sqrt(A11) */
		l11 = MATH(sqrt)(A->data[i + i * A->ld]);
		A->data[i + i * A->ld] = l11;

		if (i + 1 == n)
			break;

		/* L21 = A21 / L11 */
		matrix_submatrix(&l21, A, i + 1, i, n - i - 1, 1);
		matrix_scale(&l21, kOne / l11);

		/* A22 -= L12*L12'*/
		matrix_submatrix(&a22, A, i + 1, i + 1, n - i - 1, n - i - 1);
		blas_syrk(linalg_handle, CblasLower, CblasNoTrans, -kOne, &l21,
			kOne, &a22);
	}
}

/*
 * Block Cholesky.
 *   l11 l11^ok_float = a11
 *   l21 = a21 l11^(-T)
 *   a22 = a22 - l21 l21^T
 *
 * Stores result in Lower triangular part.
 */
void linalg_cholesky_decomp(void * linalg_handle, matrix * A)
{
	matrix L11, L21, A22;
	size_t n = A->size1, blk_dim, i, n11;

	L11.data = OK_NULL;
	L21.data = OK_NULL;
	A22.data = OK_NULL;

	/* block dimension borrowed from Eigen. */
	blk_dim = ((n / 128) * 16) < 8 ? (n / 128) * 16 : 8;
	blk_dim = blk_dim > 128 ? blk_dim : 128;

	for (i = 0; i < n; i += blk_dim) {
		n11 = blk_dim < n - i ? blk_dim : n - i;

		/* L11 = chol(A11) */
		matrix_submatrix(&L11, A, i, i, n11, n11);
		__linalg_cholesky_decomp_noblk(linalg_handle, &L11);

		if (i + blk_dim >= n)
			break;

                /* L21 = A21 L21^-ok_float */
		matrix_submatrix(&L21, A, i + n11, i, n - i - n11, n11);
		blas_trsm(linalg_handle, CblasRight, CblasLower, CblasTrans,
			CblasNonUnit, kOne, &L11, &L21);

		/* A22 -= L21*L21^ok_float */
		matrix_submatrix(&A22, A, i + blk_dim, i + blk_dim,
			n - i - blk_dim, n - i - blk_dim);

		blas_syrk(linalg_handle, CblasLower, CblasNoTrans, -kOne, &L21,
			kOne, &A22);
	}
}

/* Cholesky solve */
void linalg_cholesky_svx(void * linalg_handle, const matrix * L, vector * x)
{
	blas_trsv(linalg_handle, CblasLower, CblasNoTrans, CblasNonUnit, L, x);
	blas_trsv(linalg_handle, CblasLower, CblasTrans, CblasNonUnit, L, x);
}

/*
 * if t == CblasTrans, set
 *
 * 	v_i = a_i'a_i,
 *
 * where a_i is the ith _column_ of A. otherwise, set
 *
 *	v_i = \tilde a_i'\tilde a_i,
 *
 * where \tilde a_i is the ith _row_ of A
 *
 */
void linalg_matrix_row_squares(const enum CBLAS_TRANSPOSE t, const matrix * A,
	vector * v)
{
	size_t k, nvecs = (t == CblasTrans) ? A->size2 : A->size1;
	int vecsize = (t == CblasTrans) ? (int) A->size1 : (int) A->size2;
	size_t ptrstride = ((t == CblasTrans) == (A->order == CblasRowMajor))
		? (int) 1 : A->ld;
	int stride = ((t == CblasTrans) == (A->order == CblasRowMajor)) ?
		(int) A->ld : 1;

	/* check size == v->size */
	if (v->size != nvecs) {
		printf("%s %s\n", "ERROR: linalg_matrix_row_squares()",
			"incompatible dimensions");
		return;
	}

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (k = 0; k < nvecs; ++k) {
		v->data[k * v->stride] = CBLAS(dot)(vecsize,
			A->data + k * ptrstride, stride,
			A->data + k * ptrstride, stride);
	}
}

/*
 * if operation == OkTransformScale:
 * 	if side == CblasLeft, set
 * 		A = A * diag(v)
 *	else, set
 *		A = diag(v) * A
 * if operation == OkTransformAdd:
 * 	if side == CblasLeft, perform
 * 		A += v1^T
 *	else, set
 *		A += 1v^T
 */
void linalg_matrix_broadcast_vector(matrix * A, const vector * v,
	const enum OPTKIT_TRANSFORM operation, const enum CBLAS_SIDE side)
{
	size_t k, nvecs = (side == CblasLeft) ? A->size2 : A->size1;
	size_t ptrstride = ((side == CblasLeft) == (A->order == CblasRowMajor))
		? (int) 1 : A->ld;

	size_t stride = ((side == CblasLeft) == (A->order == CblasRowMajor)) ?
		A->ld : 1;

	/* check size == v->size */
	if (((side == CblasLeft) && (v->size != A->size1)) ||
		((side == CblasRight) && (v->size != A->size2))) {
		printf("%s %s\nA (%u x %u)\nv %u\n",
			"ERROR: linalg_matrix_broadcast_vector()",
			"incompatible dimensions", (uint) A->size1,
			(uint) A->size2, (uint) v->size);
		return;
	}

	if (operation == OkTransformScale) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (k = 0; k < v->size; ++k)
			CBLAS(scal)( (int) nvecs, v->data[k * v->stride],
				A->data + k * stride, (int) ptrstride);
	} else {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (k = 0; k < nvecs; ++k)
			CBLAS(axpy)((int) v->size, kOne, v->data,
				(int) v->stride, A->data + k * ptrstride,
				(int) stride);
	}
}

void linalg_matrix_reduce_indmin(indvector * indices, vector * minima,
	matrix * A, const enum CBLAS_SIDE side)
{
	int reduce_rows = (side == CblasLeft);
	size_t output_dim = (reduce_rows) ? A->size2 : A->size1;
	size_t k;
	vector a;

	void (* matrix_subvec)(vector * row_col, matrix * A, size_t k) =
		(reduce_rows) ? matrix_column : matrix_row;

	for (k = 0; k < output_dim; ++k) {
		matrix_subvec(&a, A, k);
		indices->data[k] = vector_indmin(&a);
		minima->data[k * minima->stride] = a.data[indices->data[k] *
			a.stride];
	}
}

static void __matrix_extrema(vector * extrema, matrix * A,
	const enum CBLAS_SIDE side, const int minima)
{
	int reduce_rows = (side == CblasLeft);
	size_t output_dim = (reduce_rows) ? A->size2 : A->size1;
	size_t k;
	vector a;

	void (* matrix_subvec)(vector * row_col, matrix * A, size_t k) =
		(reduce_rows) ? matrix_column : matrix_row;

	if (minima)
		for (k = 0; k < output_dim; ++k) {
			matrix_subvec(&a, A, k);
			extrema->data[k] = vector_min(&a);
		}
	else
		for (k = 0; k < output_dim; ++k) {
			matrix_subvec(&a, A, k);
			extrema->data[k] = vector_max(&a);
		}
}

void linalg_matrix_reduce_min(vector * minima, matrix * A,
	const enum CBLAS_SIDE side)
{
	__matrix_extrema(minima, A, side, 1);
}

void linalg_matrix_reduce_max(vector * maxima, matrix * A,
	const enum CBLAS_SIDE side)
{
	__matrix_extrema(maxima, A, side, 0);
}

#ifdef __cplusplus
}
#endif
