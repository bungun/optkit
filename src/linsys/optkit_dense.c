#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

void denselib_version(int * maj, int * min, int * change, int * status)
{
	* maj = OPTKIT_VERSION_MAJOR;
	* min = OPTKIT_VERSION_MINOR;
	* change = OPTKIT_VERSION_CHANGE;
	* status = (int) OPTKIT_VERSION_STATUS;
}

#ifdef __cplusplus
}
#endif

/*
 * LINEAR ALGEBRA routines
 * =======================
 */

/* Non-Block Cholesky. */
template<typename T>
void __linalg_cholesky_decomp_noblk(void * linalg_handle, matrix<T> *A) {
	T l11;
	matrix l21, a22;
	size_t n = A->size1, i;

	/* get order-specific matrix getter/setter */
	void (* mset)(matrix<T> * M, size_t i, size_t j, ok_float x) =
		(A->order == CblasRowMajor) ?
		__matrix_set_rowmajor : __matrix_set_colmajor;
	T (* mget)(const matrix<T> * M, size_t i, size_t j) =
		(A->order == CblasRowMajor) ?
		__matrix_get_rowmajor : __matrix_get_colmajor;

	l21= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};
	a22= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};

	for (i = 0; i < n; ++i) {
		/* L11 = sqrt(A11) */
		l11 = static_cast<T>(MATH(sqrt)(mget(A, i, i)));
		mset(A, i, i, l11);

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
 *   l11 l11^T = a11
 *   l21 = a21 l11^(-T)
 *   a22 = a22 - l21 l21^T
 *
 * Stores result in Lower triangular part.
 */
template<typename T>
void linalg_cholesky_decomp(void * linalg_handle, matrix<T> * A)
{
	matrix L11, L21, A22;
	size_t n = A->size1, blk_dim, i, n11;

	L11= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};
	L21= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};
	A22= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};

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

                /* L21 = A21 L21^-T */
		matrix_submatrix(&L21, A, i + n11, i, n - i - n11, n11);
		blas_trsm(linalg_handle, CblasRight, CblasLower, CblasTrans,
			CblasNonUnit, kOne, &L11, &L21);

		/* A22 -= L21*L21^T */
		matrix_submatrix(&A22, A, i + blk_dim, i + blk_dim,
			n - i - blk_dim, n - i - blk_dim);

		blas_syrk(linalg_handle, CblasLower, CblasNoTrans, -kOne, &L21,
			kOne, &A22);
	}
}

/* Cholesky solve */
template<typename T>
void linalg_cholesky_svx(void * linalg_handle, const matrix<T> * L, vector_<T> * x)
{
	blas_trsv(linalg_handle, CblasLower, CblasNoTrans, CblasNonUnit, L, x);
	blas_trsv(linalg_handle, CblasLower, CblasTrans, CblasNonUnit, L, x);
}

/* device reset */
ok_status ok_device_reset()
{
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
