#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

inline ok_float __matrix_get_colmajor(const matrix *A, size_t i, size_t j)
{
	return A->data[i + j * A->ld];
}

inline ok_float __matrix_get_rowmajor(const matrix *A, size_t i, size_t j)
{
	return A->data[i * A->ld + j];
}

inline void __matrix_set_rowmajor(matrix *A, size_t i, size_t j, ok_float x)
{
	A->data[i * A->ld + j] = x;
}

inline void __matrix_set_colmajor(matrix *A, size_t i, size_t j, ok_float x)
{
	A->data[i + j * A->ld] = x;
}

/* Non-Block Cholesky. */
static ok_status __linalg_cholesky_decomp_noblk(void *linalg_handle, matrix *A,
	int silence_domain_err)
{
	ok_status err = OPTKIT_SUCCESS;
	ok_float l11;
	matrix l21, a22;
	size_t n = A->size1, i;

	for (i = 0; i < n && !err; ++i) {
		/* L11 = sqrt(A11) */
		l11 = A->data[i + i * A->ld];
		if (l11 < 0) {
			if (silence_domain_err)
				return OPTKIT_ERROR_DOMAIN;
			else
				return OK_SCAN_ERR( OPTKIT_ERROR_DOMAIN );
		}
		if (l11 == 0) {
			if (silence_domain_err)
				return OPTKIT_ERROR_DIVIDE_BY_ZERO;
			else
				return OK_SCAN_ERR( OPTKIT_ERROR_DIVIDE_BY_ZERO );
		}

		l11 = MATH(sqrt)(l11);
		A->data[i + i * A->ld] = l11;

		if (i + 1 == n)
			break;

		/* L21 = A21 / L11 */
		l21.data = OK_NULL;
		OK_CHECK_ERR( err, matrix_submatrix(&l21, A, i + 1, i,
			n - i - 1, 1) );
		OK_CHECK_ERR( err, matrix_scale(&l21, kOne / l11) );

		/* A22 -= L12*L12'*/
		a22.data = OK_NULL;
		OK_CHECK_ERR( err, matrix_submatrix(&a22, A, i + 1, i + 1,
			n - i - 1, n - i - 1) );
		OK_CHECK_ERR( err, blas_syrk(linalg_handle, CblasLower,
			CblasNoTrans, -kOne, &l21, kOne, &a22) );
	}
	return err;
}

/*
 * Block Cholesky.
 *   l11 l11^ok_float = a11
 *   l21 = a21 l11^(-T)
 *   a22 = a22 - l21 l21^T
 *
 * Stores result in Lower triangular part.
 */
ok_status linalg_cholesky_decomp_flagged(void *linalg_handle, matrix *A,
	int silence_domain_err)
{
	OK_CHECK_MATRIX(A);

	ok_status err = OPTKIT_SUCCESS;
	matrix L11, L21, A22;
	size_t n = A->size1, blk_dim, i, n11;

	L11.data = OK_NULL;
	L21.data = OK_NULL;
	A22.data = OK_NULL;

	/* block dimension borrowed from Eigen. */
	blk_dim = ((n / 128) * 16) < 8 ? (n / 128) * 16 : 8;
	blk_dim = blk_dim > 128 ? blk_dim : 128;

	/* check A square */
	if (A->size1 != A->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	for (i = 0; i < n; i += blk_dim) {
		n11 = blk_dim < n - i ? blk_dim : n - i;

		/* L11 = chol(A11) */
		L11.data = OK_NULL;
		OK_CHECK_ERR( err, matrix_submatrix(&L11, A, i, i, n11, n11) );
		if (!err)
			err = __linalg_cholesky_decomp_noblk(
				linalg_handle, &L11, silence_domain_err);
		if (!silence_domain_err)
			OK_SCAN_ERR(err);

		if (i + blk_dim >= n)
			break;

		/* L21 = A21 L21^-ok_float */
		L21.data = OK_NULL;
		OK_CHECK_ERR( err, matrix_submatrix(&L21, A, i + n11, i,
			n - i - n11, n11) );
		OK_CHECK_ERR( err, blas_trsm(linalg_handle, CblasRight,
			CblasLower, CblasTrans, CblasNonUnit, kOne, &L11,
			&L21) );

		/* A22 -= L21*L21^ok_float */
		A22.data =OK_NULL;
		OK_CHECK_ERR( err, matrix_submatrix(&A22, A, i + blk_dim,
			                       i + blk_dim, n - i - blk_dim,
			                       n - i - blk_dim) );

		OK_CHECK_ERR( err, blas_syrk(linalg_handle, CblasLower,
			CblasNoTrans, -kOne, &L21, kOne, &A22) );
	}
	return err;
}

ok_status linalg_cholesky_decomp(void *linalg_handle, matrix *A)
{
	return linalg_cholesky_decomp_flagged(linalg_handle, A, 0);
}

/* Cholesky solve */
ok_status linalg_cholesky_svx(void *linalg_handle, const matrix *L, vector *x)
{
	OK_RETURNIF_ERR( blas_trsv(linalg_handle, CblasLower, CblasNoTrans,
		CblasNonUnit, L, x) );
	return blas_trsv(linalg_handle, CblasLower, CblasTrans, CblasNonUnit, L,
		x);
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
ok_status linalg_matrix_row_squares(const enum CBLAS_TRANSPOSE t,
	const matrix *A, vector * v)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);

	size_t k, nvecs = (t == CblasTrans) ? A->size2 : A->size1;
	int vecsize = (t == CblasTrans) ? (int) A->size1 : (int) A->size2;
	size_t ptrstride = ((t == CblasTrans) == (A->order == CblasRowMajor))
		? (int) 1 : A->ld;
	int stride = ((t == CblasTrans) == (A->order == CblasRowMajor)) ?
		(int) A->ld : 1;

	/* check size == v->size */
	if (v->size != nvecs)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (k = 0; k < nvecs; ++k) {
		v->data[k * v->stride] = CBLAS(dot)(vecsize,
			A->data + k * ptrstride, stride,
			A->data + k * ptrstride, stride);
	}
	return OPTKIT_SUCCESS;
}

/*
 * if operation == OkTransformScale:
 * 	if side == CblasLeft, set
 * 		A = diag(v) * A
 *	else, set
 *		A = A * diag(v)
 * if operation == OkTransformAdd:
 * 	if side == CblasLeft, perform
 * 		A += v1^T
 *	else, set
 *		A += 1v^T
 */
ok_status linalg_matrix_broadcast_vector(matrix *A, const vector *v,
	const enum OPTKIT_TRANSFORM operation, const enum CBLAS_SIDE side)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);

	size_t k, nvecs = (side == CblasLeft) ? A->size2 : A->size1;
	size_t ptrstride = ((side == CblasLeft) == (A->order == CblasRowMajor))
		? 1 : A->ld;

	size_t stride = ((side == CblasLeft) == (A->order == CblasRowMajor)) ?
		A->ld : 1;

	/* check size == v->size */
	if (((side == CblasLeft) && (v->size != A->size1)) ||
		((side == CblasRight) && (v->size != A->size2)))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	switch (operation) {
	case OkTransformScale :
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (k = 0; k < v->size; ++k)
			CBLAS(scal)( (int) nvecs, v->data[k * v->stride],
				A->data + k * stride, (int) ptrstride);
		break;
	case OkTransformAdd :
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (k = 0; k < nvecs; ++k)
			CBLAS(axpy)((int) v->size, kOne, v->data,
				(int) v->stride, A->data + k * ptrstride,
				(int) stride);
		break;
	default :
		return OK_SCAN_ERR( OPTKIT_ERROR_DOMAIN );
	}
	return OPTKIT_SUCCESS;
}

ok_status linalg_matrix_reduce_indmin(indvector *indices, vector *minima,
	const matrix *A, const enum CBLAS_SIDE side)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(indices);
	OK_CHECK_VECTOR(minima);

	const int reduce_by_row = (side == CblasRight);
	const int rowmajor = (A->order == CblasRowMajor);
	size_t output_dim = (reduce_by_row) ? A->size1 : A->size2;
	size_t reduced_dim = (reduce_by_row) ? A->size2 : A->size1;
	size_t stride_k = (reduce_by_row == rowmajor) ? A->ld : 1;
	size_t stride_i = (reduce_by_row == rowmajor) ? 1 : A->ld;
	size_t k, idx;
	ok_float *min_ = minima->data, *A_ = A->data;
	size_t *ind_ = indices->data;

	if (output_dim != minima->size || indices->size != minima->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (k = 0; k < output_dim; ++k) {
		min_[k * minima->stride] = OK_FLOAT_MAX;
		for (idx = 0; idx < reduced_dim; ++idx)
			if ( A_[k * stride_k + idx * stride_i] <
				min_[k * minima->stride] ) {
				min_[k * minima->stride] = A_[k * stride_k +
					idx * stride_i];
				ind_[k * indices->stride] = idx;
			}
	}
	return OPTKIT_SUCCESS;
}

static ok_status __matrix_extrema(vector *extrema, const matrix *A,
	const enum CBLAS_SIDE side, const int minima)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(extrema);

	const int reduce_by_row = (side == CblasRight);
	const int rowmajor = (A->order == CblasRowMajor);
	size_t output_dim = (reduce_by_row) ? A->size1 : A->size2;
	size_t reduced_dim = (reduce_by_row) ? A->size2 : A->size1;
	size_t stride_k = (reduce_by_row == rowmajor) ? A->ld : 1;
	size_t stride_idx = (reduce_by_row == rowmajor) ? 1 : A->ld;
	size_t k, idx;

	if (output_dim != extrema->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if (minima)
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (k = 0; k < output_dim; ++k) {
			extrema->data[k * extrema->stride] = OK_FLOAT_MAX;
			for (idx = 0; idx < reduced_dim; ++idx)
				extrema->data[k * extrema->stride] = MATH(fmin)(
					extrema->data[k * extrema->stride],
					A->data[k * stride_k + idx * stride_idx]
					);
		}
	else
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (k = 0; k < output_dim; ++k) {
			extrema->data[k * extrema->stride] = -OK_FLOAT_MAX;
			for (idx = 0; idx < reduced_dim; ++idx)
				extrema->data[k * extrema->stride] = MATH(fmax)(
					extrema->data[k * extrema->stride],
					A->data[k * stride_k + idx * stride_idx]
					);
		}
	return OPTKIT_SUCCESS;
}

ok_status linalg_matrix_reduce_min(vector *minima, const matrix *A,
	const enum CBLAS_SIDE side)
{
	return __matrix_extrema(minima, A, side, 1);
}

ok_status linalg_matrix_reduce_max(vector *maxima, const matrix *A,
	const enum CBLAS_SIDE side)
{
	return __matrix_extrema(maxima, A, side, 0);
}

#ifdef __cplusplus
}
#endif
