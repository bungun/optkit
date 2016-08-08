#include "optkit_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

template<typename T, typename I>
ok_status sp_matrix_alloc_(sp_matrix_<T, I> * A, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order)
{
	if (!A)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (A->val || A->ind || A->ptr)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );

	/* Stored forward and adjoint operators */
	A->size1 = m;
	A->size2 = n;
	A->nnz = nnz;
	A->ptrlen = (order == CblasColMajor) ? n + 1 : m + 1;
	A->val = (T *) malloc(2 * nnz * sizeof(T));
	A->ind = (I *) malloc(2 * nnz * sizeof(I));
	A->ptr = (I *) malloc((2 + m + n) * sizeof(I));
	A->order = order;

	return OPTKIT_SUCCESS;
}

template<typename T, typename I>
ok_status sp_matrix_calloc_(sp_matrix_<T, I> * A, size_t m, size_t n,
	size_t nnz, enum CBLAS_ORDER order)
{
	OK_RETURNIF_ERR( (sp_matrix_alloc_<T, I>(A, m, n, nnz, order)) );
	memset(A->val, 0, 2 * nnz * sizeof(T));
	memset(A->ind, 0, 2 * nnz * sizeof(I));
	memset(A->ptr, 0, (2 + m + n) * sizeof(I));
	return OPTKIT_SUCCESS;
}

template<typename T, typename I>
ok_status sp_matrix_free_(sp_matrix_<T, I> * A)
{
	OK_CHECK_SPARSEMAT(A);
	ok_free(A->val);
	ok_free(A->ind);
	ok_free(A->ptr);
	A->size1 = (size_t) 0;
	A->size2 = (size_t) 0;
	A->nnz = (size_t) 0;
	A->ptrlen = (size_t) 0;
	return OPTKIT_SUCCESS;
}

template<typename T, typename I>
ok_status sp_matrix_memcpy_mm_(sp_matrix_<T, I> * A,
	const sp_matrix_<T, I> * B)
{
	OK_CHECK_SPARSEMAT(A);
	OK_CHECK_SPARSEMAT(B);
	if (A->nnz != B->nnz || A->size1 + A->size2 != B->size1 + B->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	memcpy(A->val, B->val, 2 * A->nnz * sizeof(T));
	memcpy(A->ind, B->ind, 2 * A->nnz * sizeof(I));
	memcpy(A->ptr, B->ptr, (A->size1 + A->size2 + 2) * sizeof(I));
	return OPTKIT_SUCCESS;
}

template<typename T, typename I>
ok_status sp_matrix_memcpy_vals_mm_(sp_matrix_<T, I> * A,
	const sp_matrix_<T, I> * B)
{
	OK_CHECK_SPARSEMAT(A);
	OK_CHECK_SPARSEMAT(B);
	memcpy(A->val, B->val, 2 * A->nnz * sizeof(T));
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

void __csr2csc(size_t m, size_t n, size_t nnz, ok_float * csr_val,
	ok_int * row_ptr, ok_int * col_ind, ok_float * csc_val,
	ok_int * row_ind, ok_int * col_ptr)
{
	ok_int i, j, k, l;
	memset(col_ptr, 0, (n + 1) * sizeof(ok_int));

	/* count nonzeros in each column */
	for (i = 0; i < (ok_int) nnz; i++)
		col_ptr[col_ind[i] + 1]++;

	/* cumulative sum of column counts = pointer offsets */
	for (i = 0; i < (ok_int) n; i++)
		col_ptr[i + 1] += col_ptr[i];

	/* copy data row-wise by using pointer offsets as write heads */
	for (i = 0; i < (ok_int) m; i++) {
		for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
			k = col_ind[j];
			l = col_ptr[k]++;
			row_ind[l] = i;
			csc_val[l] = csr_val[j];
		}
	}

	/* return write heads back to pointer offsets */
	for (i = (ok_int) n; i > 0; i--)
	    col_ptr[i] = col_ptr[i - 1];

	col_ptr[0] = 0;
}

void __transpose_inplace(sp_matrix * A, SPARSE_TRANSPOSE_DIRECTION dir)
{
	if (dir == Forward2Adjoint)
		if (A->order == CblasRowMajor)
			__csr2csc(A->size1, A->size2, A->nnz, A->val, A->ptr,
				A->ind, A->val + A->nnz, A->ind + A->nnz,
				A->ptr + A->ptrlen);
		else
			__csr2csc(A->size2, A->size1, A->nnz, A->val, A->ptr,
				A->ind, A->val + A->nnz, A->ind + A->nnz,
				A->ptr + A->ptrlen);
	else
		if (A->order == CblasRowMajor)
			__csr2csc(A->size2, A->size1, A->nnz, A->val + A->nnz,
				A->ptr + A->ptrlen, A->ind + A->nnz, A->val,
				A->ind, A->ptr);
		else
			__csr2csc(A->size1, A->size2, A->nnz, A->val + A->nnz,
				A->ptr + A->ptrlen, A->ind + A->nnz, A->val,
				A->ind, A->ptr);
}

ok_status sp_make_handle(void ** sparse_handle)
{
	* sparse_handle = OK_NULL;
	return OPTKIT_SUCCESS;
}

ok_status sp_destroy_handle(void * sparse_handle)
{
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order)
	{ return sp_matrix_alloc_<ok_float, ok_int>(A, m, n, nnz, order); }

ok_status sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order)
	{ return sp_matrix_calloc_<ok_float, ok_int>(A, m, n, nnz, order); }

ok_status sp_matrix_free(sp_matrix * A)
	{ return sp_matrix_free_<ok_float, ok_int>(A); }

ok_status sp_matrix_memcpy_mm(sp_matrix * A, const sp_matrix * B)
	{ return sp_matrix_memcpy_mm_<ok_float, ok_int>(A, B); }

ok_status sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A,
	const ok_float * val, const ok_int * ind, const ok_int * ptr)
{
	OK_CHECK_SPARSEMAT(A);
	if (!val || !ind || !ptr)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	memcpy(A->val, val, A->nnz * sizeof(ok_float));
	memcpy(A->ind, ind, A->nnz * sizeof(ok_int));
	memcpy(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
	__transpose_inplace(A, Forward2Adjoint);
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr,
	const sp_matrix * A)
{
	OK_CHECK_SPARSEMAT(A);
	if (!val || !ind || !ptr)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );

	memcpy(val, A->val, A->nnz * sizeof(ok_float));
	memcpy(ind, A->ind, A->nnz * sizeof(ok_int));
	memcpy(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_memcpy_vals_mm(sp_matrix * A, const sp_matrix * B)
	{ return sp_matrix_memcpy_vals_mm_<ok_float, ok_int>(A, B); }

ok_status sp_matrix_memcpy_vals_ma(void * sparse_handle, sp_matrix * A,
  const ok_float * val)
{
	OK_CHECK_SPARSEMAT(A);
	OK_CHECK_PTR(val);

	memcpy(A->val, val, A->nnz * sizeof(ok_float));
	__transpose_inplace(A, Forward2Adjoint);
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_memcpy_vals_am(ok_float * val, const sp_matrix * A)
{
	OK_CHECK_SPARSEMAT(A);
	OK_CHECK_PTR(val);

	memcpy(val, A->val, A->nnz * sizeof(ok_float));
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_abs(sp_matrix * A)
{
	size_t i;
	OK_CHECK_SPARSEMAT(A);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < 2 * A->nnz; ++i)
		A->val[i] = MATH(fabs)(A->val[i]);
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_pow(sp_matrix * A, const ok_float x)
{
	size_t i;
	OK_CHECK_SPARSEMAT(A);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < 2 * A->nnz; ++i)
		A->val[i] = MATH(pow)(A->val[i], x);
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_scale(sp_matrix * A, const ok_float alpha)
{
	OK_CHECK_SPARSEMAT(A);
	CBLAS(scal)( (int) (2 * A->nnz), alpha, A->val, 1);
	return OPTKIT_SUCCESS;
}

ok_status __sp_matrix_scale_diag(sp_matrix * A, const vector * v,
	enum CBLAS_SIDE side)
{
	size_t i, offset, offsetnz, stop;
	SPARSE_TRANSPOSE_DIRECTION dir;

	OK_CHECK_SPARSEMAT(A);
	if ((side == CblasLeft && A->size1 != v->size ) ||
		(side == CblasRight && A->size2 != v->size))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if (side == CblasLeft) {
		offsetnz = (A->order == CblasRowMajor) ? 0 : A->nnz;
		offset = (A->order == CblasRowMajor) ? 0 : A->ptrlen;
		stop = (A->order == CblasRowMajor) ? A->ptrlen - 1 :
						     1 + A->size1 + A->size2;
		dir = (A->order == CblasRowMajor) ? Forward2Adjoint :
						    Adjoint2Forward;
	} else {
		offsetnz = (A->order == CblasRowMajor) ? A->nnz : 0;
		offset = (A->order == CblasRowMajor) ? A->ptrlen : 0;
		stop = (A->order == CblasRowMajor) ? 1 + A->size1 + A->size2 :
						     A->ptrlen - 1;
		dir = (A->order == CblasRowMajor) ? Adjoint2Forward :
						    Forward2Adjoint;
	  }

	for (i = offset; i < stop; ++i) {
		if (A->ptr[i + 1] == A->ptr[i])
			continue;
		CBLAS(scal)( (int) (A->ptr[i + 1] - A->ptr[i]),
			v->data[i - offset], A->val + A->ptr[i] + offsetnz, 1);
	}
	__transpose_inplace(A, dir);
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_scale_left(void * sparse_handle, sp_matrix * A,
	const vector * v)
	{ return __sp_matrix_scale_diag(A, v, CblasLeft); }

ok_status sp_matrix_scale_right(void * sparse_handle, sp_matrix * A,
	const vector * v)
	{ return __sp_matrix_scale_diag(A, v, CblasRight); }

ok_status sp_matrix_print(const sp_matrix * A)
{
	size_t i;
	ok_int j, ptr1, ptr2;

	OK_CHECK_SPARSEMAT(A);
	if (A->order == CblasRowMajor)
		printf("sparse CSR matrix:\n");
	else
		printf("sparse CSC matrix:\n");

	printf("dims: %u, %u\n", (uint) A->size1, (uint) A->size2);
	printf("# nonzeros: %u\n", (uint) A->nnz);

	if (A->order == CblasRowMajor)
		for(i = 0; i < A->ptrlen - 1; ++ i) {
			ptr1 = A->ptr[i];
			ptr2 = A->ptr[i + 1];
			for(j = ptr1; j < ptr2; ++j){
				printf("(%i, %i)\t%e\n", (int) i,  A->ind[j],
					A->val[j]);
			}
		}
	else
		for(i = 0; i < A->ptrlen - 1; ++ i) {
			ptr1 = A->ptr[i];
			ptr2 = A->ptr[i + 1];
			for(j = ptr1; j < ptr2; ++j)
				printf("(%i, %i)\t%e\n", A->ind[j], (int) i,
					A->val[j]);
		}
	printf("\n");
	return OPTKIT_SUCCESS;
}

ok_status sp_matrix_print_transpose(const sp_matrix * A)
{
	size_t i, ptrlen = A->size1 + A->size2 + 2 - A->ptrlen;
	ok_int j, ptr1, ptr2;
	ok_int * ptr_base = A->ptr + A->ptrlen;
	ok_int * ind_base = A->ind + A->nnz;
	ok_float * val_base = A->val + A->nnz;

	OK_CHECK_SPARSEMAT(A);
	if (A->order == CblasRowMajor)
		printf("sparse CSC matrix:\n");
	else
		printf("sparse CSR matrix:\n");

	printf("dims: %u, %u\n", (uint) A->size1, (uint) A->size2);
	printf("# nonzeros: %u\n", (uint) A->nnz);

	if (A->order == CblasRowMajor)
		for(i = 0; i < ptrlen - 1; ++ i) {
			ptr1 = ptr_base[ + i];
			ptr2 = ptr_base[i + 1];
			for(j = ptr1; j < ptr2; ++j)
				printf("(%i, %i)\t%e\n", ind_base[j], (int) i,
					val_base[j]);
		}
	else
		for(i = 0; i < ptrlen - 1; ++ i) {
			ptr1 = ptr_base[i];
			ptr2 = ptr_base[i + 1];
			for(j = ptr1; j < ptr2; ++j)
				printf("(%i, %i)\t%e\n", (int) i,  ind_base[j],
					val_base[j]);
		}
	printf("\n");
	return OPTKIT_SUCCESS;
}

/*
 * Always perform forward (non-transpose) operations.
 * cusparse library uses csr, so for (input, op) combination:
 *      csr, forward op -> forward
 *      csr, adjoint op -> adjoint
 *      csc, forward op -> adjoint
 *      csc, adjoint op -> forward
 */
ok_status sp_blas_gemv(void * sparse_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, sp_matrix * A, vector * x, ok_float beta, vector * y)
{
	size_t ptrlen, i;
	ok_int j;
	ok_float * val, tmp;
	ok_int * ind, * ptr;

	OK_CHECK_SPARSEMAT(A);
	OK_CHECK_VECTOR(x);
	OK_CHECK_VECTOR(y);
	if ((transA == CblasNoTrans &&
		(A->size1 != y->size || A->size2 != x->size)) ||
	    (transA == CblasTrans &&
	    	(A->size1 != x->size || A->size2 != y->size)))
	    	return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if ((A->order == CblasRowMajor) != (transA == CblasTrans)) {
		ptrlen = A->ptrlen;
		ptr = A->ptr;
		ind = A->ind;
		val = A->val;
	} else {
		ptrlen = A->size1 + A->size2 + 2 - A->ptrlen;
		ptr = A->ptr + A->ptrlen;
		ind = A->ind + A->nnz;
		val = A->val + A->nnz;
	}

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < ptrlen - 1; ++i) {
		tmp = kZero;
		for (j = ptr[i]; j < ptr[i + 1]; ++j)
			tmp += val[j] * x->data[ind[j]];
		y->data[i] = alpha * tmp + beta * y->data[i];
	}
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
