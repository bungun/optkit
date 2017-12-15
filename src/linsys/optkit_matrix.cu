#include "optkit_defs_gpu.h"
#include "optkit_matrix.h"

/*
 * MATRIX CUDA helper methods
 * ==========================
 */

template<typename T>
static __global__ void __strided_memcpy(T *x, size_t stride_x, const T *y,
	size_t stride_y, size_t size)
{
	uint i, tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (i = tid; i < size; i += gridDim.x * blockDim.x)
	x[i * stride_x] = y[i * stride_y];
}

/*row major setter */
template<typename T>
static __global__ void __matrix_set_r(T *data, T x, size_t stride,
	size_t size1, size_t size2)
{
	uint i, j;
	uint thread_id_row = blockIdx.x * blockDim.x + threadIdx.x;
	uint thread_id_col = blockIdx.y * blockDim.y + threadIdx.y;
	uint incr_x = gridDim.x * blockDim.x;
	uint incr_y = gridDim.y * blockDim.y;
	for (i = thread_id_row; i < size1; i += incr_x)
		for (j = thread_id_col; j < size2; j += incr_y)
			data[i * stride + j] = x;
}

/*column major setter */
template<typename T>
static __global__ void __matrix_set_c(T *data, T x, size_t stride,
	size_t size1, size_t size2)
{
	uint i, j;
	uint thread_id_row = blockIdx.x * blockDim.x + threadIdx.x;
	uint thread_id_col = blockIdx.y * blockDim.y + threadIdx.y;
	uint incr_x = gridDim.x * blockDim.x;
	uint incr_y = gridDim.y * blockDim.y;
	for (j = thread_id_col; j < size2; j += incr_y)
		for (i = thread_id_row; i < size1; i += incr_x)
			data[i + j * stride] = x;
}

template<typename T>
static ok_status __matrix_set_all(matrix_<T> *A, T x)
{
	uint grid_dimx = calc_grid_dim(A->size1);
	uint grid_dimy = calc_grid_dim(A->size2);
	dim3 grid_dim(grid_dimx, grid_dimy, 1u);
	dim3 block_dim(kBlockSize2D, kBlockSize2D, 1u);

	if (A->order == CblasRowMajor)
		__matrix_set_r<T><<<grid_dim, block_dim>>>(A->data, x, A->ld,
			A->size1, A->size2);
	else
		__matrix_set_c<T><<<grid_dim, block_dim>>>(A->data, x, A->ld,
			A->size1, A->size2);
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

template<typename T>
static __global__ void __matrix_add_constant_diag(T *data, T x, size_t stride)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	data[i * stride + i] += x;
}

template<typename T>
ok_status matrix_alloc_(matrix_<T> *A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	OK_CHECK_PTR(A);
	if (A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_OVERWRITE );
	A->size1 = m;
	A->size2 = n;
	A->ld = (ord == CblasRowMajor) ? n : m;
	A->order = ord;
	return ok_alloc_gpu(A->data, m * n * sizeof(ok_float));

}

template<typename T>
ok_status matrix_calloc_(matrix_<T> *A, size_t m, size_t n,
        enum CBLAS_ORDER ord)
{
	OK_RETURNIF_ERR( matrix_alloc(A, m, n, ord) );
	return ok_memset_gpu(A->data, 0, m * n * sizeof(ok_float));
}

template<typename T>
ok_status matrix_free_(matrix_<T> *A)
{
	OK_CHECK_MATRIX(A);
	A->size1 = (size_t) 0;
	A->size2 = (size_t) 0;
	A->ld = (size_t) 0;
	return ok_free_gpu(A->data);
}

template<typename T>
ok_status matrix_submatrix_(matrix_<T> *A_sub, matrix_<T> *A, size_t i,
	size_t j, size_t n1, size_t n2)
{
	if (!A_sub || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	A_sub->size1 = n1;
	A_sub->size2 = n2;
	A_sub->ld = A->ld;
	A_sub->data = (A->order == CblasRowMajor) ?
		      A->data + (i * A->ld) + j : A->data + i + (j * A->ld);
	A_sub->order = A->order;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_row_(vector_<T> *row, matrix_<T> *A, size_t i)
{
	if (!row || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	row->size = A->size2;
	row->stride = (A->order == CblasRowMajor) ? 1 : A->ld;
	row->data = (A->order == CblasRowMajor) ?
		    A->data + (i * A->ld) : A->data + i;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_column_(vector *col, matrix *A, size_t j)
{
	if (!col || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	col->size = A->size1;
	col->stride = (A->order == CblasRowMajor) ? A->ld : 1;
	col->data = (A->order == CblasRowMajor) ?
		    A->data + j : A->data + (j * A->ld);
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_diagonal_(vector_<T> *diag, matrix_<T> *A)
{
	if (!diag || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	diag->data = A->data;
	diag->stride = A->ld + 1;
	diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_cast_vector_(vector_<T> *v, matrix_<T> *A)
{
	if (!v || !A || !A->data)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	v->size = A->size1 * A->size2;
	v->stride = 1;
	v->data = A->data;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_view_array_(matrix_<T> *A, const T *base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord)
{
	if (!A || !base)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	A->size1 = n1;
	A->size2 = n2;
	A->data = (ok_float *) base;
	A->ld = (ord == CblasRowMajor) ? n2 : n1;
	A->order = ord;
	return OPTKIT_SUCCESS;
}

template<typename T>
ok_status matrix_set_all_(matrix_<T> *A, T x)
{
	OK_CHECK_MATRIX(A);
	return __matrix_set_all<T>(A, x);
}

template<typename T>
ok_status matrix_memcpy_mm_(matrix_<T> *A, const matrix_<T> *B)
{
	uint i, j, grid_dim;
	OK_CHECK_MATRIX(A);
	OK_CHECK_MATRIX(B);
	if (A->size1 != B->size1 || (A->size2 != B->size2))
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	if (A->order == B->order) {
		return ok_memcpy_gpu(A->data, B->data,
			      A->size1 * A->size2 * sizeof(T));
	} else if (A->order == CblasRowMajor) {
		/* A row major, B column major */
		grid_dim = calc_grid_dim(A->size1);
		for (i = 0; i < A->size1; ++i) {
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
				A->data + i * A->size2, 1,
				B->data + i, A->ld, A->size2);
			cudaDeviceSynchronize();
		}
	} else {
		/* A column major, B row major */
		grid_dim = calc_grid_dim(A->size2);
		for (j= 0; j < A->size2; ++j) {
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
				A->data + j * A->size1, 1,
				B->data + j, A->ld, A->size1);
			cudaDeviceSynchronize();
		}
	}
	return OK_STATUS_CUDA;
}

/*
 * Handle the following cases:
 *      A->order = ord
 *      A->order != ord, ord == CblasColMajor (A row major, B col major)
 *      A->order != ord, ord == CblasRowMajor (A col major, B row major)
 */
template<typename T>
ok_status matrix_memcpy_ma_(matrix_<T> *A, const T *B,
	const enum CBLAS_ORDER ord)
{
	ok_status err = OPTKIT_SUCCESS;
	uint i, j, grid_dim;
	T *row, *col;

	OK_CHECK_MATRIX(A);
	OK_CHECK_PTR(B);

	if (A->order == ord) {
		if (ord == CblasRowMajor)
			for (i = 0; i < A->size1 && !err; ++i)
				err = ok_memcpy_gpu(A->data + i * A->ld,
					B + i * A->size2, A->size2 * sizeof(T));
		else
			for (j = 0; j < A->size2 && !err; ++j)
				err = ok_memcpy_gpu(A->data + j * A->ld,
					B + j * A->size1, A->size1 * sizeof(T));
	} else if (ord == CblasColMajor) {
		err = ok_alloc_gpu(col, A->size1 * sizeof(ok_float));
		grid_dim = calc_grid_dim(A->size1);
		for (j = 0; j < A->size2 && !err; ++j) {
			err = ok_memcpy_gpu(col, B + j * A->size1,
				A->size1 * sizeof(T));
			if (!err) {
				__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
					A->data + j, A->ld, col, 1, A->size1);
				cudaDeviceSynchronize();
				err = OK_STATUS_CUDA;
			}
		}
		OK_MAX_ERR( err, ok_free_gpu(col) );
	} else {
		err = ok_alloc_gpu(row, A->size2 * sizeof(ok_float));
		grid_dim = calc_grid_dim(A->size2);
		for (i = 0; i < A->size1; ++i) {
			err = ok_memcpy_gpu(row, B + i * A->size2,
				      A->size2 * sizeof(T));
			if (!err)
				__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
					A->data + i, A->ld, row, 1, A->size2);
				cudaDeviceSynchronize();
				err = OK_STATUS_CUDA;
		}
		OK_MAX_ERR( err, ok_free_gpu(row) );
	}
	return err;
}

/*
 * Handle the following cases:
 *      ord = B->order
 *      ord != B->order, ord == CblasRowMajor (A row major, B col major)
 *      ord != B->ord, order == CblasColMajor (A col major, B row major)
 */
template<typename T>
ok_status matrix_memcpy_am_(T *A, const matrix_<T> *B,
	const enum CBLAS_ORDER ord)
{
	ok_status err = OPTKIT_SUCCESS;
	uint i, j, grid_dim;
	T *row, *col;

	OK_CHECK_MATRIX(B);
	OK_CHECK_PTR(A);

	if (ord == B->order) {
		if (ord == CblasRowMajor)
			for (i = 0; i < B->size1 && !err; ++i)
				err = ok_memcpy_gpu(A + i * B->size2,
					B->data + i * B->ld,
					B->size2 * sizeof(T));
		else
			for (j = 0; j < B->size2 && !err; ++j)
				err = ok_memcpy_gpu(A + j * B->size1,
					B->data + j * B->ld,
					B->size1 * sizeof(T));
	} else if (ord == CblasRowMajor) {
		err = ok_alloc_gpu(row, B->size2 * sizeof(ok_float));
		grid_dim = calc_grid_dim(B->size2);
		for (i = 0; i < B->size1 && !err; ++i) {
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(row, 1,
				B->data + i, B->ld, B->size2);
			cudaDeviceSynchronize();
			err = OK_STATUS_CUDA;
			OK_CHECK_ERR( err, ok_memcpy_gpu(A + i * B->size2, row,
				      B->size2 * sizeof(T)) );
		}
		OK_MAX_ERR( err, ok_free_gpu(row) );
	} else {
		err = ok_alloc_gpu(col, B->size1 * sizeof(ok_float));
		grid_dim = calc_grid_dim(B->size1);
		for (j = 0; j < B->size2 && !err; ++j) {
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(col, 1,
				B->data + j, B->ld, B->size1);
			cudaDeviceSynchronize();
			err = OK_STATUS_CUDA;
			OK_CHECK_ERR(err, ok_memcpy_gpu(A + j * B->size1, col,
				      B->size1 * sizeof(T)) );
		}
		OK_MAX_ERR( err, ok_free_gpu(col) );
	}
	return err;
}

#ifdef __cplusplus
extern "C" {
#endif

ok_status matrix_alloc(matrix *A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ return matrix_alloc_<ok_float>(A, m, n, ord); }

ok_status matrix_calloc(matrix *A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ return matrix_calloc_<ok_float>(A, m, n, ord); }

ok_status matrix_free(matrix *A)
	{ return matrix_free_<ok_float>(A); }

ok_status matrix_submatrix(matrix *A_sub, matrix *A, size_t i, size_t j,
	size_t n1, size_t n2)
	{ return matrix_submatrix_<ok_float>(A_sub, A, i, j, n1, n2); }

ok_status matrix_row(vector *row, matrix *A, size_t i)
	{ return matrix_row_<ok_float>(row, A, i); }

ok_status matrix_column(vector *col, matrix *A, size_t j)
	{ return matrix_column_<ok_float>(col, A, j); }

ok_status matrix_diagonal(vector *diag, matrix *A)
	{ return matrix_diagonal_<ok_float>(diag, A); }

ok_status matrix_cast_vector(vector *v, matrix *A)
	{ return matrix_cast_vector_<ok_float>(v, A); }

ok_status matrix_view_array(matrix *A, const ok_float *base, size_t n1,
	size_t n2, enum CBLAS_ORDER ord)
	{ return matrix_view_array_<ok_float>(A, base, n1, n2, ord); }

ok_status matrix_set_all(matrix *A, ok_float x)
	{ return matrix_set_all_<ok_float>(A, x); }

ok_status matrix_memcpy_mm(matrix *A, const matrix *B)
	{ return matrix_memcpy_mm_<ok_float>(A, B); }

ok_status matrix_memcpy_ma(matrix *A, const ok_float *B,
	const enum CBLAS_ORDER ord)
	{ return matrix_memcpy_ma_<ok_float>(A, B, ord); }

ok_status matrix_memcpy_am(ok_float *A, const matrix *B,
	const enum CBLAS_ORDER ord)
	{ return matrix_memcpy_am_<ok_float>(A, B, ord); }

ok_status matrix_print(matrix *A)
{
	ok_float row_host[A->size2];
	vector row;

	row.data = OK_NULL;
	OK_CHECK_MATRIX(A);

	for (uint i = 0; i < A->size1; ++i) {
		OK_RETURNIF_ERR( matrix_row(&row, A, i) );
		OK_RETURNIF_ERR( vector_memcpy_av(row_host, &row, 1) );
		for (uint j = 0; j < A->size2; ++j)
			printf("%0.2e ", row_host[j]);
		printf("\n");
	}
	printf("\n");
	return OPTKIT_SUCCESS;
}

ok_status matrix_scale(matrix *A, ok_float x)
{
	size_t i;
	vector row_col;
	ok_status err = OPTKIT_SUCCESS;

	row_col.data = OK_NULL;
	OK_CHECK_MATRIX(A);

	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			err = vector_scale(&row_col, x);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			err = vector_scale(&row_col, x);
		}
	return OK_SCAN_ERR( err );
}

ok_status matrix_scale_left(matrix *A, const vector *v)
{
	size_t i;
	vector col;
	ok_status err = OPTKIT_SUCCESS;

	col.data = OK_NULL;
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);
	if (A->size1 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	for(i = 0; i < A->size2 && !err; ++i) {
		matrix_column(&col, A, i);
		err = vector_mul(&col, v);
	}
	return OK_SCAN_ERR( err );
}

ok_status matrix_scale_right(matrix *A, const vector *v)
{
	size_t i;
	vector row;
	ok_status err = OPTKIT_SUCCESS;

	row.data = OK_NULL;
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);
	if (A->size2 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	for(i = 0; i < A->size1 && !err; ++i) {
		matrix_row(&row, A, i);
		err = vector_mul(&row, v);
	}
	return OK_SCAN_ERR( err );
}

ok_status matrix_abs(matrix *A)
{
	size_t i;
	vector row_col;
	ok_status err = OPTKIT_SUCCESS;

	row_col.data = OK_NULL;
	OK_CHECK_MATRIX(A);

	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1 && !err; ++i) {
			matrix_row(&row_col, A, i);
			err = vector_abs(&row_col);
		}
	else
		for(i = 0; i < A->size2 && !err; ++i) {
			matrix_column(&row_col, A, i);
			err = vector_abs(&row_col);
		}
	return OK_SCAN_ERR( err );
}

ok_status matrix_pow(matrix *A, const ok_float x)
{
	size_t i;
	vector row_col;
	ok_status err = OPTKIT_SUCCESS;

	row_col.data = OK_NULL;
	OK_CHECK_MATRIX(A);

	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1 && !err; ++i) {
			matrix_row(&row_col, A, i);
			err = vector_pow(&row_col, x);
		}
	else
		for(i = 0; i < A->size2 && !err; ++i) {
			matrix_column(&row_col, A, i);
			err = vector_pow(&row_col, x);
		}
	return OK_SCAN_ERR( err );
}

#ifdef __cplusplus
}
#endif
