#include "optkit_defs_gpu.h"
#include "optkit_matrix.hpp"

/*
 * MATRIX CUDA helper methods
 * ==========================
 */

/* row major setter */
 template<typename T>
__global__ void __matrix_set_r(T * data, T x, size_t stride, size_t size1,
	size_t size2)
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

/* column major setter */
template<typename T>
__global__ void __matrix_set_c(T * data, T x, size_t stride,
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
void __matrix_set_all(matrix_<T> * A, T x)
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
	CUDA_CHECK_ERR;
}

template<typename T>
__global__ void __matrix_add_constant_diag(T * data, T x,
	size_t stride)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	data[i * stride + i] += x;
}

template<typename T>
void matrix_alloc_(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	A->size1 = m;
	A->size2 = n;
	ok_alloc_gpu(A->data, m * n * sizeof(ok_float));
	A->ld = (ord == CblasRowMajor) ? n : m;
	A->order = ord;
}

template<typename T>
void matrix_calloc_(matrix_<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	if (!A)
		return;
	matrix_alloc(A, m, n, ord);
	cudaMemset(A->data, 0, m * n * sizeof(ok_float));
	CUDA_CHECK_ERR;
}

template<typename T>
void matrix_free_(matrix_<T> * A)
{
	if (!A || !A->data)
		return;
	ok_free_gpu(A->data);
	A->size1 = (size_t) 0;
	A->size2 = (size_t) 0;
	A->ld = (size_t) 0;

}

template<typename T>
void matrix_submatrix(matrix_<T> * A_sub, matrix_<T> * A, size_t i, size_t j,
	size_t n1, size_t n2)
{
	if (!A_sub || !A)
		return
	A_sub->size1 = n1;
	A_sub->size2 = n2;
	A_sub->ld = A->ld;
	A_sub->data = (A->order == CblasRowMajor) ?
		      A->data + (i * A->ld) + j : A->data + i + (j * A->ld);
	A_sub->order = A->order;
}

template<typename T>
void matrix_row(vector_<T> * row, matrix_<T> * A, size_t i)
{
	if (!row || !A)
		return;
	row->size = A->size2;
	row->stride = (A->order == CblasRowMajor) ? 1 : A->ld;
	row->data = (A->order == CblasRowMajor) ?
		    A->data + (i * A->ld) : A->data + i;
}

template<typename T>
void matrix_column(vector * col, matrix *A, size_t j)
{
	if (!col || !A)
		return;
	col->size = A->size1;
	col->stride = (A->order == CblasRowMajor) ? A->ld : 1;
	col->data = (A->order == CblasRowMajor) ?
		    A->data + j : A->data + (j * A->ld);
}

template<typename T>
void matrix_diagonal(vector_<T> * diag, matrix_<T> *A)
{
	if (!diag || !A)
		return;
	diag->data = A->data;
	diag->stride = A->ld + 1;
	diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

template<typename T>
void matrix_cast_vector(vector_<T> * v, matrix_<T> * A)
{
	if (!v || !A)
		return;
	v->size = A->size1 * A->size2;
	v->stride = 1;
	v->data = A->data;
}

template<typename T>
void matrix_view_array(matrix_<T> * A, const T * base, size_t n1, size_t n2,
	enum CBLAS_ORDER ord)
{
	if (!A || !base)
		return;
	A->size1 = n1;
	A->size2 = n2;
	A->data = (ok_float *) base;
	A->ld = (ord == CblasRowMajor) ? n2 : n1;
	A->order = ord;
}

template<typename T>
void matrix_set_all(matrix_<T> * A, T x)
{
	if (!A)
		return;
	__matrix_set_all<T>(A, x);
}

template<typename T>
void matrix_memcpy_mm(matrix_<T> * A, const matrix_<T> * B)
{
	uint i, j, grid_dim;
	if (A->size1 != B->size1) {
		printf("error: m-dimensions must match for matrix memcpy\n");
		return;
	} else if (A->size2 != B->size2) {
		printf("error: n-dimensions must match for matrix memcpy\n");
		return;
	}

	if (A->order == B->order) {
		ok_memcpy_gpu(A->data, B->data,
			      A->size1 * A->size2 * sizeof(T));
	} else if (A->order == CblasRowMajor) {
		/* A row major, B column major */
		grid_dim = calc_grid_dim(A->size1);
		for (i = 0; i < A->size1; ++i)
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
				A->data + i * A->size2, 1,
				B->data + i, A->ld, A->size2);
	} else {
		/* A column major, B row major */
		grid_dim = calc_grid_dim(A->size2);
		for (j= 0; j < A->size2; ++j)
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
				A->data + j * A->size1, 1,
				B->data + j, A->ld, A->size1);
	}
	CUDA_CHECK_ERR;
}

/*
 * Handle the following cases:
 *      A->order = ord
 *      A->order != ord, ord == CblasColMajor (A row major, B col major)
 *      A->order != ord, ord == CblasRowMajor (A col major, B row major)
 */
template<typename T>
void matrix_memcpy_ma(matrix_<T> * A, const T * B, const enum CBLAS_ORDER ord)
{
	uint i, j, grid_dim;
	T * row, * col;

	if (A->order == ord) {
		if (ord == CblasRowMajor)
			for (i = 0; i < A->size1; ++i)
				ok_memcpy_gpu(A->data + i * A->ld,
					B + i * A->size2, A->size2 * sizeof(T));
		else
			for (j = 0; j < A->size2; ++j)
				ok_memcpy_gpu(A->data + j * A->ld,
					B + j * A->size1, A->size1 * sizeof(T));
	} else if (ord == CblasColMajor) {
		ok_alloc_gpu(col, A->size1 * sizeof(ok_float));
		grid_dim = calc_grid_dim(A->size1);
		for (j = 0; j < A->size2; ++j) {
			ok_memcpy_gpu(col, B + j * A->size1,
				A->size1 * sizeof(T));
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
				A->data + j, A->ld, col, 1, A->size1);
		}
		ok_free_gpu(col);
	} else {
		ok_alloc_gpu(row, A->size2 * sizeof(ok_float));
		grid_dim = calc_grid_dim(A->size2);
		for (i = 0; i < A->size1; ++i) {
			ok_memcpy_gpu(row, B + i * A->size2,
				      A->size2 * sizeof(T));
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(
				A->data + i, A->ld, row, 1, A->size2);
		}
		ok_free_gpu(row);
	}
	CUDA_CHECK_ERR;
}

/*
 * Handle the following cases:
 *      ord = B->order
 *      ord != B->order, ord == CblasRowMajor (A row major, B col major)
 *      ord != B->ord, order == CblasColMajor (A col major, B row major)
 */
template<typename T>
void matrix_memcpy_am(T * A, const matrix_<T> * B, const enum CBLAS_ORDER ord)
{
	uint i, j, grid_dim;
	T * row, * col;
	if (ord == B->order) {
		if (ord == CblasRowMajor)
			for (i = 0; i < B->size1; ++i)
				ok_memcpy_gpu(A + i * B->size2,
					B->data + i * B->ld,
					B->size2 * sizeof(T));
		else
			for (j = 0; j < B->size2; ++j)
				ok_memcpy_gpu(A + j * B->size1,
					B->data + j * B->ld,
					B->size1 * sizeof(T));
	} else if (ord == CblasRowMajor) {
		ok_alloc_gpu(row, B->size2 * sizeof(ok_float));
		grid_dim = calc_grid_dim(B->size2);
		for (i = 0; i < B->size1; ++i) {
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(row, 1,
				B->data + i, B->ld, B->size2);
			ok_memcpy_gpu(A + i * B->size2, row,
				      B->size2 * sizeof(T));
		}
		ok_free_gpu(row);
	} else {
		ok_alloc_gpu(col, B->size1 * sizeof(ok_float));
		grid_dim = calc_grid_dim(B->size1);
		for (j = 0; j < B->size2; ++j) {
			__strided_memcpy<T><<<grid_dim, kBlockSize>>>(col, 1,
				B->data + j, B->ld, B->size1);
			ok_memcpy_gpu(A + j * B->size1, col,
				      B->size1 * sizeof(T));
		}
		ok_free_gpu(col);
	}
	CUDA_CHECK_ERR;
}

template<typename T>
void matrix_scale(matrix_<T> * A, T x)
{
	size_t i;
	vector_<T> row_col;
	row_col.data = OK_NULL;

	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row_<T>(&row_col, A, i);
			vector_scale_<T>(&row_col, x);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column_<T>(&row_col, A, i);
			vector_scale_<T>(&row_col, x);
		}
}

template<typename T>
void matrix_scale_left(matrix_<T> * A, const vector_<T> * v)
{
	size_t i;
	vector_<T> col;
	col.data = OK_NULL;

	for(i = 0; i < A->size2; ++i) {
		matrix_column_<T>(&col, A, i);
		vector_mul_<T>(&col, v);
	}
}

template<typename T>
void matrix_scale_right(matrix_<T> * A, const vector_<T> * v)
{
	size_t i;
	vector_<T> row;
	row.data = OK_NULL;

	for(i = 0; i < A->size1; ++i) {
		matrix_row_<T>(&row, A, i);
		vector_mul_<T>(&row, v);
	}
}

#ifdef __cplusplus
extern "C" {
#endif

void matrix_alloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ matrix_alloc_<ok_float>(A, m, n, ord); }

void matrix_calloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord)
	{ matrix_calloc_<ok_float>(A, m, n, ord); }

void matrix_free(matrix * A)
	{ matrix_free_<ok_flat>(A); }

void matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1,
	size_t n2)
	{ matrix_submatrix_<ok_float>(A_sub, A, i, j, n1, n2); }

void matrix_row(vector * row, matrix * A, size_t i)
	{ matrix_row_<ok_float>(row, A, i); }

void matrix_column(vector * col, matrix * A, size_t j)
	{ matrix_column_<ok_float>(col, A, j); }

void matrix_diagonal(vector * diag, matrix * A)
	{ matrix_diagonal_<ok_float>(diag, A); }

void matrix_cast_vector(vector * v, matrix * A)
	{ matrix_cast_vector_<ok_float>(v, A); }

void matrix_view_array(matrix * A, const ok_float * base, size_t n1, size_t n2,
        enum CBLAS_ORDER ord)
	{ matrix_view_array_<ok_float>(A, base, n1, n2, ord); }

void matrix_set_all(matrix * A, ok_float x)
	{ matrix_set_all_<ok_float>(A, x); }

void matrix_memcpy_mm(matrix * A, const matrix * B)
	{ matrix_memcpy_mm_<ok_float>(A, B); }

void matrix_memcpy_ma(matrix * A, const ok_float * B,
	const enum CBLAS_ORDER ord)
	{ matrix_memcpy_ma_<ok_float>(A, B, ord); }

void matrix_memcpy_am(ok_float * A, const matrix * B,
	const enum CBLAS_ORDER ord)
	{ matrix_memcpy_am_<ok_float>(A, B, ord); }

void matrix_print(matrix * A)
{
	ok_float row_host[A->size2];
	vector row = (vector){0, 0, OK_NULL};

	for (uint i = 0; i < A->size1; ++i) {
		matrix_row(&row, A, i);
		vector_memcpy_av(row_host, &row, 1);
		for (uint j = 0; j < A->size2; ++j)
			printf("%0.2e ", row_host[j]);
		printf("\n");
	}
	printf("\n");
}

void matrix_scale(matrix *A, ok_float x)
	{ matrix_scale_<ok_float>(A, x); }

void matrix_scale_left(matrix * A, const vector * v)
	{ matrix_scale_left_<ok_float>(A, v); }

void matrix_scale_right(matrix * A, const vector * v)
	{ matrix_scale_right_<ok_float>(A, v); }

void matrix_abs(matrix * A)
{
	size_t i;
	vector row_col = (vector){0,0,OK_NULL};
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			vector_abs(&row_col);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			vector_abs(&row_col);
		}
}

void matrix_pow(matrix * A, const ok_float x)
{
	size_t i;
	vector row_col = (vector){0,0,OK_NULL};
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			vector_pow(&row_col, x);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			vector_pow(&row_col, x);
		}
}

#ifdef __cplusplus
}
#endif