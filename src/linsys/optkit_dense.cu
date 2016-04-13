#include "optkit_defs_gpu.h"
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

int __matrix_order_compat(const matrix * A, const matrix * B, const char * nm_A,
	const char * nm_B, const char * nm_routine)
{
	if (A->order == B->order)
	    return 1;

	printf("OPTKIT ERROR (%s) matrices %s and %s must have same layout.\n",
		 nm_routine, nm_A, nm_B);
	return 0;
}

/* row major data retrieval */
__device__ inline ok_float& __matrix_get_r(ok_float * A, uint i, uint j,
	uint stride)
{
	return A[i * stride + j];
}

/* column major data retrieval */
__device__ inline ok_float& __matrix_get_c(ok_float * A, uint i, uint j,
	uint stride)
{
	return A[i + j * stride];
}

/* cholesky decomposition of a single block */
static __global__ void __block_chol(ok_float * A, uint iter, uint ld,
	const enum CBLAS_ORDER ord)
{
	uint col, row, mat_dim, global_col, global_row, i;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float L[kTileLD * kTileSize];
	ok_float a11;

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	col = threadIdx.x;
	row = threadIdx.y;
	mat_dim = blockDim.x;

	global_col = iter * kTileSize + col;
	global_row = iter * kTileSize + row;

	get(L, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	for (i = 0; i < mat_dim; ++i) {
		/* l11 = sqrt(a11) */
		a11 = sqrt(get(L, i, i, kTileLD));
		__syncthreads();


		/* l21 = a21 / l11 */
		if (row >= i && col == 0)
			get(L, row, i, kTileLD) /= a11;
		__syncthreads();

		/* a22 -= l21 * l21' */
		if (row >= col && col > i)
			get(L, row, col, kTileLD) -=
			     get(L, col, i, kTileLD) *
			     get(L, row, i, kTileLD);
		__syncthreads();
	}

	if (row >= col)
		get(A, global_row, global_col, ld) = get(L, row, col, kTileLD);
}

static __global__ void __block_trsv(ok_float * A, uint iter, uint n, uint ld,
	const enum CBLAS_ORDER ord)
{
	uint tile_idx, row, global_row, global_col, i, j;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float L[kTileLD * kTileSize];
	__shared__ ok_float A12[kTileLD * kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	tile_idx = blockIdx.x;
	row = threadIdx.x;
	global_col = iter * kTileSize;
	global_row = iter * kTileSize + row;

	/* Load A -> L columnwise. */
	for (i = 0; i < kTileSize; ++i)
		get(L, row, i, kTileLD) = get(A, global_row, global_col + i,
			ld);
	__syncthreads();

	global_row = row + (iter + tile_idx + 1u) * kTileSize;

	if (global_row < n)
		for (i = 0; i < kTileSize; ++i)
			get(A12, row, i, kTileLD) =
				get(A, global_row, global_col + i, ld);
	__syncthreads();

	if (global_row < n)
		for (i = 0; i < kTileSize; ++i) {
			for (j = 0; j < i; ++j)
				get(A12, row, i, kTileLD) -=
					get(A12, row, j, kTileLD) *
					get(L, i, j, kTileLD);
			get(A12, row, i, kTileLD) /= get(L, i, i, kTileLD);
		}
	__syncthreads();

	if (global_row < n)
		for (uint i = 0; i < kTileSize; ++i)
			get(A, global_row, global_col + i, ld) =
				get(A12, row, i, kTileLD);
	__syncthreads();
}

/*
 * Block Cholesky.
 *   l11 l11^T = a11
 *   l21 = a21 l11^(-T)
 *   a22 = a22 - l21 l21^T
 *
 * Stores result in Lower triangular part.
 */
void linalg_cholesky_decomp(void * linalg_handle, matrix * A)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_tiles, grid_dim, i;
	matrix L21, A22;

	err = cublasGetStream(*(cublasHandle_t *) linalg_handle, &stm);
	num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;

	for (i = 0; i < num_tiles; ++i) {
		if (err != CUBLAS_STATUS_SUCCESS)
			break;

		/* L11 = chol(A11) */
		uint block_dim_1d = kTileSize < A->size1 - i * kTileSize ? \
				    kTileSize : A->size1 - i * kTileSize;
		dim3 block_dim(block_dim_1d, block_dim_1d);

		__block_chol<<<1, block_dim, 0, stm>>>(A->data, i, (uint) A->ld,
			A->order);
		CUDA_CHECK_ERR;

		if (i == num_tiles - 1u)
			break;

		/* L21 = A21 * L21^-T */
		grid_dim = num_tiles - i - 1u;
		matrix_submatrix(&L21, A, (i + 1) * kTileSize, i * kTileSize,
			A->size1 - (i + 1) * kTileSize, kTileSize);

		__block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data, i,
			(uint) A->size1, (uint) A->ld, A->order);
		CUDA_CHECK_ERR;

		/* A22 -= L21 * L21^T */
		matrix_submatrix(&A22, A, (i + 1) * kTileSize,
			(i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
			A->size1 - (i + 1) * kTileSize);

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

static __global__ void __block_diag_gramian(ok_float * A, ok_float * v, uint i,
	uint j, uint ld, uint stride_v, const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float vsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	vsub[row] = v[global_row * stride_v];
	__syncthreads();

	v[global_row * stride_v] += MATH(pow)(get(Asub, row, col, kTileLD), 2);
	__syncthreads();

	v[global_row * stride_v] = vsub[row];
}

void linalg_diag_gramian(void * linalg_handle, const matrix * A, vector * v)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_row_tiles, num_col_tiles, i, j, size1, size2;
	uint ld = (uint) A->ld, stride = (uint) v->stride;
	int skinny = A->size1 == v->size;
	enum CBLAS_ORDER order;

	if (!(v && A)) {
		printf("%s\n", "A and v must both be initialized");
		return;
	}

	if (v->size != A->size1 && v->size != A->size2) {
		printf("%s\n", "dimensions of A and v are incompatible");
		return;
	}

	err = cublasGetStream(*(cublasHandle_t *) linalg_handle, &stm);

	size1 = (skinny) ? A->size1 : A->size2;
	size2 = (skinny) ? A->size2 : A->size1;
	order = (skinny == (A->order == CblasRowMajor)) ?
		CblasRowMajor : CblasColMajor;

	num_row_tiles = (size1 + kTileSize - 1u) / kTileSize;
	num_col_tiles = (size2 + kTileSize - 1u) / kTileSize;

	for (i = 0; i < num_row_tiles; ++i) {
		uint block_dim_v = kTileSize < size1 - i * kTileSize ? \
				   kTileSize : size1 - i * kTileSize;

		for (j = 0; j < num_col_tiles; ++j) {

			if (err != CUBLAS_STATUS_SUCCESS)
				break;
			uint block_dim_h = kTileSize < size2 - j * kTileSize ? \
					   kTileSize : size2 - j * kTileSize;

			dim3 block_dim(block_dim_v, block_dim_h);
			__block_diag_gramian<<<1, block_dim, 0, stm>>>(
				A->data, v->data, i, j, ld, stride, order);
			CUDA_CHECK_ERR;
		}
	}
}

static __global__ void __matrix_scale_left(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v, const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float vsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	vsub[row] = v[global_row * stride_v];
	__syncthreads();

	get(Asub, row, col, kTileLD) *= vsub[row];
	__syncthreads();

	get(A, global_row, global_col, ld) = get(Asub, row, col, kTileLD);
}

static __global__ void __matrix_scale_right(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v, const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float vsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	vsub[col] = v[global_col * stride_v];
	__syncthreads();

	get(Asub, row, col, kTileLD) *= vsub[col];
	__syncthreads();

	get(A, global_row, global_col, ld) = get(Asub, row, col, kTileLD);
}

static __global__ void __matrix_add_left(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v, const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float vsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	vsub[row] = v[global_row * stride_v];
	__syncthreads();

	get(Asub, row, col, kTileLD) += vsub[row];
	__syncthreads();

	get(A, global_row, global_col, ld) = get(Asub, row, col, kTileLD);
}

static __global__ void __matrix_add_right(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v, const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float vsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	vsub[col] = v[global_col * stride_v];
	__syncthreads();

	get(Asub, row, col, kTileLD) += vsub[col];
	__syncthreads();

	get(A, global_row, global_col, ld) = get(Asub, row, col, kTileLD);
}

void linalg_matrix_broadcast_vector(void * linalg_handle, matrix * A,
	const vector * v, const enum OPTKIT_TRANSFORM operation,
	const enum CBLAS_SIDE side)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_row_tiles, num_col_tiles, i, j;
	uint ld = (uint) A->ld, stride = (uint) v->stride;
	void (*transform)(ok_float * A, const ok_float * v, uint i, uint j,
		size_t ld, size_t stride_v, const enum CBLAS_ORDER);

	if (operation == OkTransformScale)
		transform = (side == CblasLeft) ? __matrix_scale_left :
			__matrix_scale_right;
	else
		transform = (side == CblasLeft) ? __matrix_add_left :
			__matrix_add_right;

	err = cublasGetStream(*(cublasHandle_t *) linalg_handle, &stm);
	num_row_tiles = (A->size1 + kTileSize - 1u) / kTileSize;
	num_col_tiles = (A->size2 + kTileSize - 1u) / kTileSize;

	for (i = 0; i < num_row_tiles; ++i) {
		uint block_dim_v = kTileSize < A->size1 - i * kTileSize ? \
				   kTileSize : A->size1 - i * kTileSize;

		for (j = 0; j < num_col_tiles; ++j) {
			if (err != CUBLAS_STATUS_SUCCESS)
				break;

			uint block_dim_h =
				kTileSize < A->size2 - j * kTileSize ? \
				kTileSize : A->size2 - j * kTileSize;

			dim3 block_dim(block_dim_v, block_dim_h);
			transform<<<1, block_dim, 0, stm>>>(A->data, v->data, i,
				j, ld, stride, A->order);
			CUDA_CHECK_ERR;
		}
	}
}

static __global__ void __matrix_row_indmin(size_t * minind, ok_float * minima,
	ok_float * A, size_t i, size_t j, size_t ld, size_t stride_indmin,
	size_t stride_minima, const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row, k;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float minsub[kTileSize];
	__shared__ size_t indminsub[kTileSize];
	ok_float prev;

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;
	// ok_float& (* getp)(ok_float * A, uint i, uint j, uint stride) =
	// 	(ord == CblasRowMajor) ?
	// 	__matrix_get_ptr_r : __matrix_get_ptr_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	if (col == 0)
		minsub[row] = MATH(fmin)(minima[global_row * stride_minima],
			OK_FLOAT_MAX);
	__syncthreads();

	if (col == 0)
		for (k = 0; k < blockDim.y; ++k) {
			prev = minsub[row];
			minsub[row] = MATH(fmin)(minsub[row],
				get(Asub, row, k, kTileLD));
			if (minsub[row] != prev)
				indminsub[row] = global_col + k;
		}
	__syncthreads();

	minima[global_row * stride_minima] = minsub[row];
	__syncthreads();

	minind[global_row * stride_indmin] = indminsub[row];
}

static __global__ void __matrix_col_indmin(size_t * minind, ok_float * minima,
	ok_float * A, size_t i, size_t j, size_t ld, size_t stride_indmin,
	size_t stride_minima, const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row, k;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float minsub[kTileSize];
	__shared__ size_t indminsub[kTileSize];
	ok_float prev;

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	if (row == 0)
		minsub[col] = MATH(fmin)(minima[global_col * stride_minima],
			OK_FLOAT_MAX);
	__syncthreads();

	if (row == 0)
		for (k = 0; k < blockDim.x; ++k) {
			prev = minsub[col];
			minsub[col] = MATH(fmin)(minsub[col],
				get(Asub, k, col, kTileLD));
			if (minsub[col] != prev)
				indminsub[col] = global_row + k;
		}
	__syncthreads();

	minima[global_col * stride_minima] = minsub[col];
	__syncthreads();

	minind[global_col * stride_indmin] = indminsub[col];
}

static __global__ void __matrix_row_min(ok_float * minima, ok_float * A,
	size_t i, size_t j, size_t ld, size_t stride_minima,
	const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row, k;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float minsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	if (col == 0)
		minsub[row] = MATH(fmin)(minima[global_row * stride_minima],
			OK_FLOAT_MAX);
	__syncthreads();

	if (col == 0)
		for (k = 0; k < blockDim.y; ++k)
			minsub[row] = MATH(fmin)(minsub[row],
				get(Asub, row, k, kTileLD));
	__syncthreads();

	minima[global_row * stride_minima] = minsub[row];
}

static __global__ void __matrix_col_min(ok_float * minima, ok_float * A,
	size_t i, size_t j, size_t ld, size_t stride_minima,
	const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row, k;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float minsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	if (row == 0)
		minsub[col] = MATH(fmin)(minima[global_col * stride_minima],
			OK_FLOAT_MAX);
	__syncthreads();

	if (row == 0)
		for (k = 0; k < blockDim.x; ++k)
			minsub[col] = MATH(fmin)(minsub[col],
				get(Asub, k, col, kTileLD));
	__syncthreads();

	minima[global_col * stride_minima] = minsub[col];
}

static __global__ void __matrix_row_max(ok_float * maxima, ok_float * A,
	size_t i, size_t j, size_t ld, size_t stride_maxima,
	const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row, k;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float maxsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	if (col == 0)
		maxsub[row] = MATH(fmax)(maxima[global_row * stride_maxima],
			-OK_FLOAT_MAX);
	__syncthreads();

	if (col == 0)
		for (k = 0; k < blockDim.y; ++k)
			maxsub[row] = MATH(fmax)(maxsub[row],
				get(Asub, row, k, kTileLD));

	__syncthreads();

	maxima[global_col * stride_maxima] = maxsub[col];
}

static __global__ void __matrix_col_max(ok_float * maxima, ok_float * A,
	size_t i, size_t j, size_t ld, size_t stride_maxima,
	const enum CBLAS_ORDER ord)
{
	uint col, row, global_col, global_row, k;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float maxsub[kTileSize];

	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	get(Asub, row, col, kTileLD) = get(A, global_row, global_col, ld);
	__syncthreads();

	if (row == 0)
		maxsub[col] = MATH(fmax)(maxima[global_col * stride_maxima],
			-OK_FLOAT_MAX);
	__syncthreads();

	if (row == 0)
		for (k = 0; k < blockDim.x; ++k)
			maxsub[col] = MATH(fmax)(maxsub[col],
				get(Asub, k, col, kTileLD));
	__syncthreads();

	maxima[global_col * stride_maxima] = maxsub[col];
}

void linalg_matrix_reduce_indmin(void * linalg_handle, size_t * indices,
	vector * minima, matrix * A, const enum CBLAS_SIDE side)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_row_tiles, num_col_tiles, i, j;
	size_t ind_local[minima->size];

	void (* reduce)(size_t * minind, ok_float * minima, ok_float * A,
		size_t i, size_t j, size_t ld, size_t stride_indmin,
		size_t stride_minima, const enum CBLAS_ORDER);

	reduce = (side == CblasLeft) ? __matrix_col_indmin :
		__matrix_row_indmin;

	err = cublasGetStream(*(cublasHandle_t *) linalg_handle, &stm);
	num_row_tiles = (A->size1 + kTileSize - 1u) / kTileSize;
	num_col_tiles = (A->size2 + kTileSize - 1u) / kTileSize;

	vector_set_all(minima, OK_FLOAT_MAX);

	for (i = 0; i < num_row_tiles; ++i) {
		uint block_dim_v = kTileSize < A->size1 - i * kTileSize ? \
				   kTileSize : A->size1 - i * kTileSize;

		for (j = 0; j < num_col_tiles; ++j) {
			if (err != CUBLAS_STATUS_SUCCESS)
				break;

			uint block_dim_h =
				kTileSize < A->size2 - j * kTileSize ? \
				kTileSize : A->size2 - j * kTileSize;

			dim3 block_dim(block_dim_v, block_dim_h);
			reduce<<<1, block_dim, 0, stm>>>(ind_local,
				minima->data, A->data, i, j, (uint) A->ld, 1,
				minima->stride, A->order);
			CUDA_CHECK_ERR;
		}
	}
	ok_memcpy_gpu(indices, ind_local, minima->size * sizeof(*indices));
}

static void __matrix_extrema(void * linalg_handle, vector * extrema,
	matrix * A, const enum CBLAS_SIDE side, const int minima)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_row_tiles, num_col_tiles, i, j;
	uint ld = (uint) A->ld, stride = (uint) extrema->stride;

	void (* reduce)(ok_float * extrema, ok_float * A, size_t i, size_t j,
		size_t ld, size_t stride_maxima, const enum CBLAS_ORDER);

	if (minima)
		reduce = (side == CblasLeft) ? __matrix_col_min :
			__matrix_row_min;
	else
		reduce = (side == CblasLeft) ? __matrix_col_max :
			__matrix_row_max;

	err = cublasGetStream(*(cublasHandle_t *) linalg_handle, &stm);
	num_row_tiles = (A->size1 + kTileSize - 1u) / kTileSize;
	num_col_tiles = (A->size2 + kTileSize - 1u) / kTileSize;

	for (i = 0; i < num_row_tiles; ++i) {
		uint block_dim_v = kTileSize < A->size1 - i * kTileSize ? \
				   kTileSize : A->size1 - i * kTileSize;

		for (j = 0; j < num_col_tiles; ++j) {
			if (err != CUBLAS_STATUS_SUCCESS)
				break;

			uint block_dim_h =
				kTileSize < A->size2 - j * kTileSize ? \
				kTileSize : A->size2 - j * kTileSize;

			dim3 block_dim(block_dim_v, block_dim_h);
			reduce<<<1, block_dim, 0, stm>>>(extrema->data, A->data,
				i, j, ld, stride, A->order);
			CUDA_CHECK_ERR;
		}
	}
}

void linalg_matrix_reduce_min(void * linalg_handle, vector * minima,
	matrix * A, const enum CBLAS_SIDE side)
{
	vector_set_all(minima, OK_FLOAT_MAX);
	__matrix_extrema(linalg_handle, minima, A, side, 1);
}

void linalg_matrix_reduce_max(void * linalg_handle, vector * maxima,
	matrix * A, const enum CBLAS_SIDE side)
{
	vector_set_all(maxima, -OK_FLOAT_MAX);
	__matrix_extrema(linalg_handle, maxima, A, side, 0);
}

/* device reset */
ok_status ok_device_reset()
{
	cudaDeviceReset();
	CUDA_CHECK_ERR;
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
