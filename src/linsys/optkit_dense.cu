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
__global__ void __block_chol(ok_float * A, uint iter, uint ld,
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride))
{
	uint col, row, mat_dim, global_col, global_row, i;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float L[kTileLD * kTileSize];
	ok_float a11;

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

__global__ void __block_trsv(ok_float * A, uint iter, uint n, uint ld,
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride))
{
	uint tile_idx, row, global_row, global_col, i, j;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float L[kTileLD * kTileSize];
	__shared__ ok_float A12[kTileLD * kTileSize];

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
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(A->order == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

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
			A->order, get);
		CUDA_CHECK_ERR;

		if (i == num_tiles - 1u)
			break;

		/* L21 = A21 * L21^-T */
		grid_dim = num_tiles - i - 1u;
		matrix L21 = matrix_submatrix_gen(A, (i + 1) * kTileSize,
			i * kTileSize, A->size1 - (i + 1) * kTileSize,
			kTileSize);

		__block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data, i,
			(uint) A->size1, (uint) A->ld, A->order, get);
		CUDA_CHECK_ERR;

		/* A22 -= L21 * L21^T */
		matrix A22 = matrix_submatrix_gen(A, (i + 1) * kTileSize,
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

__global__ void __block_diag_gramian(ok_float * A, ok_float * v, uint i, uint j,
	uint ld, uint stride_v,
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride))
{
	uint global_col, global_row;

	global_row = i * kTileSize + threadIdx.x;
	global_col = j * kTileSize + threadIdx.y;

	v[global_row * stride_v] += get(A, global_row, global_col, ld) *
		get(A, global_row, global_col, ld);
}

void linalg_diag_gramian(void * linalg_handle, matrix * A, vector * v)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_row_tiles, num_col_tiles, grid_dim, i, j, size1, size2;
	uint ld = (uint) A->ld, stride = (uint) v->stride;
	int skinny = A->size1 == v->size;
	enum CBLAS_ORDER order;
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride);


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
	order = (skinny) ? A->order : CblasColMajor + CblasRowMajor - A->order;
	get = (order == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

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
				A->data, v->data, i, j, ld, stride, order, get);
			CUDA_CHECK_ERR;
		}
	}
}

__global__ void __matrix_scale_left(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v,
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride))
{
	uint global_col, global_row;

	global_row = i * kTileSize + threadIdx.x;
	global_col = j * kTileSize + threadIdx.y;

	get(A, global_row, global_col, ld) *= v[global_row * stride_v];
}

__global__ void __matrix_scale_right(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v,
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride))
{
	uint global_col, global_row;

	global_row = i * kTileSize + threadIdx.x;
	global_col = j * kTileSize + threadIdx.y;

	get(A, global_row, global_col, ld) *= v[global_col * stride_v];
}

__global__ void __matrix_add_left(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v,
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride))
{
	uint global_col, global_row;

	global_row = i * kTileSize + threadIdx.x;
	global_col = j * kTileSize + threadIdx.y;

	get(A, global_row, global_col, ld) += v[global_row * stride_v];
}

__global__ void __matrix_add_right(ok_float * A, const ok_float * v,
	uint i, uint j, size_t ld, size_t stride_v,
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride))
{
	uint global_col, global_row;

	global_row = i * kTileSize + threadIdx.x;
	global_col = j * kTileSize + threadIdx.y;

	get(A, global_row, global_col, ld) += v[global_col * stride_v];
}

void linalg_matrix_broadcast_vector(void * linalg_handle, matrix * A,
	const vector * v, enum OPTKIT_TRANSFORM operation, enum CBLAS_SIDE side)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_row_tiles, num_col_tiles, grid_dim, i, j, size1, size2;
	uint ld = (uint) A->ld, stride = (uint) v->stride;
	ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
		(A->order == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;
	void (*transform)(ok_float * A, const ok_float * v, uint i, uint j,
		size_t ld, size_t stride_v, ok_float& (* get)(ok_float * A,
		uint i, uint j, uint stride));

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
				j, ld, stride, order, get);
			CUDA_CHECK_ERR;
		}
	}
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
