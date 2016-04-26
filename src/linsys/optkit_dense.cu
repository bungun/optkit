#include "optkit_defs_gpu.h"
#include "optkit_dense.h"


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
ok_status linalg_cholesky_decomp(void * linalg_handle, matrix * A)
{
	cublasStatus_t err;
	cudaStream_t stm;
	uint num_tiles, grid_dim, i;
	matrix L21, A22;

	if (!linalg_handle)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_MATRIX(A);
	if (A->size1 != A->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	err = OK_SCAN_CUBLAS( cublasGetStream(*(cublasHandle_t *) linalg_handle,
		&stm) );
	num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;

	for (i = 0; i < num_tiles && !err; ++i) {
		/* L11 = chol(A11) */
		uint block_dim_1d = kTileSize < A->size1 - i * kTileSize ? \
				    kTileSize : A->size1 - i * kTileSize;
		dim3 block_dim(block_dim_1d, block_dim_1d);

		if (!err)
			__block_chol<<<1, block_dim, 0, stm>>>(A->data, i,
				(uint) A->ld, A->order);
		cudaDeviceSynchronize();
		OK_RETURNIF_ERR( OK_STATUS_CUDA )

		if (i == num_tiles - 1u)
			break;

		/* L21 = A21 * L21^-T */
		grid_dim = num_tiles - i - 1u;
		OK_RETURNIF_ERR( matrix_submatrix(&L21, A, (i + 1) * kTileSize,
			i * kTileSize, A->size1 - (i + 1) * kTileSize,
			kTileSize) );

		if (!err)
			__block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data,
				i, (uint) A->size1, (uint) A->ld, A->order);
		cudaDeviceSynchronize();
		OK_RETURNIF_ERR( OK_STATUS_CUDA )

		/* A22 -= L21 * L21^T */
		OK_RETURNIF_ERR( matrix_submatrix(&A22, A, (i + 1) * kTileSize,
			(i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
			A->size1 - (i + 1) * kTileSize) );

		OK_RETURNIF_ERR( blas_syrk(linalg_handle, CblasLower,
			CblasNoTrans, -kOne, &L21, kOne, &A22) );
	}
	return err;
}

/* Cholesky solve */
ok_status linalg_cholesky_svx(void * linalg_handle, const matrix * L,
	vector * x)
{
	OK_CHECK_MATRIX(L);
	OK_CHECK_VECTOR(x);

	if (!linalg_handle)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (L->size1 != L->size2 || L->size1 != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_RETURNIF_ERR( err, blas_trsv(linalg_handle, CblasLower, CblasNoTrans,
		CblasNonUnit, L, x) );
	return blas_trsv(linalg_handle, CblasLower, CblasTrans,
		CblasNonUnit, L, x) );
}

static __global__ void __block_col_squares(const ok_float * A,
	const size_t size1, const size_t size2, const size_t row_stride,
	const size_t col_stride, ok_float * v, const size_t stride_v,
	const size_t i, const size_t j)
{
	uint col, row, global_col, global_row;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float vsub[kTileSize];

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	if (global_row >= size1 || global_col >= size2)
		return;

	Asub[row * kTileLD + col] = A[global_row * row_stride +
		global_col * col_stride];
	if (row == 0)
		vsub[row] = v[global_row * stride_v];
	__syncthreads();

	vsub[row] += Asub[row * kTileLD + col] * Asub[row * kTileLD + col];
	__syncthreads();

	v[global_row * stride_v] = vsub[row];
}

ok_status linalg_matrix_row_squares(const enum CBLAS_TRANSPOSE t,
	const matrix * A, vector * v)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);

	uint i, j;
	uint block_size = kTiles2D * kTileSize;

	dim3 grid_dim(kTileSize, kTileSize);
	dim3 blk_dim(kTiles2D, kTiles2D);

	int transpose = t == CblasTrans;
	int rowmajor = A->order == CblasRowMajor;

	/*
	 *	transpose: multiply A^T * A: work with columns of A
	 *	non-transpose: multiply A * A^T: work with rows of A
	 *		(columns of A^T)
	 */
	size_t size1 = (transpose) ? A->size1 : A->size2;
	size_t size2 = (transpose) ? A->size2 : A->size1;
	size_t row_stride = (transpose == rowmajor) ? A->ld : 1;
	size_t col_stride = (transpose == rowmajor) ? 1 : A->ld;

	if (v->size != size1)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	/* transform one bundle of [block_size] columns x M rows per stream */
	for (j = 0; j < size2; j += block_size) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		for (i = 0; i < size1; i += block_size)
			__block_col_squares<<<grid_dim, blk_dim, 0, s>>>(
				A->data, size1, size2, row_stride, col_stride,
				v->data, v->stride, i, j);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

static __device__ void __entry_add(ok_float * data, const size_t row,
	const size_t col, const size_t stride_r, const size_t stride_c,
	const ok_float value)
{
	data[row * stride_r + col * stride_c] += value;
}

static __device__ void __entry_mul(ok_float * data, const size_t row,
	const size_t col, const size_t stride_r, const size_t stride_c,
	const ok_float value)
{
	data[row * stride_r + col * stride_c] *= value;
}

static __global__ void __matrix_broadcast_vector(ok_float * A,
	const size_t size1, const size_t size2, const size_t row_stride,
	const size_t col_stride, const ok_float * v, const size_t stride_v,
	const size_t i, const size_t j,
	void (* inplace_op_)(ok_float * data, const size_t row,
	const size_t col, const size_t stride_r, const size_t stride_c,
	const ok_float value))
{
	uint col, row, global_col, global_row;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float vsub[kTileSize];

	row = threadIdx.x;
	col = threadIdx.y;
	global_row = i * kTileSize + row;
	global_col = j * kTileSize + col;

	if (global_row >= size1 || global_col >= size2)
		return;

	Asub[row * kTileLD + col] = A[global_row * row_stride +
		global_col * col_stride];
	if (col == 0)
		vsub[row] = v[global_row * stride_v];
	__syncthreads();

	inplace_op_(Asub, row, col, kTileLD, 1, vsub[row]);
	__syncthreads();

	A[global_row * row_stride + global_col * col_stride] =
		Asub[row * kTileLD + col];
}

ok_status linalg_matrix_broadcast_vector(matrix * A, const vector * v,
	const enum OPTKIT_TRANSFORM operation, const enum CBLAS_SIDE side)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);

	uint i, j;
	uint block_size = kTiles2D * kTileSize;

	dim3 grid_dim(kTileSize, kTileSize);
	dim3 blk_dim(kTiles2D, kTiles2D);
	int left = side == CblasLeft;
	int rowmajor = A->order == CblasRowMajor;

	/*
	 * logic for row/col broadcast:
	 *	side = left: broadcast to each row of A^T
	 *	side = right: broadcast to each row of A
	 */
	size_t size1 = (left) ? A->size1 : A->size2;
	size_t size2 = (left) ? A->size2 : A->size1;
	size_t row_stride = (left == rowmajor) ? A->ld : 1;
	size_t col_stride = (left == rowmajor) ? 1 : A->ld;
	void (*transform)(ok_float * data, const size_t row, const size_t col,
		const size_t stride_r, const size_t stride_c,
		const ok_float value) = (operation == OkTransformScale) ?
		__entry_mul : __entry_add;

	if (size1 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	/* transform one bundle of [block_size] rows x N cols per stream */
	for (i = 0; i < size1; i += block_size) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		for (j = 0; j < size2; j += block_size)
			__matrix_broadcast_vector<<<grid_dim, blk_dim, 0, s>>>(
				A->data, size1, size2, row_stride, col_stride,
				v->data, v->stride, i, j, transform);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

static __global__ void __matrix_row_indmin(size_t * indmin,
	const size_t stride_indmin, ok_float * minima, const size_t stride_min,
	const ok_float * A, const size_t size1, const size_t size2,
	const size_t row_stride, const size_t col_stride, const size_t i,
	const size_t j)
{
	uint row = threadIdx.x;
	uint col = threadIdx.y;
	uint global_row = i + row;
	uint global_col = j + col;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float minsub[kTileSize];
	__shared__ size_t indminsub[kTileSize];
	ok_float previous;

	if (global_row >= size1 || global_col >= size2)
		return;

	Asub[row * kTileLD + col] = A[global_row * row_stride +
		global_col * col_stride];
	__syncthreads();

	if (col == 0)
		minsub[row] = MATH(fmin)(minima[global_row * stride_min],
			OK_FLOAT_MAX);
	__syncthreads();

	previous = minsub[row];
	minsub[row] = MATH(fmin)(minsub[row], Asub[row * kTileLD + col]);
	if (minsub[row] != previous)
		indminsub[row] = global_col;
	__syncthreads();

	if (col == 0)
		minima[global_row * stride_min] = minsub[row];
	__syncthreads();

	if (col == 0)
		indmin[global_row * stride_indmin] = indminsub[row];
}

static __global__ void __matrix_row_reduce(ok_float * reduced,
	const size_t stride, const ok_float * A, const size_t size1,
	const size_t size2, const size_t row_stride, const size_t col_stride,
	const size_t i, const size_t j, const ok_float default_value,
	ok_float (* binary_op_)(const ok_float first, const ok_float second))
{
	uint row = threadIdx.x;
	uint col = threadIdx.y;
	uint global_row = i + row;
	uint global_col = j + col;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float Asub[kTileLD * kTileSize];
	__shared__ ok_float reduced_sub[kTileSize];

	if (global_row >= size1 || global_col >= size2)
		return;

	Asub[row * kTileLD + col] = A[global_row * row_stride +
		global_col * col_stride];
	__syncthreads();

	if (col == 0)
		reduced_sub[row] = binary_op_(reduced[global_row * stride],
			default_value);
	__syncthreads();

	reduced_sub[row] = binary_op_(reduced_sub[row],
		Asub[row * kTileLD + col]);
	__syncthreads();

	if (col == 0)
		reduced[global_row * stride] = reduced_sub[row];
}

ok_status linalg_matrix_reduce_indmin(indvector * indices, vector * minima,
	matrix * A, const enum CBLAS_SIDE side)
{
	OK_CHECK_VECTOR(indices);
	OK_CHECK_VECTOR(minima);
	OK_CHECK_MATRIX(A);

	uint i, j;
	uint block_size = kTiles2D * kTileSize;

	dim3 grid_dim(kTileSize, kTileSize);
	dim3 blk_dim(kTiles2D, kTiles2D);
	int left = side == CblasLeft;
	int rowmajor = A->order == CblasRowMajor;

	/*
	 * logic for row/col reduction:
	 *	side = left: reduce each row of A^T 	(analoguous to A^T * 1)
	 *	side = right: reduce each row of A 	(analogous to A * 1)
	 */
	size_t size1 = (left) ? A->size1 : A->size2;
	size_t size2 = (left) ? A->size2 : A->size1;
	size_t row_stride = (left == rowmajor) ? A->ld : 1;
	size_t col_stride = (left == rowmajor) ? 1 : A->ld;

	if (size2 != minima->size || indices->size != minima->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_RETURNIF_ERR( vector_set_all(minima, OK_FLOAT_MAX) );

	/* reduce one bundle of [block_size] rows x N cols per stream */
	for (i = 0; i < size1; i += block_size) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		for (j = 0; j < size2; j += block_size)
			__matrix_row_indmin<<<grid_dim, blk_dim, 0, s>>>(
				indices->data, indices->stride, minima->data,
				minima->stride, A->data, size1, size2,
				row_stride, col_stride, i, j);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

static ok_status __matrix_reduce_binary(vector * reduced, matrix * A,
	const enum CBLAS_SIDE side, const ok_float default_value,
	ok_float (* binary_op_)(const ok_float first, const ok_float second))
{
	OK_CHECK_VECTOR(reduced);
	OK_CHECK_MATRIX(A);

	uint i, j;
	uint block_size = kTiles2D * kTileSize;

	dim3 grid_dim(kTileSize, kTileSize);
	dim3 blk_dim(kTiles2D, kTiles2D);
	int left = side == CblasLeft;
	int rowmajor = A->order == CblasRowMajor;

	/*
	 * logic for row/col reduction:
	 *	side = left: reduce each row of A^T 	(analoguous to A^T * 1)
	 *	side = right: reduce each row of A 	(analogous to A * 1)
	 */
	size_t size1 = (left) ? A->size1 : A->size2;
	size_t size2 = (left) ? A->size2 : A->size1;
	size_t row_stride = (left == rowmajor) ? A->ld : 1;
	size_t col_stride = (left == rowmajor) ? 1 : A->ld;

	if (size2 != reduced->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_RETURNIF_ERR( vector_set_all(reduced, default_value) );

	/* reduce one bundle of [block_size] rows x N cols per stream */
	for (i = 0; i < size1; i += block_size) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		for (j = 0; j < size2; j += block_size)
			__matrix_row_reduce<<<grid_dim, blk_dim, 0, s>>>(
				reduced->data, reduced->stride, A->data, size1,
				size2, row_stride, col_stride, i, j,
				default_value, binary_op_);
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

ok_status linalg_matrix_reduce_min(vector * minima, matrix * A,
	const enum CBLAS_SIDE side)
{
	return __matrix_reduce_binary(minima, A, side, OK_FLOAT_MAX,
		MATH(fmin));
}

ok_status linalg_matrix_reduce_max(vector * maxima, matrix * A,
	const enum CBLAS_SIDE side)
{
	return __matrix_reduce_binary(maxima, A, side, -OK_FLOAT_MAX,
		MATH(fmax));
}

#ifdef __cplusplus
}
#endif
