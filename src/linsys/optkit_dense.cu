#include "optkit_defs_gpu.h"
#include "optkit_dense.h"

/* row major data retrieval */
__device__ inline ok_float& __matrix_get_r(ok_float *A, uint i, uint j,
	uint stride)
{
	return A[i * stride + j];
}

/* column major data retrieval */
__device__ inline ok_float& __matrix_get_c(ok_float *A, uint i, uint j,
	uint stride)
{
	return A[i + j * stride];
}

#ifdef __cplusplus
extern "C" {
#endif

void denselib_version(int *maj, int *min, int *change, int *status)
{
	*maj = OPTKIT_VERSION_MAJOR;
	*min = OPTKIT_VERSION_MINOR;
	*change = OPTKIT_VERSION_CHANGE;
	*status = (int) OPTKIT_VERSION_STATUS;
}

/* cholesky decomposition of a single block */
static __global__ void __block_chol(ok_float *A, uint iter, uint ld,
	const enum CBLAS_ORDER ord)
{
	uint col, row, mat_dim, global_col, global_row, i;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float L[kTileLD * kTileSize];
	ok_float a11;

	ok_float& (* get)(ok_float *A, uint i, uint j, uint stride) =
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

static __global__ void __block_trsv(ok_float *A, uint iter, uint n, uint ld,
	const enum CBLAS_ORDER ord)
{
	uint tile_idx, row, global_row, global_col, i, j;
	const uint kTileLD = kTileSize + 1u;
	__shared__ ok_float L[kTileLD * kTileSize];
	__shared__ ok_float A12[kTileLD * kTileSize];

	ok_float& (* get)(ok_float *A, uint i, uint j, uint stride) =
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
ok_status linalg_cholesky_decomp_flagged(void *blas_handle, matrix *A,
	int silence_domain_err)
{
	ok_status err;
	cudaStream_t stm;
	uint num_tiles, grid_dim, i;
	matrix L21, A22;

	if (!blas_handle)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	OK_CHECK_MATRIX(A);
	if (A->size1 != A->size2)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	err = OK_SCAN_CUBLAS( cublasGetStream(*(cublasHandle_t *) blas_handle,
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
		if (silence_domain_err) {
                        err = OK_STATUS_CUDA_QUIET;
                        if (err)
                        	return err;
		} else {

			OK_RETURNIF_ERR( OK_STATUS_CUDA );
		}

		if (i == num_tiles - 1u)
			break;

		/* L21 = A21 * L21^-T */
		grid_dim = num_tiles - i - 1u;
                L21.data = OK_NULL;
		OK_RETURNIF_ERR( matrix_submatrix(&L21, A, (i + 1) * kTileSize,
			i * kTileSize, A->size1 - (i + 1) * kTileSize,
			kTileSize) );

		if (!err)
			__block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data,
				i, (uint) A->size1, (uint) A->ld, A->order);
		cudaDeviceSynchronize();
		OK_RETURNIF_ERR( OK_STATUS_CUDA );

		/* A22 -= L21 * L21^T */
                A22.data = OK_NULL;
		OK_RETURNIF_ERR( matrix_submatrix(&A22, A, (i + 1) * kTileSize,
			(i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
			A->size1 - (i + 1) * kTileSize) );

		OK_RETURNIF_ERR( blas_syrk(blas_handle, CblasLower,
			CblasNoTrans, -kOne, &L21, kOne, &A22) );
	}
	return err;
}

ok_status linalg_cholesky_decomp(void *blas_handle, matrix *A)
{
	return linalg_cholesky_decomp_flagged(blas_handle, A, 0);
}

/* Cholesky solve */
ok_status linalg_cholesky_svx(void *blas_handle, const matrix *L, vector *x)
{
	OK_CHECK_MATRIX(L);
	OK_CHECK_VECTOR(x);

	if (!blas_handle)
		return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED );
	if (L->size1 != L->size2 || L->size1 != x->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_RETURNIF_ERR( blas_trsv(blas_handle, CblasLower, CblasNoTrans,
		CblasNonUnit, L, x) );
	return OK_SCAN_ERR( blas_trsv(blas_handle, CblasLower, CblasTrans,
		CblasNonUnit, L, x) );
}

#ifdef __cplusplus
}
#endif

namespace optkit {

template<typename T>
static T ok_float_max(void)
	{ return (sizeof(T) == sizeof(float)) ? FLT_MAX : DBL_MAX; }

template<typename T>
static T ok_float_min(void)
	{ return (sizeof(T) == sizeof(float)) ? -FLT_MAX : -DBL_MAX; }

static  uint calc_block_dim_(uint size) {
	uint blocksize = kBlockSize;
	while ( blocksize > size && blocksize > 2 * kWarpSize )
		blocksize >>= 1;
	return blocksize;
}

static inline uint calc_grid_dim_(uint size, uint block_size)
{
	uint grid_dim = (size + block_size - 1) / (block_size);
	return grid_dim < kMaxGridSize ? grid_dim : kMaxGridSize;
}

template <typename T>
static __global__ void strided_memcpy_(T *out, const uint stride_out,
	const T *in, const uint stride_in, uint size)
{
	const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint i;
	for (i = idx; i < size; i += gridDim.x * blockDim.x)
		out[i * stride_out] = in[i * stride_in];
}

template<typename T>
struct opUnaryAdd {
	T *const offset;
	opUnaryAdd(T *const offset) : offset(offset)
		{}
	__device__ void operator()(T & input)
		{ input += *offset; }
};

template<typename T>
struct opUnaryMul {
	T *const scaling;
	opUnaryMul(T *const scaling) : scaling(scaling)
		{}
	__device__ void operator()(T & input)
		{ input *= *scaling; }
};

template<typename T>
struct opAdd {
	__device__ void operator()(T *first, T *const second)
		{ *first += *second; }
};

template<typename T>
struct opAddSquare {
	__device__ void operator()(T *first, T *const second)
		{ *first += *second * (*second); }
};

template<typename T>
struct opMin {
	__device__ void operator()(T *first, T *const second)
		{ *first = *second < *first ? *second : *first; }
};

template<typename T>
struct opMax {
	__device__ void operator()(T *first, T *const second)
		{ *first = *second > *first ? *second : *first; }
};

template<typename T>
struct opMinIdx{
	__device__ void operator()(T *first, T *const second, size_t *idx1,
		size_t *const idx2)
	{
		if (*second < *first) {
			*first = *second;
			*idx1 = *idx2;
		}
	}
};

template <typename T, uint blockSize, typename loadingOp, typename reductionOp>
static __global__ void row_reduce(T *const in, uint rowstride_in,
	uint colstride_in, T *out, uint rowstride_out, uint colstride_out,
	uint row, uint n, loadingOp transform_reduce_, reductionOp reduce_,
	const T default_value)
{
	uint col = threadIdx.x;
	uint block_stride = blockSize * 2;
	uint grid_stride = block_stride * gridDim.x;
	uint global_col = blockIdx.x * block_stride + col;
	__shared__ T d[blockSize];

	d[col] = default_value;

	/* load and pre-reduce 1 in every <grid_stride> elements */
	while (global_col < n) {
		transform_reduce_(d + col, in + row * rowstride_in +
			global_col * colstride_in);
		if (global_col + blockSize < n) {
			transform_reduce_(d + col, in + row * rowstride_in +
				(global_col + blockSize) * colstride_in);
		}
		global_col += grid_stride;
	}
	__syncthreads();

	if (blockSize >= 1024) {
		if (col < 512)
			reduce_(d + col, d + col + 512);
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (col < 256)
			reduce_(d + col, d + col + 256);
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (col < 128)
			reduce_(d + col, d + col + 128);
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (col < 64)
			reduce_(d + col, d + col + 64);
		__syncthreads();
	}

	if (col < 32) {
		if (blockSize >= 64)
			reduce_(d + col, d + col + 32);
		__syncthreads();

		if (blockSize >= 32)
			reduce_(d + col, d + col + 16);
		__syncthreads();

		if (blockSize >= 16)
			reduce_(d + col, d + col + 8);
		__syncthreads();

		if (blockSize >= 8)
			reduce_(d + col, d + col + 4);
		__syncthreads();

		if (blockSize >= 4)
			reduce_(d + col, d + col + 2);
		__syncthreads();

		if (blockSize >= 2)
			reduce_(d + col, d + col + 1);
		__syncthreads();
	}

	if (col == 0)
		out[blockIdx.x * colstride_out + row * rowstride_out] = d[0];
}

template <typename T, typename loadingOp, typename reductionOp>
static ok_status reduction_innerloop(T *const A_k, uint rowstride_A,
	uint colstride_A, T *reduction_k, uint rowstride_r, uint colstride_r,
	uint nrows, uint cols_k, uint width_k, uint blocksize_k,
	loadingOp transform_reduce_, reductionOp reduce_, T default_value)
{
	uint row;

	for (row = 0; row < nrows; ++row) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		switch(blocksize_k) {
		case 32:
			row_reduce<T, 32, loadingOp, reductionOp>
			<<<width_k, blocksize_k, 0, s>>>(A_k,
				rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, row, cols_k,
				transform_reduce_, reduce_, default_value);
			break;
		case 64:
			row_reduce<T, 64, loadingOp, reductionOp>
			<<<width_k, blocksize_k, 0, s>>>(A_k,
				rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, row, cols_k,
				transform_reduce_, reduce_, default_value);
			break;
		case 128:
			row_reduce<T, 128, loadingOp, reductionOp>
			<<<width_k, blocksize_k, 0, s>>>(A_k,
				rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, row, cols_k,
				transform_reduce_, reduce_, default_value);
			break;
		case 256:
			row_reduce<T, 256, loadingOp, reductionOp>
			<<<width_k, blocksize_k, 0, s>>>(A_k,
				rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, row, cols_k,
				transform_reduce_, reduce_, default_value);
			break;
		case 512:
			row_reduce<T, 512, loadingOp, reductionOp>
			<<<width_k, blocksize_k, 0, s>>>(A_k,
				rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, row, cols_k,
				transform_reduce_, reduce_, default_value);
			break;
		default:
			row_reduce<T, 1024, loadingOp, reductionOp>
			<<<width_k, blocksize_k, 0, s>>>(A_k,
				rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, row, cols_k,
				transform_reduce_, reduce_, default_value);
			break;
		}
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

template <typename T, typename loadingOp, typename reductionOp>
static ok_status matrix_row_reduce(T *const A, size_t nrows, size_t ncols,
	size_t rowstride_A, size_t colstride_A, T *output, size_t stride_out,
	loadingOp transform_reduce_, reductionOp reduce_, T default_value)
{
	ok_status err = OPTKIT_SUCCESS;
	T *A_k;
	T *reduction, *reduction_k;

	uint nreductions = 0, nlaunches = 0, k;
	uint width_k, blocksize_k, cols_k;
	uint rowstride_r = 0, colstride_r = 1;
	uint grid_dim_copy;

	cols_k = ncols;
	while (cols_k > 1) {
		blocksize_k = calc_block_dim_(cols_k);
		cols_k = calc_grid_dim_(cols_k, 2 * blocksize_k);
		nreductions += cols_k;
		++nlaunches;
	}
	rowstride_r = nreductions;

	// reduction_host = (T *) malloc(nrows * nreductions * sizeof(T));
	OK_CHECK_ERR( err,
		ok_alloc_gpu(reduction, nrows * nreductions * sizeof(T)) );

	A_k = A;
	reduction_k = reduction;
	cols_k = ncols;

	for (k = 0; k < nlaunches && !err; ++k) {
		blocksize_k = calc_block_dim_(cols_k);
		width_k = calc_grid_dim_(cols_k, 2 * blocksize_k);

		if (k == 0)
			err = reduction_innerloop<T, loadingOp, reductionOp>(
				A_k, rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, nrows, cols_k,
				width_k, blocksize_k, transform_reduce_,
				reduce_, default_value);
		else
			err = reduction_innerloop<T, reductionOp, reductionOp >(
				A_k, rowstride_A, colstride_A, reduction_k,
				rowstride_r, colstride_r, nrows, cols_k,
				width_k, blocksize_k, reduce_, reduce_,
				default_value);

		OK_SCAN_ERR( err );

		A_k = reduction_k;
		reduction_k += colstride_r * width_k;
		cols_k = width_k;
		rowstride_A = rowstride_r;
		colstride_A = colstride_r;
	}

	if (!err) {
		grid_dim_copy = calc_grid_dim_(nrows, kBlockSize);
		strided_memcpy_<T><<<grid_dim_copy, kBlockSize>>>(output,
			stride_out, A_k, rowstride_A, nrows);
		err = OK_STATUS_CUDA;
	}

	OK_MAX_ERR( err,
		ok_free_gpu(reduction) );

	return err;
}

template <typename T, typename unaryOp>
static __global__ void row_transform(T *in, uint rowstride_in,
	uint colstride_in, uint row, uint n, unaryOp transform_)
{
	uint block_stride = blockDim.x;
	uint grid_stride = block_stride * gridDim.x;
	uint gc, global_col = blockIdx.x * block_stride + threadIdx.x;
	T *in_ = in + row * rowstride_in;

	/* transform 1 in every <grid_stride> elements */
	for (gc = global_col; gc < n; gc += grid_stride)
		transform_(in_[gc * colstride_in]);
}


template<typename T>
static ok_status matrix_vector_op_setdims(const matrix_<T> *A,
	const vector_<T> *v, size_t *size1, size_t *size2, size_t *stride1,
	size_t *stride2, const int manipulate_by_row)
{
	OK_CHECK_MATRIX(A);
	OK_CHECK_VECTOR(v);
	const int rowmajor = A->order == CblasRowMajor;
	/*
	 * logic for row/col transformations & reductions:
	 *	side = left: for each column of A, iterate over rows
	 *	side = right: for each row of A, iterate over columns
	 */
	*size1 = (manipulate_by_row) ? A->size1 : A->size2;
	*size2 = (manipulate_by_row) ? A->size2 : A->size1;
	*stride1 = (manipulate_by_row == rowmajor) ? A->ld : 1;
	*stride2 = (manipulate_by_row == rowmajor) ? 1 : A->ld;
	if (*size1 != v->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );
	return OPTKIT_SUCCESS;
}

template <typename T, size_t blockSize>
static __global__ void row_indmin(T *const in, size_t *const idx_in,
	size_t rowstride_in, size_t colstride_in, T *out, size_t *idx,
	size_t rowstride_out, size_t colstride_out, size_t row, size_t n,
	const T default_value, const int first_pass)
{
	size_t col = threadIdx.x;
	size_t block_stride = blockSize * 2;
	size_t grid_stride = block_stride * gridDim.x;
	size_t global_col = blockIdx.x * block_stride + col;
	size_t global_col_half = global_col + blockSize;
	__shared__ T d[blockSize];
	__shared__ size_t id[blockSize];
	T *in_ = in + row * rowstride_in;
	size_t *idx_in_ = idx_in + row * rowstride_in;
	opMinIdx<T> min_;

	d[col] = default_value;

	/* load and pre-reduce 1 in every <grid_stride> elements */

	if (first_pass)
		while (global_col < n) {
			min_(d + col, in_ + global_col * colstride_in, id + col,
				&global_col);
			if (global_col + blockSize < n) {
				min_(d + col, in_ + global_col_half *
					colstride_in, id +col,
					&global_col_half);
			}
			global_col += grid_stride;
			global_col_half += grid_stride;
		}
	else
		while (global_col < n) {
			min_(d + col, in_ + global_col * colstride_in, id + col,
				idx_in_ + global_col * colstride_in);
			if (global_col + blockSize < n) {
				min_(d + col, in_ + global_col_half *
					colstride_in, id + col, idx_in_ +
					global_col_half * colstride_in);
			}
			global_col += grid_stride;
			global_col_half += grid_stride;
		}
	__syncthreads();

	if (blockSize >= 1024) {
		if (col < 512)
			min_(d + col, d + col + 512, id + col, id + col + 512);
		__syncthreads();
	}

	if (blockSize >= 512) {
		if (col < 256)
			min_(d + col, d + col + 256, id + col, id + col + 256);
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (col < 128)
			min_(d + col, d + col + 128, id + col, id + col + 128);
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (col < 64)
			min_(d + col, d + col + 64, id + col, id + col + 64);
		__syncthreads();
	}

	if (col < 32) {
		if (blockSize >= 64)
			min_(d + col, d + col + 32, id + col, id + col + 32);
		__syncthreads();

		if (blockSize >= 32)
			min_(d + col, d + col + 16, id + col, id + col + 16);
		__syncthreads();

		if (blockSize >= 16)
			min_(d + col, d + col + 8, id + col, id + col + 8);
		__syncthreads();

		if (blockSize >= 8)
			min_(d + col, d + col + 4, id + col, id + col + 4);
		__syncthreads();

		if (blockSize >= 4)
			min_(d + col, d + col + 2, id + col, id + col + 2);
		__syncthreads();

		if (blockSize >= 2)
			min_(d + col, d + col + 1, id + col, id + col + 1);
		__syncthreads();
	}

	if (col == 0) {
		out[blockIdx.x * colstride_out + row * rowstride_out] = d[0];
		idx[blockIdx.x * colstride_out + row * rowstride_out] = id[0];
	}
}

template <typename T>
static ok_status indmin_innerloop(T *const A_k, size_t *const idx_k,
	uint rowstride_A, uint colstride_A, T *reduction_k,
	size_t *idx_reduction_k, uint rowstride_r, uint colstride_r,
	uint nrows, uint cols_k, uint width_k, uint blocksize_k,
	T default_value, const int first_pass)
{
	uint row;

	for (row = 0; row < nrows; ++row) {
		cudaStream_t s;
		cudaStreamCreate(&s);
		switch(blocksize_k) {
		case 32:
			row_indmin<T, 32><<<width_k, blocksize_k, 0, s>>>(A_k,
				idx_k, rowstride_A, colstride_A, reduction_k,
				idx_reduction_k, rowstride_r, colstride_r, row,
				cols_k, default_value, first_pass);
			break;
		case 64:
			row_indmin<T, 64><<<width_k, blocksize_k, 0, s>>>(A_k,
				idx_k, rowstride_A, colstride_A, reduction_k,
				idx_reduction_k, rowstride_r, colstride_r, row,
				cols_k, default_value, first_pass);
			break;
		case 128:
			row_indmin<T, 128><<<width_k, blocksize_k, 0, s>>>(A_k,
				idx_k, rowstride_A, colstride_A, reduction_k,
				idx_reduction_k, rowstride_r, colstride_r, row,
				cols_k, default_value, first_pass);
			break;
		case 256:
			row_indmin<T, 256><<<width_k, blocksize_k, 0, s>>>(A_k,
				idx_k, rowstride_A, colstride_A, reduction_k,
				idx_reduction_k, rowstride_r, colstride_r, row,
				cols_k, default_value, first_pass);
			break;
		case 512:
			row_indmin<T, 512><<<width_k, blocksize_k, 0, s>>>(A_k,
				idx_k, rowstride_A, colstride_A, reduction_k,
				idx_reduction_k, rowstride_r, colstride_r, row,
				cols_k, default_value, first_pass);
			break;
		default:
			row_indmin<T, 1024><<<width_k, blocksize_k, 0, s>>>(A_k,
				idx_k, rowstride_A, colstride_A, reduction_k,
				idx_reduction_k, rowstride_r, colstride_r, row,
				cols_k, default_value, first_pass);
			break;
		}
		cudaStreamDestroy(s);
	}
	cudaDeviceSynchronize();
	return OK_STATUS_CUDA;
}

template <typename T>
static ok_status matrix_row_indmin(T *const A, size_t nrows, size_t ncols,
	size_t rowstride_A, size_t colstride_A, T *minima, size_t stride_min,
	size_t *indices, size_t stride_idx, T default_value)
{
	ok_status err = OPTKIT_SUCCESS;
	T *A_k;
	T *reduction, *reduction_k;
	size_t *idx_k, *idx_reduction, *idx_reduction_k;

	uint nreductions = 0, nlaunches = 0, k;
	uint width_k, blocksize_k, cols_k;
	uint rowstride_r = 0, colstride_r = 1;
	uint grid_dim_copy;

	cols_k = ncols;
	while (cols_k > 1) {
		blocksize_k = calc_block_dim_(cols_k);
		cols_k = calc_grid_dim_(cols_k, 2 * blocksize_k);
		nreductions += cols_k;
		++nlaunches;
	}
	rowstride_r = nreductions;

	// reduction_host = (T *) malloc(nrows * nreductions * sizeof(T));
	OK_CHECK_ERR( err,
		ok_alloc_gpu(reduction, nrows * nreductions * sizeof(T)) );
	OK_CHECK_ERR( err,
		ok_alloc_gpu(idx_reduction, nrows * nreductions *
			sizeof(size_t)) );

	A_k = A;
	idx_k = OK_NULL;
	reduction_k = reduction;
	idx_reduction_k = idx_reduction;
	cols_k = ncols;

	for (k = 0; k < nlaunches && !err; ++k) {
		blocksize_k = calc_block_dim_(cols_k);
		width_k = calc_grid_dim_(cols_k, 2 * blocksize_k);

		OK_CHECK_ERR( err,
			indmin_innerloop<T>(A_k, idx_k, rowstride_A,
				colstride_A, reduction_k, idx_reduction_k,
				rowstride_r, colstride_r, nrows, cols_k,
				width_k, blocksize_k, default_value, k == 0) );

		A_k = reduction_k;
		idx_k = idx_reduction_k;
		reduction_k += colstride_r * width_k;
		idx_reduction_k += colstride_r * width_k;
		cols_k = width_k;
		rowstride_A = rowstride_r;
		colstride_A = colstride_r;
	}

	if (!err) {
		grid_dim_copy = calc_grid_dim_(nrows, kBlockSize);
		strided_memcpy_<T><<<grid_dim_copy, kBlockSize>>>(minima,
			stride_min, A_k, rowstride_A, nrows);
		err = OK_STATUS_CUDA;
	}

	if (!err) {
		idx_reduction_k = idx_reduction + (nreductions - 1) *
			colstride_r;
		grid_dim_copy = calc_grid_dim_(nrows, kBlockSize);
		strided_memcpy_<size_t><<<grid_dim_copy, kBlockSize>>>(indices,
			stride_idx, idx_reduction_k, rowstride_r, nrows);
		err = OK_STATUS_CUDA;
	}
	OK_MAX_ERR( err,
		ok_free_gpu(reduction) );
	OK_MAX_ERR( err,
		ok_free_gpu(idx_reduction) );

	return err;
}

} /* namespace optkit */

template<typename T>
static ok_status __linalg_matrix_row_squares(const enum CBLAS_TRANSPOSE t,
	const matrix_<T> *A, vector_<T> *v)
{
	/*
	 *	transpose: v_i = multiply A^T * A: work with columns of A
	 *	non-transpose: multiply A * A^T: work with rows of A
	 *		(columns of A^T)
	 */
	ok_status err = OPTKIT_SUCCESS;
	const int square_row_entries = t == CblasNoTrans;
	size_t size1, size2, row_stride, col_stride;

	OK_RETURNIF_ERR(
		optkit::matrix_vector_op_setdims<T>(A, v, &size1, &size2,
			&row_stride, &col_stride, square_row_entries) );

	err = optkit::matrix_row_reduce<T, optkit::opAddSquare<T>,
		optkit::opAdd<T> >(A->data, size1, size2, row_stride,
			col_stride, v->data, v->stride,
			optkit::opAddSquare<T>(), optkit::opAdd<T>(),
			static_cast<T>(0));
	return OK_SCAN_ERR( err );
}

/*
 * perform
 *
 *	A = diag(v) *A, 	operation == OkTransformScale, side = CblasLeft
 *	A = A * diag(v), 	operation == OkTransformScale, side = CblasRight
 *
 *	A += v * 1^T, 		operation == OkTransformAdd, side == CblasLeft
 *	A += 1 * v^T,		operation == OkTransformAdd, side == CblasRight
 *
 */
template<typename T>
static ok_status __linalg_matrix_broadcast_vector(matrix_<T> *A,
	const vector_<T> *v, const enum OPTKIT_TRANSFORM operation,
	const enum CBLAS_SIDE side)
{
	/*
	 * logic for row/col broadcast:
	 *	side = left: broadcast to each row of A^T
	 *	side = right: broadcast to each row of A
	 */

	size_t row, grid_dim, size1, size2, row_stride, col_stride;
	const int broadcast_by_row = side == CblasLeft;
	OK_RETURNIF_ERR(
		optkit::matrix_vector_op_setdims<T>(A, v, &size1, &size2,
			&row_stride, &col_stride, broadcast_by_row) );

	grid_dim = calc_grid_dim(size2);

	switch (operation) {
	case OkTransformScale :
		for (row = 0; row < size1; ++row) {
			cudaStream_t s;
			cudaStreamCreate(&s);
			optkit::opUnaryMul<T> mul_(v->data + row * v->stride);
			optkit::row_transform<T, optkit::opUnaryMul<T> >
			<<<grid_dim, kBlockSize, 0, s>>>(A->data, row_stride,
				col_stride, row, size2, mul_);
			cudaStreamDestroy(s);
		}
		cudaDeviceSynchronize();
		break;
	case OkTransformAdd :
		for (row = 0; row < size1; ++row) {
			cudaStream_t s;
			cudaStreamCreate(&s);
			optkit::opUnaryAdd<T> add_(v->data + row * v->stride);
			optkit::row_transform<T, optkit::opUnaryAdd<T> >
			<<<grid_dim, kBlockSize, 0, s>>>(A->data, row_stride,
				col_stride, row, size2, add_);
			cudaStreamDestroy(s);
		}
		cudaDeviceSynchronize();
		break;
	default :
		return OK_SCAN_ERR( OPTKIT_ERROR_DOMAIN );
	}

	return OK_STATUS_CUDA;
}

template<typename T>
static ok_status __linalg_matrix_reduce_indmin(vector_<size_t> *indices,
	vector_<T> *minima, const matrix_<T> *A, const enum CBLAS_SIDE side)
{
	ok_status err = OPTKIT_SUCCESS;
	const int reduce_by_row = side == CblasRight;
	size_t size1, size2, row_stride = 0, col_stride = 0;

	OK_CHECK_VECTOR(indices);
	if (indices->size != minima->size)
		return OK_SCAN_ERR( OPTKIT_ERROR_DIMENSION_MISMATCH );

	OK_RETURNIF_ERR(
		optkit::matrix_vector_op_setdims<T>(A, minima, &size1, &size2,
			&row_stride, &col_stride, reduce_by_row) );

	err = optkit::matrix_row_indmin<T>(A->data, size1, size2, row_stride,
			col_stride, minima->data, minima->stride, indices->data,
			indices->stride, optkit::ok_float_max<T>());
	return OK_SCAN_ERR( err );
}

template<typename T>
static ok_status __linalg_matrix_reduce_min(vector_<T> *minima,
	const matrix_<T> *A, const enum CBLAS_SIDE side)
{
	ok_status err = OPTKIT_SUCCESS;
	const int reduce_by_row = side == CblasRight;
	size_t size1, size2, row_stride, col_stride;

	OK_RETURNIF_ERR(
		optkit::matrix_vector_op_setdims<T>(A, minima, &size1, &size2,
			&row_stride, &col_stride, reduce_by_row) );

	err = optkit::matrix_row_reduce<T, optkit::opMin<T>, optkit::opMin<T> >(
		A->data, size1, size2, row_stride, col_stride, minima->data,
		minima->stride, optkit::opMin<T>(), optkit::opMin<T>(),
		optkit::ok_float_max<T>());
	return OK_SCAN_ERR( err );
}

template<typename T>
static ok_status __linalg_matrix_reduce_max(vector_<T> *maxima,
	const matrix_<T> *A, const enum CBLAS_SIDE side)
{
	ok_status err = OPTKIT_SUCCESS;
	const int reduce_by_row = side == CblasRight;
	size_t size1, size2, row_stride, col_stride;

	OK_RETURNIF_ERR(
		optkit::matrix_vector_op_setdims<T>(A, maxima, &size1, &size2,
			&row_stride, &col_stride, reduce_by_row) );

	err = optkit::matrix_row_reduce<T, optkit::opMax<T> >(A->data, size1,
		size2, row_stride, col_stride, maxima->data, maxima->stride,
		optkit::opMax<T>(), optkit::opMax<T>(),
		optkit::ok_float_min<T>());
	return OK_SCAN_ERR( err );
}

#ifdef __cplusplus
extern "C" {
#endif

ok_status linalg_matrix_row_squares(const enum CBLAS_TRANSPOSE t,
	const matrix *A, vector *v)
{
	return OK_SCAN_ERR( __linalg_matrix_row_squares<ok_float>(t, A, v) );
}

ok_status linalg_matrix_broadcast_vector(matrix *A, const vector *v,
	const enum OPTKIT_TRANSFORM operation, const enum CBLAS_SIDE side)
{
	return OK_SCAN_ERR( __linalg_matrix_broadcast_vector<ok_float>(A, v,
		operation, side) );
}

ok_status linalg_matrix_reduce_indmin(indvector *indices, vector *minima,
	const matrix *A, const enum CBLAS_SIDE side)
{
	return OK_SCAN_ERR( __linalg_matrix_reduce_indmin<ok_float>(indices,
		minima, A, side) );
}

ok_status linalg_matrix_reduce_min(vector *minima, const matrix *A,
	const enum CBLAS_SIDE side)
{
	return OK_SCAN_ERR( __linalg_matrix_reduce_min<ok_float>(minima, A,
		side) );
}

ok_status linalg_matrix_reduce_max(vector *maxima, const matrix *A,
	const enum CBLAS_SIDE side)
{
	return OK_SCAN_ERR( __linalg_matrix_reduce_max<ok_float>(maxima, A,
		side) );
}

#ifdef __cplusplus
}
#endif
