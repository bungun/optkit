#include "optkit_dense.h"
#include "optkit_defs_gpu.h"
#include "optkit_thrust.hpp"


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


__global__ void _get_cuda_nan(ok_float * val)
{
        *val = OK_CUDA_NAN;
}

inline ok_float get_cuda_nan()
{
        ok_float res;
        ok_float * res_dev;

        ok_alloc_gpu(res_dev, 1 * sizeof(ok_float));
        cudaMemcpy(&res, res_dev, 1 * sizeof(ok_float), cudaMemcpyDeviceToHost);
        ok_free_gpu(res_dev);

        return res;
}

/*
 * VECTOR CUDA helper methods
 * ==========================
 */
__global__ void __vector_set(ok_float * data, ok_float val, size_t stride,
        size_t size)
{
        uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
        data[i * stride] = val;
}

void __vector_set_all(vector * v, ok_float x)
{
        uint grid_dim = calc_grid_dim(v->size);
        __vector_set<<<grid_dim, kBlockSize>>>(v->data, x, v->stride, v->size);
}

__global__ void __strided_memcpy(ok_float * x, size_t stride_x,
        const ok_float * y, size_t stride_y, size_t size)
{
        uint i, tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (i = tid; i < size; i += gridDim.x * blockDim.x)
        x[i * stride_x] = y[i * stride_y];
}


/*
 * VECTOR methods
 * ==============
 */
inline int __vector_exists(vector * v)
{
        if (v == OK_NULL) {
                printf("Error: cannot write to uninitialized vector pointer\n");
                return 0;
        } else {
                return 1;
        }
}

void vector_alloc(vector * v, size_t n)
{
        if (!__vector_exists(v))
                return;
        v->size=n;
        v->stride=1;
        ok_alloc_gpu(v->data, n * sizeof(ok_float));
}

void vector_calloc(vector * v, size_t n)
{
        vector_alloc(v, n);
        __vector_set_all(v, ok_float(0));
}

void vector_free(vector * v)
{
        if (v != OK_NULL)
                if (v->data != OK_NULL) ok_free_gpu(v->data);
        v->size = (size_t) 0;
        v->stride = (size_t) 0;
}

void vector_set_all(vector * v, ok_float x)
{
        __vector_set_all(v, x);
}

void vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n)
{
        if (!__vector_exists(v_out))
                return;
        v_out->size=n;
        v_out->stride=v_in->stride;
        v_out->data=v_in->data + offset * v_in->stride;
}

vector vector_subvector_gen(vector * v_in, size_t offset, size_t n)
{
        return (vector){
                .size = n,
                .stride = v_in->stride,
                .data = v_in->data + offset * v_in->stride
        };
}

void vector_view_array(vector * v, ok_float * base, size_t n)
{
          if (!__vector_exists(v))
                return;
          v->size=n;
          v->stride=1;
          v->data=base;
}


void vector_memcpy_vv(vector * v1, const vector * v2)
{
        uint grid_dim;
        if ( v1->stride == 1 && v2->stride == 1) {
                ok_memcpy_gpu(v1->data, v2->data, v1->size * sizeof(ok_float));
        } else {
                grid_dim = calc_grid_dim(v1->size);
                __strided_memcpy<<<grid_dim, kBlockSize>>>(v1->data, v1->stride,
                        v2->data, v2->stride, v1->size);
        }
}

void vector_memcpy_va(vector * v, const ok_float *y, size_t stride_y)
{
        uint i;
        if (v->stride == 1 && stride_y == 1)
                ok_memcpy_gpu(v->data, y, v->size * sizeof(ok_float));
        else
                for (i = 0; i < v->size; ++i)
        ok_memcpy_gpu(v->data + i * v->stride, y + i * stride_y,
        sizeof(ok_float));
}

void vector_memcpy_av(ok_float *x, const vector *v, size_t stride_x)
{
        uint i;
        if (v->stride == 1 && stride_x == 1)
                ok_memcpy_gpu(x, v->data, v->size * sizeof(ok_float));
        else
                for (i = 0; i < v->size; ++i)
        ok_memcpy_gpu(x + i * stride_x, v->data + i * v->stride,
        sizeof(ok_float));
}

void vector_print(const vector * v)
{
        uint i;
        ok_float v_host[v->size];
        vector_memcpy_av(v_host, v, 1);
        for (i = 0; i < v->size; ++i)
                printf("%e ", v_host[i]);
        printf("\n");
}

void vector_scale(vector * v, ok_float x)
{
        __thrust_vector_scale(v, x);
        CUDA_CHECK_ERR;
}

void vector_add(vector * v1, const vector * v2)
{
        __thrust_vector_add(v1, v2);
        CUDA_CHECK_ERR;
}

void vector_sub(vector * v1, const vector * v2)
{
        __thrust_vector_sub(v1, v2);
        CUDA_CHECK_ERR;
}

void vector_mul(vector * v1, const vector * v2)
{
        __thrust_vector_mul(v1, v2);
        CUDA_CHECK_ERR;
}

void vector_div(vector * v1, const vector * v2)
{
        __thrust_vector_div(v1, v2);
        CUDA_CHECK_ERR;
}

void vector_add_constant(vector * v, const ok_float x)
{
        __thrust_vector_add_constant(v, x);
        CUDA_CHECK_ERR;
        }

void vector_abs(vector * v)
{
        __thrust_vector_abs(v);
        CUDA_CHECK_ERR;
}

void vector_recip(vector * v)
{
        __thrust_vector_recip(v);
        CUDA_CHECK_ERR;
}

void vector_sqrt(vector * v)
{
        __thrust_vector_sqrt(v);
        CUDA_CHECK_ERR;
}

void vector_pow(vector * v, const ok_float x)
{
        __thrust_vector_pow(v, x);
        CUDA_CHECK_ERR;
}


/*
 * MATRIX CUDA helper methods
 * ==========================
 */

/* row major setter */
__global__ void __matrix_set_r(ok_float * data, ok_float x, size_t stride,
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

/* column major setter */
__global__ void __matrix_set_c(ok_float * data, ok_float x, size_t stride,
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


void __matrix_set_all(matrix * A, ok_float x)
{
        uint grid_dimx = calc_grid_dim(A->size1);
        uint grid_dimy = calc_grid_dim(A->size2);
        dim3 grid_dim(grid_dimx, grid_dimy, 1u);
        dim3 block_dim(kBlockSize, kBlockSize - 1, 1u);

        if (A->order == CblasRowMajor)
                __matrix_set_r<<<grid_dim, block_dim>>>(A->data, x, A->ld,
                        A->size1, A->size2);
        else
                __matrix_set_c<<<grid_dim, block_dim>>>(A->data, x, A->ld,
                        A->size1, A->size2);
        CUDA_CHECK_ERR;
}

__global__ void __matrix_add_constant_diag(ok_float * data, ok_float x,
        size_t stride)
{
        uint i = blockIdx.x * blockDim.x + threadIdx.x;
        data[i * stride + i] += x;
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


/*
 * MATRIX methods
 * ==============
 */
inline int __matrix_exists(matrix * A)
{
        if (A == OK_NULL) {
                printf("Error: cannot write to uninitialized matrix pointer\n");
                return 0;
        } else {
                return 1;
        }
}

void matrix_alloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
        A->size1 = m;
        A->size2 = n;
        ok_alloc_gpu(A->data, m * n * sizeof(ok_float));
        A->ld = (ord == CblasRowMajor) ? n : m;
        A->order = ord;
}

void matrix_calloc(matrix * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
        if (!__matrix_exists(A))
                return;
        matrix_alloc(A, m, n, ord);
        cudaMemset(A->data, 0, m * n * sizeof(ok_float));
        CUDA_CHECK_ERR;
}

void matrix_free(matrix * A)
{
        if (A == OK_NULL || A->data != OK_NULL)
                return;
        ok_free_gpu(A->data);
        A->size1 = (size_t) 0;
        A->size2 = (size_t) 0;
        A->ld = (size_t) 0;

}

void matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1,
        size_t n2)
{
        __matrix_exists(A_sub);
        A_sub->size1 = n1;
        A_sub->size2 = n2;
        A_sub->ld = A->ld;
        A_sub->data = (A->order == CblasRowMajor) ?
                      A->data + (i * A->ld) + j : A->data + i + (j * A->ld);
        A_sub->order = A->order;
}

matrix matrix_submatrix_gen(matrix * A, size_t i, size_t j, size_t n1,
        size_t n2)
{
        return (matrix){
                .size1 = n1,
                .size2 = n2,
                .ld = A->ld,
                .data = (A->order == CblasRowMajor) ?
                        A->data + (i * A->ld) + j : A->data + i + (j * A->ld),
                .order = A->order
        };
}

void matrix_row(vector * row, matrix * A, size_t i)
{
        if (!__vector_exists(row))
                return;
        row->size = A->size2;
        row->stride = (A->order == CblasRowMajor) ? 1 : A->ld;
        row->data = (A->order == CblasRowMajor) ?
                    A->data + (i * A->ld) : A->data + i;
}

void matrix_column(vector * col, matrix *A, size_t j)
{
        if (!__vector_exists(col))
                return;
        col->size = A->size1;
        col->stride = (A->order == CblasRowMajor) ? A->ld : 1;
        col->data = (A->order == CblasRowMajor) ?
                    A->data + j : A->data + (j * A->ld);
}

void matrix_diagonal(vector * diag, matrix *A)
{
        if (!__vector_exists(diag))
                return;
        diag->data = A->data;
        diag->stride = A->ld + 1;
        diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

void matrix_cast_vector(vector * v, matrix * A)
{
        v->size = A->size1 * A->size2;
        v->stride = 1;
        v->data = A->data;
}

void matrix_view_array(matrix * A, const ok_float *base, size_t n1,
                       size_t n2, enum CBLAS_ORDER ord)
{
        if (!__matrix_exists(A))
                return;
        A->size1 = n1;
        A->size2 = n2;
        A->data = (ok_float *) base;
        A->ld = (ord == CblasRowMajor) ? n2 : n1;
        A->order = ord;
}

void matrix_set_all(matrix * A, ok_float x)
{
        __matrix_set_all(A, x);
}


void matrix_memcpy_mm(matrix * A, const matrix * B)
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
                              A->size1 * A->size2 * sizeof(ok_float));
        } else if (A->order == CblasRowMajor) {
                /* A row major, B column major */
                grid_dim = calc_grid_dim(A->size1);
                for (i = 0; i < A->size1; ++i)
                        __strided_memcpy<<<grid_dim, kBlockSize>>>(
                                A->data + i * A->size2, 1,
                                B->data + i, A->ld, A->size2);
        } else {
                /* A column major, B row major */
                grid_dim = calc_grid_dim(A->size2);
                for (j= 0; j < A->size2; ++j)
                        __strided_memcpy<<<grid_dim, kBlockSize>>>(
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
void matrix_memcpy_ma(matrix * A, const ok_float * B,
	const enum CBLAS_ORDER ord)
{
        uint i, j, grid_dim;
        ok_float * row, * col;

        if (A->order == ord) {
        	if (ord == CblasRowMajor)
	        	for (i = 0; i < A->size1; ++i)
		                ok_memcpy_gpu(A->data + i * A->ld,
		                	B + i * A->size2,
		                	A->size2 * sizeof(ok_float));
		else
	        	for (j = 0; j < A->size2; ++j)
		                ok_memcpy_gpu(A->data + j * A->ld,
		                	B + j * A->size1,
		                	A->size1 * sizeof(ok_float));
        } else if (ord == CblasColMajor) {
                ok_alloc_gpu(col, A->size1 * sizeof(ok_float));
                grid_dim = calc_grid_dim(A->size1);
                for (j = 0; j < A->size2; ++j) {
                        ok_memcpy_gpu(col, B + j * A->size1,
                                      A->size1 * sizeof(ok_float));
                        __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + j,
                                A->ld, col, 1, A->size1);
                }
                ok_free_gpu(col);
        } else {
                ok_alloc_gpu(row, A->size2 * sizeof(ok_float));
                grid_dim = calc_grid_dim(A->size2);
                for (i = 0; i < A->size1; ++i) {
                        ok_memcpy_gpu(row, B + i * A->size2,
                                      A->size2 * sizeof(ok_float));
                        __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + i,
                                A->ld, row, 1, A->size2);
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
void matrix_memcpy_am(ok_float * A, const matrix * B,
	const enum CBLAS_ORDER ord)
{
        uint i, j, grid_dim;
        ok_float * row, * col;
        if (ord == B->order) {
        	if (ord == CblasRowMajor)
	        	for (i = 0; i < B->size1; ++i)
		                ok_memcpy_gpu(A + i * B->size2,
		                	B->data + i * B->ld,
		                	B->size2 * sizeof(ok_float));
		else
	        	for (j = 0; j < B->size2; ++j)
		                ok_memcpy_gpu(A + j * B->size1,
		                	B->data + j * B->ld,
		                	B->size1 * sizeof(ok_float));
        } else if (ord == CblasRowMajor) {
                ok_alloc_gpu(row, B->size2 * sizeof(ok_float));
                grid_dim = calc_grid_dim(B->size2);
                for (i = 0; i < B->size1; ++i) {
                        __strided_memcpy<<<grid_dim, kBlockSize>>>(row, 1,
                                B->data + i, B->ld, B->size2);
                        ok_memcpy_gpu(A + i * B->size2, row,
                                      B->size2 * sizeof(ok_float));
                }
                ok_free_gpu(row);
        } else {
                ok_alloc_gpu(col, B->size1 * sizeof(ok_float));
                grid_dim = calc_grid_dim(B->size1);
                for (j = 0; j < B->size2; ++j) {
                        __strided_memcpy<<<grid_dim, kBlockSize>>>(col, 1,
                                B->data + j, B->ld, B->size1);
                        ok_memcpy_gpu(A + j * B->size1, col,
                                      B->size1 * sizeof(ok_float));
                }
                ok_free_gpu(col);
        }
        CUDA_CHECK_ERR;
}

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

void matrix_scale(matrix * A, ok_float x)
{
        size_t i;
        vector row_col = (vector){0, 0, OK_NULL};
        if (A->order == CblasRowMajor)
                for(i = 0; i < A->size1; ++i) {
                        matrix_row(&row_col, A, i);
                        vector_scale(&row_col, x);
                }
        else
                for(i = 0; i < A->size2; ++i) {
                        matrix_column(&row_col, A, i);
                        vector_scale(&row_col, x);
                }
}

void matrix_scale_left(matrix * A, const vector * v)
{
        size_t i;
        vector col = (vector){0, 0, OK_NULL};
        for(i = 0; i < A->size2; ++i) {
                matrix_column(&col, A, i);
                vector_mul(&col, v);
        }
}

void matrix_scale_right(matrix * A, const vector * v)
{
        size_t i;
        vector row = (vector){0, 0, OK_NULL};
        for(i = 0; i < A->size1; ++i) {
                matrix_row(&row, A, i);
                vector_mul(&row, v);
        }
}

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

int __matrix_order_compat(const matrix * A, const matrix * B, const char * nm_A,
        const char * nm_B, const char * nm_routine)
{
        if (A->order == B->order)
            return 1;

        printf("OPTKIT ERROR (%s) matrices %s and %s must have same layout.\n",
                 nm_routine, nm_A, nm_B);
        return 0;
}

/*
 * BLAS routines
 * =============
 */
ok_status blas_make_handle(void ** handle)
{
        cublasStatus_t status;
        cublasHandle_t * hdl;
        hdl = (cublasHandle_t *) malloc(sizeof(cublasHandle_t));
        status = cublasCreate(hdl);
        if (status != CUBLAS_STATUS_SUCCESS) {
                printf("CUBLAS initialization failed\n");
                ok_free(hdl);
                *handle = OK_NULL;
                return OPTKIT_ERROR_CUBLAS;
        } else {
                *handle = (void *) hdl;
                return OPTKIT_SUCCESS;
        }
}

ok_status blas_destroy_handle(void * handle)
{
        cublasDestroy(*(cublasHandle_t *) handle);
        CUDA_CHECK_ERR;
        ok_free(handle);
        return OPTKIT_SUCCESS;
}


/* BLAS LEVEL 1 */
void blas_axpy(void * linalg_handle, ok_float alpha, const vector *x, vector *y)
{
        if (!linalg_handle)
                return;
        CUBLAS(axpy)(*(cublasHandle_t *) linalg_handle, (int) x->size, &alpha,
                x->data, (int) x->stride, y->data, (int) y->stride);
        CUDA_CHECK_ERR;
}

ok_float blas_nrm2(void * linalg_handle, const vector *x)
{
        ok_float result = kZero;
        if (!linalg_handle)
                return get_cuda_nan();
        CUBLAS(nrm2)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
                (int) x->stride, &result);
        CUDA_CHECK_ERR;
        return result;
}

void blas_scal(void * linalg_handle, const ok_float alpha, vector *x)
{
        if (!linalg_handle)
                return;
        CUBLAS(scal)(*(cublasHandle_t *) linalg_handle, (int) x->size, &alpha,
                x->data, (int) x->stride);
        CUDA_CHECK_ERR;
}

ok_float blas_asum(void * linalg_handle, const vector * x)
{
        ok_float result = kZero;
        if (!linalg_handle)
                return get_cuda_nan();
        CUBLAS(asum)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
                (int) x->stride, &result);
        CUDA_CHECK_ERR;
        return result;
}

ok_float blas_dot(void * linalg_handle, const vector * x, const vector * y)
{
        ok_float result = kZero;
        if (!linalg_handle)
                return get_cuda_nan();
        CUBLAS(dot)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
                (int) x->stride, y->data, (int) y->stride, &result);
        CUDA_CHECK_ERR;
        return result;
}

void blas_dot_inplace(void * linalg_handle, const vector * x, const vector * y,
        ok_float * deviceptr_result)
{
        CUBLAS(dot)(*(cublasHandle_t *) linalg_handle, (int) x->size, x->data,
                (int) x->stride, y->data, (int) y->stride, deviceptr_result);
        CUDA_CHECK_ERR;
}

/* BLAS LEVEL 2 */

void blas_gemv(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, const matrix *A, const vector *x, ok_float beta,
	vector *y)
{
        cublasOperation_t tA;
        int s1, s2;

        if (A->order == CblasColMajor)
                tA = (transA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
        else
                tA = (transA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

        s1 = (A->order == CblasRowMajor) ? (int) A->size2 : (int) A->size1;
        s2 = (A->order == CblasRowMajor) ? (int) A->size1 : (int) A->size2;

        if (!linalg_handle)
                return;

        CUBLAS(gemv)(*(cublasHandle_t *) linalg_handle, tA, s1, s2, &alpha,
                A->data, (int) A->ld, x->data, (int) x->stride, &beta, y->data,
                (int) y->stride);
        CUDA_CHECK_ERR;
}

void blas_trsv(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, const matrix *A,
	vector *x)
{
        cublasOperation_t tA;
        cublasDiagType_t di;
        cublasFillMode_t ul;

        if (A->order == CblasColMajor) {
                tA = (transA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
                ul = (uplo == CblasLower) ?
                     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        } else {
                tA = (transA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
                ul = (uplo == CblasLower) ?
                     CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        }

        di = Diag == CblasNonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;

        if (!linalg_handle)
                return;

        CUBLAS(trsv)(*(cublasHandle_t *) linalg_handle, ul, tA, di,
                (int) A->size1, A->data, (int) A->ld, x->data, (int) x->stride);
        CUDA_CHECK_ERR;
}


void blas_sbmv(void * linalg_handle, enum CBLAS_ORDER order,
	enum CBLAS_UPLO uplo, const size_t num_superdiag, const ok_float alpha,
	const vector * vecA, const vector * x, const ok_float beta, vector * y)
{
        cublasFillMode_t ul;
        if (order == CblasRowMajor)
                ul = (uplo == CblasLower) ?
                     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        else
                ul = (uplo == CblasLower) ?
                     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

        CUBLAS(sbmv)(*(cublasHandle_t *) linalg_handle, ul,
               (int) y->size, (int) num_superdiag, &alpha,
               vecA->data, (int) (num_superdiag + 1),
               x->data, (int) x->stride, &beta, y->data, (int) y->stride);
}

void blas_diagmv(void * linalg_handle, const ok_float alpha,
        const vector * vecA, const vector * x, const ok_float beta, vector * y)
{
        blas_sbmv(linalg_handle, CblasColMajor, CblasLower, 0, alpha, vecA, x,
                beta, y);
}

/* BLAS LEVEL 3 */
void blas_syrk(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, ok_float alpha, const matrix * A,
	ok_float beta, matrix * C)
{

        cublasOperation_t tA;
        cublasFillMode_t ul;
        const int k = (transA == CblasNoTrans) ?
                      (int) A->size2 : (int) A->size1;

        if (A->order == CblasColMajor) {
                tA = (transA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
                ul = (uplo == CblasLower) ?
                     CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        } else {
                tA = (transA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
                ul = (uplo == CblasLower) ?
                     CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        }


        if (!linalg_handle)
                return;

        if ( !__matrix_order_compat(A, C, "A", "C", "blas_syrk") )
                return;


        CUBLAS(syrk)(*(cublasHandle_t *) linalg_handle, ul, tA, (int) C->size2,
                k, &alpha, A->data, (int) A->ld, &beta, C->data, (int) C->ld);

        CUDA_CHECK_ERR;
}

void blas_gemm(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
        enum CBLAS_TRANSPOSE transB, ok_float alpha, const matrix * A,
        const matrix * B, ok_float beta, matrix * C)
{
        cublasOperation_t tA, tB;
        int s1, s2;

        const int k = (transA == CblasNoTrans) ?
                      (int) A->size2 : (int) A->size1;

        s1 = (A->order == CblasRowMajor) ? (int) C->size2 : (int) C->size1;
        s2 = (A->order == CblasRowMajor) ? (int) C->size1 : (int) C->size2;
        if (A->order == CblasColMajor) {
                tA = transA == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
                tB = transB == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
        } else {
                tA = transB == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
                tB = transA == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
        }

        if (!linalg_handle)
                return;

        if (!__matrix_order_compat(A, B, "A", "B", "blas_gemm") ||
                !__matrix_order_compat(A, C, "A", "C", "blas_gemm"))
                return;

        CUBLAS(gemm)(*(cublasHandle_t *) linalg_handle, tA, tB, s1, s2, k,
                &alpha, A->data, (int) A->ld, B->data, (int) B->ld, &beta,
                C->data, (int) C->ld);

        CUDA_CHECK_ERR;
}


void blas_trsm(void * linalg_handle, enum CBLAS_SIDE Side, enum CBLAS_UPLO uplo,
        enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, ok_float alpha,
        const matrix *A, matrix *B)
{
        printf("Method `blas_trsm()` not implemented for GPU\n");
}

/*
 * LINEAR ALGEBRA routines
 * =======================
 */

/* cholesky decomposition of a single block */
__global__ void __block_chol(ok_float * A, uint iter, uint ld,
	enum CBLAS_ORDER ord)
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

        ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
                (ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;

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
                get(A, global_row, global_col, ld) =
                get(L, row, col, kTileLD);
}

__global__ void __block_trsv(ok_float * A, uint iter, uint n, uint ld,
	enum CBLAS_ORDER ord)
{
        uint tile_idx, row, global_row, global_col, i, j;
        const uint kTileLD = kTileSize + 1u;
        __shared__ ok_float L[kTileLD * kTileSize];
        __shared__ ok_float A12[kTileLD * kTileSize];

        tile_idx = blockIdx.x;
        row = threadIdx.x;
        global_col = iter * kTileSize;
        global_row = iter * kTileSize + row;

        ok_float& (* get)(ok_float * A, uint i, uint j, uint stride) =
                (ord == CblasRowMajor) ? __matrix_get_r : __matrix_get_c;




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

        err = cublasGetStream(*(cublasHandle_t *) linalg_handle, &stm);
        num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;

        for (i = 0; i < num_tiles; ++i) {
                if (err != CUBLAS_STATUS_SUCCESS)
                        break;

                /* L11 = chol(A11) */
                uint block_dim_1d = kTileSize < A->size1 - i * kTileSize ? \
                                    kTileSize : A->size1 - i * kTileSize;
	        dim3 block_dim(block_dim_1d, block_dim_1d);

                __block_chol<<<1, block_dim, 0, stm>>>(A->data, i,
                	(uint) A->ld, A->order);
                CUDA_CHECK_ERR;

                if (i == num_tiles - 1u)
                        break;

                /* L21 = A21 * L21^-T */
                grid_dim = num_tiles - i - 1u;
                matrix L21 = matrix_submatrix_gen(A, (i + 1) * kTileSize,
                	i * kTileSize, A->size1 - (i + 1) * kTileSize,
                	kTileSize);

                __block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data, i,
                        (uint) A->size1, (uint) A->ld, A->order);
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