#include "optkit_sparse.h"
#include "optkit_defs_gpu.h"
#include "optkit_thrust.hpp"
#include <cusparse.h>

#ifdef __cplusplus
extern "C" {
#endif

void sparselib_version(int * maj, int * min, int * change, int * status)
{
        * maj = OPTKIT_VERSION_MAJOR;
        * min = OPTKIT_VERSION_MINOR;
        * change = OPTKIT_VERSION_CHANGE;
        * status = (int) OPTKIT_VERSION_STATUS;
}

/* struct for cusparse handle and cusparse matrix description*/
typedef struct ok_sparse_handle{
        cusparseHandle_t * hdl;
        cusparseMatDescr_t * descr;
} ok_sparse_handle;

/* helper methods for CUDA */
__global__ void __float_set(ok_float * data, ok_float val, size_t stride,
        size_t size)
{
        uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
                data[i * stride] = val;
}

__global__ void __int_set(ok_int * data, ok_int val, size_t stride, size_t size)
{
        uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
                data[i * stride] = val;
}

void __float_set_all(ok_float * data, ok_float val, size_t stride, size_t size)
{
        uint grid_dim = calc_grid_dim(size);
        __float_set<<<grid_dim, kBlockSize>>>(data, val, stride, size);
}

void __int_set_all(ok_int * data, ok_int val, size_t stride, size_t size)
{
        uint grid_dim = calc_grid_dim(size);
        __int_set<<<grid_dim, kBlockSize>>>(data, val, stride, size);
}

void __transpose_inplace(void * sparse_handle, sp_matrix * A,
        SPARSE_TRANSPOSE_DIRECTION_t dir)
{
        ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;
        int size1, size2;
        ok_int * ptr, * ind, * ptr_T, * ind_T;
        ok_float * val, * val_T;

        if (dir == Forward2Adjoint) == (A->order == CblasRowMajor) {
                size1 = (int) A->size1;
                size2 = (int) A->size2;
                ptr = A->ptr;
                ind = A->ind;
                val = A->val;
                ptr_T = A->ptr + A->ptrlen;
                ind_T = A->ind + A->nnz;
                val_T = A->val + A->nnz;

        } else {
                size1 = (int) A->size2;
                size2 = (int) A->size1;
                ptr = A->ptr + A->ptrlen;
                ind = A->ind + A->nnz;
                val = A->val + A->nnz;
                ptr_T = A->ptr;
                ind_T = A->ind;
                val_T = A->val;
        }

        CUSPARSE(csr2csc)(*(sp_hdl->hdl), size1, size2, (int) A->nnz, val, ptr,
                ind, val_T, ind_T, ptr_T, CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO);

        CUDA_CHECK_ERR;
}


void sp_make_handle(void ** sparse_handle)
{
        ok_sparse_handle * ok_hdl;
        ok_hdl = malloc(sizeof(*ok_hdl));
        ok_hdl->hdl = malloc(sizeof(cusparseHandle_t));
        ok_hdl->descr = malloc(sizeof(cusparseMatDescr_t));
        cusparseCreate(ok_hdl->hdl);
        CUDA_CHECK_ERR;
        cusparseCreateMatDescr(ok_hdl->descr);
        CUDA_CHECK_ERR;
        * sparse_handle = (void *) ok_hdl;
}

void sp_destroy_handle(void * sparse_handle)
{
        ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;
        cusparseDestroy(*(sp_hdl->hdl));
        CUDA_CHECK_ERR;
        cusparseDestroyMatDescr(*(sp_hdl->descr));
        CUDA_CHECK_ERR;
        ok_free(sp_hdl->descr);
        ok_free(sp_hdl->hdl);
        ok_free(sparse_handle);
}

void sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, size_t nnz,
        CBLAS_ORDER_t order)
{
        /* Store forward and adjoint operators */
        A->size1 = m;
        A->size2 = n;
        A->nnz = nnz;
        A->ptrlen = (order == CblasColMajor) ? n + 1 : m + 1;
        ok_alloc_gpu(A->val, 2 * nnz * sizeof(ok_float));
        ok_alloc_gpu(A->ind, 2 * nnz * sizeof(ok_int));
        ok_alloc_gpu(A->ptr, (2 + m + n) * sizeof(ok_int));
        CUDA_CHECK_ERR;
        A->order = order;
}

void sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, size_t nnz,
        CBLAS_ORDER_t order)
{
        sp_matrix_alloc(A, m, n, nnz, order);
        __float_set_all(A->val, (ok_float) 0, 1, 2 * nnz);
        __int_set_all(A->ind, (ok_int) 0, 1, 2 * nnz);
        __int_set_all(A->ptr, (ok_int) 0, 1, 2 + A->size1 + A->size2);
        CUDA_CHECK_ERR;
}

void sp_matrix_free(sp_matrix * A)
{
        ok_free_gpu(A->val);
        ok_free_gpu(A->ind);
        ok_free_gpu(A->ptr);
        CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_mm(sp_matrix * A, const sp_matrix * B)
{
        ok_memcpy_gpu(A->val, B->val, 2 * A->nnz * sizeof(ok_float));
        ok_memcpy_gpu(A->ind, B->ind, 2 * A->nnz * sizeof(ok_int));
        ok_memcpy_gpu(A->ptr, B->ptr, (2 + A->size1 + A->size2) *
                sizeof(ok_int));
        CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A,
        const ok_float * val, const ok_int * ind, const ok_int * ptr)
{
        ok_memcpy_gpu(A->val, val, A->nnz * sizeof(ok_float));
        ok_memcpy_gpu(A->ind, ind, A->nnz * sizeof(ok_int));
        ok_memcpy_gpu(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
        CUDA_CHECK_ERR;
        __transpose_inplace(sparse_handle, A, Forward2Adjoint);
}

void sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr,
        const sp_matrix * A)
{
        ok_memcpy_gpu(val, A->val, A->nnz * sizeof(ok_float));
        ok_memcpy_gpu(ind, A->ind, A->nnz * sizeof(ok_int));
        ok_memcpy_gpu(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
        CUDA_CHECK_ERR;

}

void sp_matrix_memcpy_vals_mm(sp_matrix * A, const sp_matrix * B)
{
        ok_memcpy_gpu(A->val, B->val, 2 * A->nnz * sizeof(ok_float));
        CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_vals_ma(void * sparse_handle, sp_matrix * A,
        const ok_float * val)
{
        ok_memcpy_gpu(A->val, val, A->nnz * sizeof(ok_float));
        CUDA_CHECK_ERR;
        __transpose_inplace(sparse_handle, A, Forward2Adjoint);
}

void sp_matrix_memcpy_vals_am(ok_float * val, const sp_matrix * A)
{
        ok_memcpy_gpu(val, A->val, A->nnz * sizeof(ok_float));
        CUDA_CHECK_ERR;
}

void sp_matrix_abs(sp_matrix * A)
{
        vector vals = (vector){2 * A->nnz, 1, A->val};
        __thrust_vector_abs(&vals);
        CUDA_CHECK_ERR;
}

void sp_matrix_pow(sp_matrix * A, const ok_float x)
{
        vector vals = (vector){2 * A->nnz, 1, A->val};
        __thrust_vector_pow(&vals, x);
        CUDA_CHECK_ERR;
        }

void sp_matrix_scale(sp_matrix * A, const ok_float alpha)
{
        vector vals = (vector){2 * A->nnz, 1, A->val};
        __thrust_vector_scale(&vals, alpha);
        CUDA_CHECK_ERR;
}

/*
 * Always perform forward (non-transpose) scaling (contiguous blocks)
 * libcusparse uses csr, so for given (layout, scaling direction) pair:
 *
 *      csr, left scaling -> scale forward operator, transpose data F2A
 *      csr, right scaling -> scale adjoint operator, transpose data A2F
 *      csc, left scaling -> scale adjoint operator, transpose data A2F
 *      csc, right scaling -> scale forward operator, tranpose data F2A
 */
void __sp_matrix_scale_diag(void * sparse_handle, sp_matrix * A,
        const vector * v, CBLAS_SIDE_t side)
{
        size_t i, offset, offsetnz, stop;
        vector Asub = (vector){0, 1, OK_NULL};
        ok_float scal;
        ok_int ptr_host[2 + A->size1 + A->size2];
        SPARSE_TRANSPOSE_DIRECTION_t dir;

        ok_memcpy_gpu(ptr_host, A->ptr, (2 + A->size1 + A->size2)
                * sizeof(ok_int));

        if (side == CblasLeft) {
                offsetnz = (A->order == CblasRowMajor) ? 0 : A->nnz;
                offset = (A->order == CblasRowMajor) ? 0 : A->ptrlen;
                stop = (A->order == CblasRowMajor) ?
                       A->ptrlen - 1 : 1 + A->size1 + A->size2;
                dir = (A->order == CblasRowMajor) ?
                      Forward2Adjoint : Adjoint2Forward;
        } else {
                offsetnz = (A->order == CblasRowMajor) ? A->nnz : 0;
                offset = (A->order == CblasRowMajor) ? A->ptrlen : 0;
                stop = (A->order == CblasRowMajor) ?
                       1 + A->size1 + A->size2 : A->ptrlen - 1;
                dir = (A->order == CblasRowMajor) ?
                       Adjoint2Forward : Forward2Adjoint;
        }

        for (i = offset; i < stop; ++i) {
                if (ptr_host[i + 1] == ptr_host[i])
                        continue;
                ok_memcpy_gpu(&scal, v->data + (i - offset) * v->stride,
                        sizeof(ok_float));

                Asub.size = ptr_host[i + 1] - ptr_host[i];
                Asub.data = A->val + offsetnz + ptr_host[i];
                __thrust_vector_scale(&Asub, scal);
        }
        __transpose_inplace(sparse_handle, A, dir);
        CUDA_CHECK_ERR;
}

void sp_matrix_scale_left(void * sparse_handle, sp_matrix * A, const vector * v)
{
        if (A->size1 != v->size) {
                printf("ERROR (optkit.sparse):\n",
                        "Incompatible dimensions for A = diag(v) * A\n ",
                        "A: %i x %i, v: %i\n",
                        (int) A->size1, (int) A->size2, (int) v->size);
                return;
        }
        __sp_matrix_scale_diag(sparse_handle, A, v, CblasLeft);
}

void sp_matrix_scale_right(void * sparse_handle, sp_matrix * A,
        const vector * v)
{
        if (A->size2 != v->size) {
                printf("ERROR (optkit.sparse):\n ",
                        "Incompatible dimensions for A = A * diag(v)\n ",
                        "A: %i x %i, v: %i\n", (int) A->size1, (int) A->size2,
                        (int) v->size);
                return;
        }
        __sp_matrix_scale_diag(sparse_handle, A, v, CblasRight);
}


void sp_matrix_print(const sp_matrix * A)
{
        size_t i;
        ok_int j, ptr1, ptr2;
        ok_float val_host[A->nnz];
        ok_int ind_host[A->nnz];
        ok_int ptr_host[A->ptrlen];

        ok_memcpy_gpu(val_host, A->val, A->nnz * sizeof(ok_float));
        ok_memcpy_gpu(ind_host, A->ind, A->nnz * sizeof(ok_int));
        ok_memcpy_gpu(ptr_host, A->ptr, A->ptrlen * sizeof(ok_int));
        CUDA_CHECK_ERR;


        if (A->order == CblasRowMajor)
                printf("sparse CSR matrix:\n");
        else
                printf("sparse CSC matrix:\n");

        printf("dims: %u, %u\n", (uint) A->size1, (uint) A->size2);
        printf("# nonzeros: %u\n", (uint) A->nnz);

        if (A->order == CblasRowMajor)
                for(i = 0; i < A->ptrlen - 1; ++ i) {
                        ptr1 = ptr_host[i];
                        ptr2 = ptr_host[i + 1];
                        for(j = ptr1; j < ptr2; ++j)
                                printf("(%i, %i)\t%e\n", (int) i,  ind_host[j],
                                        val_host[j]);
                }
        else
                for(i = 0; i < A->ptrlen - 1; ++ i) {
                        ptr1 = ptr_host[i];
                        ptr2 = ptr_host[i + 1];
                        for(j = ptr1; j < ptr2; ++j)
                                printf("(%i, %i)\t%e\n", ind_host[j], (int) i,
                                        val_host[j]);
                }
        printf("\n");
}

/*
 * Always perform forward (non-transpose) operations
 * cusparse library uses csr layout, so for given (layout, op):
 *      csr, forward op -> apply stored forward operator
 *      csr, adjoint op -> apply stored adjoint operator
 *      csc, forward op -> apply stored adjoint operator
 *      csc, adjoint op -> apply stored forward operator
 */
void sp_blas_gemv(void * sparse_handle, CBLAS_TRANSPOSE_t transA,
        ok_float alpha, sp_matrix * A, vector * x, ok_float beta, vector * y)
{
        ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;

        if ((A->order == CblasRowMajor) != (transA == CblasTrans))
                /* Use forward operator stored in A */
                CUSPARSE(csrmv)( *(sp_hdl->hdl),
                        CUSPARSE_OPERATION_NON_TRANSPOSE, (int) A->size1,
                        (int) A->size2, (int) A->nnz, &alpha, *(sp_hdl->descr),
                        A->val, A->ptr, A->ind, x->data, &beta, y->data);
        else
                /* Use adjoint operator stored in A */
                CUSPARSE(csrmv)(*(sp_hdl->hdl),
                        CUSPARSE_OPERATION_NON_TRANSPOSE, (int) A->size2,
                        (int) A->size1, (int) A->nnz, &alpha, *(sp_hdl->descr),
                        A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz,
                        x->data, &beta, y->data);
}

#ifdef __cplusplus
}
#endif