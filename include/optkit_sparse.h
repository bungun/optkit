#ifndef OPTKIT_SPARSE_H_
#define OPTKIT_SPARSE_H_

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

void sparselib_version(int * maj, int * min, int * change, int * status);

/* transpose data from forward->adjoint or adjoint->forward */
typedef enum SparseTransposeDirection {
  Forward2Adjoint,
  Adjoint2Forward
} SPARSE_TRANSPOSE_DIRECTION;


/* MATRIX defition and methods */
typedef struct sp_matrix {
        size_t size1, size2, nnz, ptrlen;
        ok_float * val;
        ok_int * ind, * ptr;
        CBLAS_ORDER order;
} sp_matrix;


/* memory management */
void sp_make_handle(void ** sparse_handle);
void sp_destroy_handle(void * sparse_handle);

void sp_matrix_alloc(sp_matrix * A, size_t m, size_t n,
        size_t nnz, CBLAS_ORDER order);
void sp_matrix_calloc(sp_matrix * A, size_t m, size_t n,
        size_t nnz, CBLAS_ORDER order);
void sp_matrix_free(sp_matrix * A);

/* copy, I/O */
void sp_matrix_memcpy_mm(sp_matrix * A, const sp_matrix * B);
void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A,
        const ok_float * val, const ok_int * ind, const ok_int * ptr);
void sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr,
        const sp_matrix * A);
void sp_matrix_memcpy_vals_mm(sp_matrix * A, const sp_matrix * B);
void sp_matrix_memcpy_vals_ma(void * sparse_handle, sp_matrix * A,
        const ok_float * val);
void sp_matrix_memcpy_vals_am(ok_float * val, const sp_matrix * A);

/* elementwise, row-wise & column-wise math */
void sp_matrix_abs(sp_matrix * A);
void sp_matrix_pow(sp_matrix * A, const ok_float x);
void sp_matrix_scale(sp_matrix * A, const ok_float alpha);
void sp_matrix_scale_left(void * sparse_handle, sp_matrix * A,
        const vector * v);
void sp_matrix_scale_right(void * sparse_handle, sp_matrix * A,
        const vector * v);

/* print */
void sp_matrix_print(const sp_matrix * A);

/* matrix multiplication */
void sp_blas_gemv(void * sparse_handle, CBLAS_TRANSPOSE transA, ok_float alpha,
        sp_matrix * A, vector * x, ok_float beta, vector * y);

#ifdef __cplusplus
}
#endif

#endif  /* OPTKIT_SPARSE_H_ */