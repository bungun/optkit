#ifndef OPTKIT_SPARSE_H_GUARD
#define OPTKIT_SPARSE_H_GUARD

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif


/* MATRIX defition and methods */
typedef struct sp_matrix {
  size_t size1, size2, nnz, ptrlen;
  ok_float * val;
  ok_int * ind, * ptr;
  CBLAS_ORDER_t rowmajor;
} sp_matrix;

/* struct for cusparse handle and cusparse matrix description*/
typedef struct ok_sparse_handle{
  void * hdl;
  void * descr;
} ok_sparse_handle;

void sp_make_handle(void * sparse_handle);
void sp_destroy_handle(void * sparse_handle);

void sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order);
void sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order);
void sp_matrix_free(sp_matrix * A);
void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr);
void sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr, 
  const sp_matrix * A);
void sp_matrix_print(const sp_matrix * A);

void sp_blas_gemv(void * sparse_handle, 
  CBLAS_TRANSPOSE_t transA, ok_float alpha, sp_matrix * A, 
  vector * x, ok_float beta, vector * y);


#ifdef __cplusplus
}
#endif

#endif  /* OPTKIT_SPARSE_H_GUARD */

