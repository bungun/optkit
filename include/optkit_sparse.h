#ifndef OPTKIT_SPARSE_H_GUARD
#define OPTKIT_SPARSE_H_GUARD

#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif


/* MATRIX defition and methods */
typedef struct sp_matrix {
  size_t size1, size2, nnz, ptrlen;
  ok_float *data;
  ok_int *ind, *ptr;
  CBLAS_ORDER_t rowmajor;
} sp_matrix;

/* struct for cusparse handle and cusparse matrix description*/
typedef struct ok_sparse_handle{
  void * hdl;
  void * descr;
} ok_sparse_handle;

void sp_make_handle(void ** sparse_handle);
void sp_destroy_handle(void * sparse_handle);

void sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order);
// void sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, 
//   size_t nnz, CBLAS_ORDER_t ord);
void sp_matrix_free(sp_matrix * A);
void sp_matrix_tranpose(void * sparse_handle, sp_matrix * A);
void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr);
void sp_matrix_memcpy_am(sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr);

// void sp_matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1, size_t n2);
// void sp_matrix_row(vector * row, matrix * A, size_t i);
// void sp_matrix_column(vector * col, matrix * A, size_t j);
// void sp_matrix_diagonal(vector * diag, matrix * A);
// void sp_matrix_cast_vector(vector * v, matrix * A);
// void sp_matrix_view_array(matrix * A, const ok_float * base, size_t n1, size_t n2, CBLAS_ORDER_t ord);
// void sp_matrix_set_all(matrix * A, ok_float x);
// void sp_matrix_memcpy_mm(matrix * A, const matrix *B);
// void sp_matrix_print(matrix * A);
// void sp_matrix_scale(matrix * A, ok_float x);
// void sp_matrix_abs(matrix * A);

void sp_blas_gemv(void * sparse_handle, 
  CBLAS_TRANSPOSE_t transA, ok_float alpha, sp_matrix * A, 
  vector * x, ok_float beta, vector * y);


#ifdef __cplusplus
}
#endif

#endif  /* OPTKIT_SPARSE_H_GUARD */

