#include "optkit_sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

void sp_make_handle(void ** sparse_handle){
  ok_sparse_handle * ok_hdl = (ok_sparse_handle *) malloc(
    sizeof(ok_sparse_handle *) )
  ok_hdl->hdl = OK_NULL;
  ok_hdl->descr = OK_NULL;
  * sparse_handle = (void *) ok_hdl;
}

void sp_destroy_handle(void * sparse_handle){
  ok_free(sparse_handle);
}

void sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order) {
  /* Stored forward and adjoint operators */
  A->m = m;
  A->n = n;
  A->nnz = nnz;
  A->ptrlen = (order == CblasColMajor) ? n : m;
  A->val = (ok_float *) malloc(2 * A->nnz * sizeof(ok_float));
  A->ind = (ok_float *) malloc(2 * A->nnz * sizeof(ok_int));
  A->ptr = (ok_float *) malloc(2 * A->ptrlen * sizeof(ok_int));
}

// void sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord);

void sp_matrix_free(sp_matrix * A) {
  ok_free(A->val);
  ok_free(A->ind);
  ok_free(A->ptr);
}

void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr){

  memcpy(A->val, val, A->nnz * sizeof(ok_float));
  memcpy(A->len, ind, A->nnz * sizeof(ok_int));
  memcpy(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
  sp_matrix_tranpose(sparse_handle, S);
}

void sp_matrix_memcpy_am(sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr){

  memcpy(val, A->val, A->nnz * sizeof(ok_float));
  memcpy(len, A->ind, A->nnz * sizeof(ok_int));
  memcpy(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
}

void sp_matrix_tranpose(void * sparse_handle, sp_matrix * A){
  cusparseStatus_t err;

  // TODO: REPLACE THE CUSPARSE CALLS HERE

  if (A->rowmajor == CblasRowMajor)
    err = spblas_csr2csc(
      A->m, A->n, A->nnz, A->val, A->ptr, A->ind,
      St->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen);
  else
    err = spblas_csr2csc(
      A->m, A->n, A->nnz, A->val, A->ptr, A->ind,
      St->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen);
}

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
  vector * x, ok_float beta, vector * y){

  // TODO: REPLACE THE CUSPARSE CALLS HERE

  /* Always perform forward (non-transpose) operations */
  /* cusparse uses csr, so:
    csr, forward op -> forward
    csr, adjoint op -> adjoint
    csc, forward op -> adjoint
    csc, adjoint op -> forward */

  if ((A->rowmajor == CblasRowMajor) != (transA == CblasTrans))
    /* Use forward operator stored in A */
    err = spblas_gemv(
      A->m, A->n, A->nnz, &alpha, 
      A->val, A->ptr, A->ind, x->data, &beta, y->data);
  else
    /* Use adjoint operator stored in A */
    err = spblas_gemv(
      A->n, A->m, A->nnz, &alpha, 
      A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz, 
      x->data, &beta, y->data);
  
  // CusparseCheckError(err);
  // return err;
}


#ifdef __cplusplus
}
#endif