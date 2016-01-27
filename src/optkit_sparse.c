#include "optkit_sparse.h"


#ifdef __cplusplus
extern "C" {
#endif

void __csr2csc(size_t m, size_t n, size_t nnz, ok_float * csr_val, ok_int * row_ptr, ok_int * col_ind,
      ok_float * csc_val, ok_int * row_ind, ok_int * col_ptr){

  ok_int i, j, k, l;
  memset(col_ptr, 0, (n + 1) * sizeof(ok_int));

  for (i = 0; i < (ok_int) nnz; i++)
    col_ptr[col_ind[i] + 1]++;

  for (i = 0; i < (ok_int) n; i++)
    col_ptr[i + 1] += col_ptr[i];

  for (i = 0; i < (ok_int) m; i++) {
    for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
      k = col_ind[j];
      l = col_ptr[k]++;
      row_ind[l] = i;
      csc_val[l] = csr_val[j];
    }
  }

  for (i = (ok_int) n; i > 0; i--)
    col_ptr[i] = col_ptr[i - 1];

  col_ptr[0] = 0;
}

void __transpose_inplace(sp_matrix * A){
  if (A->rowmajor == CblasRowMajor)
    __csr2csc(A->size1, A->size2, A->nnz, 
      A->val, A->ptr, A->ind,
      A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen);
  else
    __csr2csc(A->size2, A->size1, 
      A->nnz, A->val, A->ptr, A->ind,
      A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen);
}


void sp_make_handle(void * sparse_handle){
  sparse_handle = OK_NULL;
}

void sp_destroy_handle(void * sparse_handle){
  return;
}

void sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order) {
  /* Stored forward and adjoint operators */

  A->size1 = m;
  A->size2 = n;
  A->nnz = nnz;
  A->ptrlen = (order == CblasColMajor) ? n + 1 : m + 1;
  A->val = (ok_float *) malloc(2 * nnz * sizeof(ok_float));
  A->ind = (ok_int *) malloc(2 * nnz * sizeof(ok_int));
  A->ptr = (ok_int *) malloc((2 + m + n) * sizeof(ok_int));
}

void sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order){

  sp_matrix_alloc(A, m, n, nnz, order);
  memset(A->val, 0, nnz * sizeof(ok_float));
  memset(A->val, 0, nnz * sizeof(ok_int));
  memset(A->val, 0, A->ptrlen * sizeof(ok_int));

}

void sp_matrix_free(sp_matrix * A) { 
  ok_free(A->val);
  ok_free(A->ind);
  ok_free(A->ptr);
}


void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr){

  memcpy(A->val, val, A->nnz * sizeof(ok_float));
  memcpy(A->ind, ind, A->nnz * sizeof(ok_int));
  memcpy(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
  __transpose_inplace(A);
}

void sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr,
  const sp_matrix * A){

  memcpy(val, A->val, A->nnz * sizeof(ok_float));
  memcpy(ind, A->ind, A->nnz * sizeof(ok_int));
  memcpy(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
}

void sp_matrix_print(const sp_matrix * A){
  uint i;

  if (A->rowmajor == CblasRowMajor)
    printf("sparse CSR matrix:\n");
  else
    printf("sparse CSC matrix:\n");

  printf("dims: %u, %u\n", (uint) A->size1, (uint) A->size2);
  printf("# nonzeros: %u\n", (uint) A->nnz);
  printf("\n\tpointers:\n");
  for(i = 0; i < A->ptrlen; ++i)
    printf("%i ", A->ptr[i]);
  printf("\n");
  printf("\n\tindices:\n");
  for(i = 0; i < A->nnz; ++i)
    printf("%i ", A->ind[i]);
  printf("\n");  
  printf("\n\tvalues:\n");
  for(i = 0; i < A->nnz; ++i)
    printf("%e ", A->val[i]);
  printf("\n");
}

void sp_blas_gemv(void * sparse_handle, 
  CBLAS_TRANSPOSE_t transA, ok_float alpha, sp_matrix * A, 
  vector * x, ok_float beta, vector * y){

  /* Always perform forward (non-transpose) operations */
  /* cusparse uses csr, so:
    csr, forward op -> forward
    csr, adjoint op -> adjoint
    csc, forward op -> adjoint
    csc, adjoint op -> forward */

  size_t ptrlen, i;
  ok_int j;
  ok_float * val, tmp;
  ok_int * ind, * ptr;

  if ((A->rowmajor == CblasRowMajor) != (transA == CblasTrans)){
    ptrlen = A->ptrlen;
    ptr = A->ptr;
    ind = A->ind;
    val = A->val;
  } else {
    ptrlen = A->size1 + A->size2 + 2 - A->ptrlen;
    ptr = A->ptr + A->ptrlen;
    ind = A->ind + A->nnz;
    val = A->val + A->nnz;
  }

  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (i = 0; i < ptrlen - 1; ++i) {
    tmp = kZero;
    for (j = ptr[i]; j < ptr[i + 1]; ++j) {
      tmp += val[j] * x->data[ind[j]];
    }
    y->data[i] = alpha * tmp + beta * y->data[i];
  }

  // CusparseCheckError(err);
  // return err;
}


#ifdef __cplusplus
}
#endif