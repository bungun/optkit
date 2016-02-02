#include "optkit_sparse.h"


#ifdef __cplusplus
extern "C" {
#endif

void 
sparselib_version(int * maj, int * min, int * change, int * status){
    * maj = OPTKIT_VERSION_MAJOR;
    * min = OPTKIT_VERSION_MINOR;
    * change = OPTKIT_VERSION_CHANGE;
    * status = (int) OPTKIT_VERSION_STATUS;
}

void 
__csr2csc(size_t m, size_t n, size_t nnz, 
    ok_float * csr_val, ok_int * row_ptr, ok_int * col_ind,
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

void 
__transpose_inplace(sp_matrix * A, SPARSE_TRANSPOSE_DIRECTION_t dir){
  if (dir == Forward2Adjoint)
    if (A->order == CblasRowMajor)
      __csr2csc(A->size1, A->size2, A->nnz, 
        A->val, A->ptr, A->ind,
        A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen);
    else
      __csr2csc(A->size2, A->size1, A->nnz, 
        A->val, A->ptr, A->ind,
        A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen);
  else
    if (A->order == CblasRowMajor)
      __csr2csc(A->size2, A->size1, A->nnz, 
        A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz,
        A->val, A->ind, A->ptr); 
    else
      __csr2csc(A->size1, A->size2, A->nnz, 
        A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz,
        A->val, A->ind, A->ptr);
}


void 
sp_make_handle(void ** sparse_handle){
  * sparse_handle = OK_NULL;
}

void 
sp_destroy_handle(void * sparse_handle){
  return;
}

void 
sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order) {
  /* Stored forward and adjoint operators */

  A->size1 = m;
  A->size2 = n;
  A->nnz = nnz;
  A->ptrlen = (order == CblasColMajor) ? n + 1 : m + 1;
  A->val = (ok_float *) malloc(2 * nnz * sizeof(ok_float));
  A->ind = (ok_int *) malloc(2 * nnz * sizeof(ok_int));
  A->ptr = (ok_int *) malloc((2 + m + n) * sizeof(ok_int));
  A->order = order;
}

void 
sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order){
  sp_matrix_alloc(A, m, n, nnz, order);
  memset(A->val, 0, 2 * nnz * sizeof(ok_float));
  memset(A->ind, 0, 2 * nnz * sizeof(ok_int));
  memset(A->ptr, 0, (2 + m + n) * sizeof(ok_int));
}

void 
sp_matrix_free(sp_matrix * A) { 
  ok_free(A->val);
  ok_free(A->ind);
  ok_free(A->ptr);
}

void 
sp_matrix_memcpy_mm(sp_matrix * A, const sp_matrix * B){
  memcpy(A->val, B->val, 2 * A->nnz * sizeof(ok_float));
  memcpy(A->ind, B->ind, 2 * A->nnz * sizeof(ok_int));
  memcpy(A->ptr, B->ptr, (A->size1 + A->size2 + 2) * sizeof(ok_int));  
}

void 
sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr){

  memcpy(A->val, val, A->nnz * sizeof(ok_float));
  memcpy(A->ind, ind, A->nnz * sizeof(ok_int));
  memcpy(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
  __transpose_inplace(A, Forward2Adjoint);
}

void 
sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr,
  const sp_matrix * A){

  memcpy(val, A->val, A->nnz * sizeof(ok_float));
  memcpy(ind, A->ind, A->nnz * sizeof(ok_int));
  memcpy(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
}

void 
sp_matrix_memcpy_vals_mm(sp_matrix * A, const sp_matrix * B){
  memcpy(A->val, B->val, 2 * A->nnz * sizeof(ok_float));
}

void 
sp_matrix_memcpy_vals_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val){
  memcpy(A->val, val, A->nnz * sizeof(ok_float));
  __transpose_inplace(A, Forward2Adjoint);
}

void 
sp_matrix_memcpy_vals_am(ok_float * val, const sp_matrix * A){
  memcpy(val, A->val, A->nnz * sizeof(ok_float));
}

void 
sp_matrix_abs(sp_matrix * A){
  size_t i;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (i = 0; i < 2 * A->nnz; ++i) {
    A->val[i] = MATH(fabs)(A->val[i]);
  }
}

void 
sp_matrix_pow(sp_matrix * A, const ok_float x){
  size_t i;
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (i = 0; i < 2 * A->nnz; ++i) {
    A->val[i] = MATH(pow)(A->val[i], x);
  }
}

void 
sp_matrix_scale(sp_matrix * A, const ok_float alpha){
  CBLAS(scal)( (int) (2 * A->nnz), alpha, A->val, 1);
}

void 
__sp_matrix_scale_diag(sp_matrix * A, const vector * v, CBLAS_SIDE_t side){
  size_t i, offset, offsetnz, stop;
  SPARSE_TRANSPOSE_DIRECTION_t dir;

  if (side == CblasLeft){
    offsetnz = (A->order == CblasRowMajor) ? 0 : A->nnz;
    offset = (A->order == CblasRowMajor) ? 0 : A->ptrlen;
    stop = (A->order == CblasRowMajor) ? A->ptrlen - 1 : 1 + A->size1 + A->size2;
    dir = (A->order == CblasRowMajor) ? Forward2Adjoint : Adjoint2Forward;
  } else {
    offsetnz = (A->order == CblasRowMajor) ? A->nnz : 0;
    offset = (A->order == CblasRowMajor) ? A->ptrlen : 0;
    stop = (A->order == CblasRowMajor) ? 1 + A->size1 + A->size2 : A->ptrlen - 1;
    dir = (A->order == CblasRowMajor) ? Adjoint2Forward : Forward2Adjoint;
  }

  for (i = offset; i < stop; ++i) { 
    if (A->ptr[i + 1] == A->ptr[i]) continue;
    CBLAS(scal)( (int) (A->ptr[i + 1] - A->ptr[i]), v->data[i - offset], 
      A->val + A->ptr[i] + offsetnz, 1);
  }
  __transpose_inplace(A, dir);
}



void 
sp_matrix_scale_left(void * sparse_handle,
  sp_matrix * A, const vector * v){

  if (A->size1 != v->size){
    printf("ERROR (optkit.sparse):\n \
      Incompatible dimensions for A = diag(v) * A\n \
      A: %i x %i, v: %i\n", (int) A->size1, (int) A->size2, (int) v->size);
    return;
  }
  __sp_matrix_scale_diag(A, v, CblasLeft);
}

void 
sp_matrix_scale_right(void * sparse_handle,
  sp_matrix * A, const vector * v){

  if (A->size2 != v->size){
    printf("ERROR (optkit.sparse):\n \
      Incompatible dimensions for A = A * diag(v)\n \
      A: %i x %i, v: %i\n", (int) A->size1, (int) A->size2, (int) v->size);
    return;
  }
  __sp_matrix_scale_diag(A, v, CblasRight);
}


void 
sp_matrix_print(const sp_matrix * A){
  size_t i;
  ok_int j, ptr1, ptr2;

  if (A->order == CblasRowMajor)
    printf("sparse CSR matrix:\n");
  else
    printf("sparse CSC matrix:\n");

  printf("dims: %u, %u\n", (uint) A->size1, (uint) A->size2);
  printf("# nonzeros: %u\n", (uint) A->nnz);

  if (A->order == CblasRowMajor)
    for(i = 0; i < A->ptrlen - 1; ++ i){
      ptr1 = A->ptr[i];
      ptr2 = A->ptr[i + 1];
      for(j = ptr1; j < ptr2; ++j){
        printf("(%i, %i)\t%e\n", (int) i,  A->ind[j], A->val[j]);
      }
    }
  else
    for(i = 0; i < A->ptrlen - 1; ++ i){
      ptr1 = A->ptr[i];
      ptr2 = A->ptr[i + 1];
      for(j = ptr1; j < ptr2; ++j){
        printf("(%i, %i)\t%e\n", A->ind[j], (int) i, A->val[j]);
      }
    }
  printf("\n");
}


void 
sp_blas_gemv(void * sparse_handle, 
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

  if ((A->order == CblasRowMajor) != (transA == CblasTrans)){
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