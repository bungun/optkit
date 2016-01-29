#include "optkit_sparse.h"
#include "optkit_defs_gpu.h"
#include "optkit_thrust.hpp"
#include <cusparse.h>

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

/* struct for cusparse handle and cusparse matrix description*/
typedef struct ok_sparse_handle{
  cusparseHandle_t * hdl;
  cusparseMatDescr_t * descr;
} ok_sparse_handle;

/* helper methods for CUDA */
__global__ void 
__float_set(ok_float * data, ok_float val, size_t stride, size_t size) {
  uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
    data[i * stride] = val;
}

__global__ void 
__int_set(ok_int * data, ok_int val, size_t stride, size_t size) {
  uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
    data[i * stride] = val;
}

void 
__float_set_all(ok_float * data, ok_float val, size_t stride, size_t size){
  uint grid_dim = calc_grid_dim(size);
  __float_set<<<grid_dim, kBlockSize>>>(data, val, stride, size);
}

void 
__int_set_all(ok_int * data, ok_int val, size_t stride, size_t size){
  uint grid_dim = calc_grid_dim(size);
  __int_set<<<grid_dim, kBlockSize>>>(data, val, stride, size);
}

void 
__transpose_inplace(void * sparse_handle, sp_matrix * A,
  SPARSE_TRANSPOSE_DIRECTION_t dir){

  ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;

  if (dir == Forward2Adjoint) {
    if (A->rowmajor == CblasRowMajor) {
      CUSPARSE(csr2csc)( *(sp_hdl->hdl), 
        (int) A->size1, (int) A->size2, (int) A->nnz, 
        A->val, A->ptr, A->ind,
        A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    } else {
      CUSPARSE(csr2csc)( *(sp_hdl->hdl), 
        (int) A->size2, (int) A->size1, (int) A->nnz, 
        A->val, A->ptr, A->ind,
        A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    }
  } else {
    if (A->rowmajor == CblasRowMajor) {
      CUSPARSE(csr2csc)( *(sp_hdl->hdl), 
        (int) A->size1, (int) A->size2, (int) A->nnz, 
        A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz,
        A->val, A->ind, A->ptr, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    } else {
      CUSPARSE(csr2csc)( *(sp_hdl->hdl), 
        (int) A->size2, (int) A->size1, (int) A->nnz, 
        A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz,
        A->val, A->ind, A->ptr, 
        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
    }
  }


  CUDA_CHECK_ERR;
  // CusparseCheckError(err);
  // return err;
}


void 
sp_make_handle(void ** sparse_handle){
  ok_sparse_handle * ok_hdl = (ok_sparse_handle * ) malloc(
    sizeof(ok_sparse_handle) );
  ok_hdl->hdl = (cusparseHandle_t *) malloc( 
    sizeof(cusparseHandle_t) );
  ok_hdl->descr = (cusparseMatDescr_t *) malloc( 
    sizeof(cusparseMatDescr_t) );
  cusparseCreate(ok_hdl->hdl);
  CUDA_CHECK_ERR;
  cusparseCreateMatDescr(ok_hdl->descr);
  CUDA_CHECK_ERR;
  * sparse_handle = (void *) ok_hdl;
}

void 
sp_destroy_handle(void * sparse_handle){
  ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;
  cusparseDestroy(*(sp_hdl->hdl));
  CUDA_CHECK_ERR;
  cusparseDestroyMatDescr(*(sp_hdl->descr));
  CUDA_CHECK_ERR;
  ok_free(sp_hdl->descr);
  ok_free(sp_hdl->hdl);
  ok_free(sparse_handle);
}

void 
sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order) {
  /* Stored forward and adjoint operators */
  A->size1 = m;
  A->size2 = n;
  A->nnz = nnz;
  A->ptrlen = (order == CblasColMajor) ? n + 1 : m + 1;
  ok_alloc_gpu(A->val, 2 * nnz * sizeof(ok_float));
  ok_alloc_gpu(A->ind, 2 * nnz * sizeof(ok_int));
  ok_alloc_gpu(A->ptr, (2 + m + n) * sizeof(ok_int));
  CUDA_CHECK_ERR;

}

void 
sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order){
  sp_matrix_alloc(A, m, n, nnz, order); 
  __float_set_all(A->val, (ok_float) 0, 1, 2 * nnz); 
  __int_set_all(A->ind, (ok_int) 0, 1, 2 * nnz); 
  __int_set_all(A->ptr, (ok_int) 0, 1, 2 + A->size1 + A->size2); 
  CUDA_CHECK_ERR;
}

void 
sp_matrix_free(sp_matrix * A) {
  ok_free_gpu(A->val);
  ok_free_gpu(A->ind);
  ok_free_gpu(A->ptr);
  CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_mm(sp_matrix * A, const sp_matrix * B){
  ok_memcpy_gpu(A->val, B->val, 2 * A->nnz * sizeof(ok_float));
  ok_memcpy_gpu(A->ind, B->ind, 2 * A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(A->ptr, B->ptr, (2 + A->size1 + A->size2) * sizeof(ok_int)); 
  CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr){

  ok_memcpy_gpu(A->val, val, A->nnz * sizeof(ok_float));
  ok_memcpy_gpu(A->ind, ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
  CUDA_CHECK_ERR;
  __transpose_inplace(sparse_handle, A, Forward2Adjoint);
}

void sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr, 
  const sp_matrix * A){

  ok_memcpy_gpu(val, A->val, A->nnz * sizeof(ok_float));
  ok_memcpy_gpu(ind, A->ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
  CUDA_CHECK_ERR;

}

void sp_matrix_memcpy_vals_mm(sp_matrix * A, const sp_matrix * B){
  ok_memcpy_gpu(A->val, B->val, 2 * A->nnz * sizeof(ok_float));
  CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_vals_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val){

  ok_memcpy_gpu(A->val, val, A->nnz * sizeof(ok_float));
  CUDA_CHECK_ERR;
  __transpose_inplace(sparse_handle, A, Forward2Adjoint);
}

void sp_matrix_memcpy_vals_am(ok_float * val, const sp_matrix * A){
  ok_memcpy_gpu(val, A->val, A->nnz * sizeof(ok_float));
  CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_pattern_mm(sp_matrix * A, const sp_matrix * B){
  ok_memcpy_gpu(A->ind, B->ind, 2 * A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(A->ptr, B->ptr, (2 + A->size1 + A->size2) * sizeof(ok_int)); 
  CUDA_CHECK_ERR;
}

void sp_matrix_memcpy_pattern_ma(void * sparse_handle, sp_matrix * A, 
  const ok_int * ind, const ok_int * ptr){

  ok_memcpy_gpu(A->ind, ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
  CUDA_CHECK_ERR;
  __transpose_inplace(sparse_handle, A, Forward2Adjoint);
}

void sp_matrix_memcpy_pattern_am(ok_int * ind, ok_int * ptr, 
  const sp_matrix * A){
  ok_memcpy_gpu(ind, A->ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
  CUDA_CHECK_ERR;
}


void sp_matrix_abs(sp_matrix * A){
  vector vals = (vector){2 * A->nnz, 1, A->val};
  __thrust_vector_abs(&vals);
  CUDA_CHECK_ERR;
}

void sp_matrix_scale(sp_matrix * A, const ok_float alpha){
  vector vals = (vector){2 * A->nnz, 1, A->val};
  __thrust_vector_scale(&vals, alpha);
  CUDA_CHECK_ERR;
}

void sp_matrix_scale_left(void * sparse_handle, 
  sp_matrix * A, const vector * v){
  uint i;
  vector Asub = (vector){0, 1, OK_NULL};
  ok_float v_host[v->size];
  vector_memcpy_av(v_host, v, 1);
  CUDA_CHECK_ERR;

  if (A->size1 != v->size){
    printf("ERROR (optkit.sparse):\n \
      Incompatible dimensions for A = diag(v) * A\n \
      A: %i x %i, v: %i\n", (int) A->size1, (int) A->size2, (int) v->size);
    return;
  }
  /*
  // ok_float val[1];
  */

  if (A->rowmajor == CblasRowMajor){
    for (i = 0; i < A->ptrlen - 1; ++i) {
      Asub.size = (size_t) (A->ptr[i + 1] - A->ptr[i]);
      Asub.data = A->val + A->ptr[i];
      __thrust_vector_scale(&Asub, v_host[i]);
      /*
      // ok_memcpy_gpu(val, v->data + i, sizeof(ok_float));
      // vector_scale(Asub, val);
      */
    }
    __transpose_inplace(sparse_handle, A, Forward2Adjoint);
  } else {
    for (i = A->ptrlen; i < A->size1 + A->size2 + 1; ++i) {
      Asub.size = (size_t) (A->ptr[i + 1] - A->ptr[i]);
      Asub.data = A->val + A->nnz + A->ptr[i];
      __thrust_vector_scale(&Asub, v_host[i - A->ptrlen]);
      /*
      // ok_memcpy_gpu(val, v->data + i, sizeof(ok_float));
      // vector_scale(Asub, val);
      */
    }
    __transpose_inplace(sparse_handle, A, Adjoint2Forward);
  }
  CUDA_CHECK_ERR;
}

void sp_matrix_scale_right(void * sparse_handle,
  sp_matrix * A, const vector * v){
  size_t i;
  vector Asub = (vector){0, 1, OK_NULL};
  ok_float v_host[v->size];
  vector_memcpy_av(v_host, v, 1);
  CUDA_CHECK_ERR;

  if (A->size2 != v->size){
    printf("ERROR (optkit.sparse):\n \
      Incompatible dimensions for A = A * diag(v)\n \
      A: %i x %i, v: %i\n", (int) A->size1, (int) A->size2, (int) v->size);
    return;
  }
  /*
  // ok_float val[1];
  */

  if (A->rowmajor == CblasRowMajor){
    for (i = A->ptrlen; i < A->size1 + A->size2 + 1; ++i) {
      Asub.size = (size_t) (A->ptr[i + 1] - A->ptr[i]);
      Asub.data = A->val + A->nnz + A->ptr[i];
      __thrust_vector_scale(&Asub, v_host[i - A->ptrlen]);
      /*
      // ok_memcpy_gpu(val, v->data + i, sizeof(ok_float));
      // __thrust_vector_scale(Asub, val);
      */
    }
    __transpose_inplace(sparse_handle, A, Adjoint2Forward);
  } else {
    for (i = 0; i < A->ptrlen - 1; ++i) {
      Asub.size = (size_t) (A->ptr[i + 1] - A->ptr[i]);
      Asub.data = A->val + A->ptr[i];
      __thrust_vector_scale(&Asub, v_host[i]);
      /*
      // ok_memcpy_gpu(val, v->data + i, sizeof(ok_float));
      // __thrust_vector_scale(Asub, val);
      */
    }
    __transpose_inplace(sparse_handle, A, Forward2Adjoint);
  }
  CUDA_CHECK_ERR;
}

void sp_matrix_print(const sp_matrix * A){
  size_t i;
  ok_int ptr_idx;
  ok_float val_host[A->nnz];
  ok_int ind_host[A->nnz];
  ok_int ptr_host[A->ptrlen];

  ok_memcpy_gpu(val_host, A->val, A->nnz * sizeof(ok_float));
  ok_memcpy_gpu(ind_host, A->ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(ptr_host, A->ptr, A->ptrlen * sizeof(ok_int));
  CUDA_CHECK_ERR;

  if (A->rowmajor == CblasRowMajor)
    printf("sparse CSR matrix:\n");
  else
    printf("sparse CSC matrix:\n");
  
  printf("dims: %u, %u\n", (uint) A->size1, (uint) A->size2);
  printf("# nonzeros: %u\n", (uint) A->nnz);

  ptr_idx = 0;
  if (A->rowmajor == CblasRowMajor)
    for(i = 0; i < A->nnz; ++i){
      while (ptr_host[ptr_idx + 1] - 1 <= i) ++ptr_idx;
      printf("(%i, %i)\t%e\n", ptr_idx, ind_host[i], val_host[i]);
    }
  else
    for(i = (uint) A->nnz; i < 2 * A->nnz; ++i){
      while (ptr_host[(uint) ptr_idx + A->ptrlen + 1] - 1 <= i) ++ptr_idx;
      printf("(%i, %i)\t%e\n", ind_host[i], ptr_idx, val_host[i]);
    }
  printf("\n");
  CUDA_CHECK_ERR;
}


void sp_blas_gemv(void * sparse_handle, 
  CBLAS_TRANSPOSE_t transA, ok_float alpha, sp_matrix * A, 
  vector * x, ok_float beta, vector * y){

  ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;

  /* Always perform forward (non-transpose) operations */
  /* cusparse uses csr, so:
    csr, forward op -> forward
    csr, adjoint op -> adjoint
    csc, forward op -> adjoint
    csc, adjoint op -> forward */

  if ((A->rowmajor == CblasRowMajor) != (transA == CblasTrans)){
    /* Use forward operator stored in A */
    CUSPARSE(csrmv)(
      *(sp_hdl->hdl), 
      CUSPARSE_OPERATION_NON_TRANSPOSE, 
      (int) A->size1, (int) A->size2, (int) A->nnz, &alpha, 
      *(sp_hdl->descr),        
      A->val, A->ptr, A->ind, x->data, &beta, y->data);
  } else {
    /* Use adjoint operator stored in A */
    CUSPARSE(csrmv)(
      *(sp_hdl->hdl), 
      CUSPARSE_OPERATION_NON_TRANSPOSE, 
      (int) A->size2, (int) A->size1, (int) A->nnz, &alpha, 
      *(sp_hdl->descr),        
      A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz, 
      x->data, &beta, y->data);
  }
  
  // CusparseCheckError(err);
  // return err;
}


#ifdef __cplusplus
}
#endif