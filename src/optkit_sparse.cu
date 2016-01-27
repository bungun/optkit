#include "optkit_sparse.h"
#include "optkit_defs_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif



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
  __float_set<<<grid_dim, kBlockSize>>>(v->data, x, stride, size);
}

void 
__int_set_all(ok_int * data, ok_float val, size_t stride, size_t size){
  uint grid_dim = calc_grid_dim(size);
  __float_set<<<grid_dim, kBlockSize>>>(arr, val, stride, size);
}

void __transpose_inplace(void * sparse_handle, sp_matrix * A){
  cusparseStatus_t err;
  ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;

  if (A->rowmajor == CblasRowMajor)
    err = CUSPARSE(csr2csc)(
      *(cusparseHandle_t *) sp_hdl->hdl)), 
      (int) A->size1, (int) A->size2, (int) A->nnz, A->val, A->ptr, A->ind,
      A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen, 
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  else
    err = CUSPARSE(csr2csc)(
      *(cusparseHandle_t *) sp_hdl->hdl, 
      (int) A->size2, (int) A->size1, (int) A->nnz, A->val, A->ptr, A->ind,
      A->val + A->nnz, A->ind + A->nnz, A->ptr + A->ptrlen, 
      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

  // CusparseCheckError(err);
  // return err;
}


void sp_make_handle(void * sparse_handle){
  ok_sparse_handle * ok_hdl = (ok_sparse_handle * ) malloc(
    sizeof(ok_sparse_handle) );
  cusparseHandle_t * hdl = (cusparseHandle_t *) malloc( 
    sizeof(cusparseHandle_t) );
  cusparseMatDescr_t * descr = (cusparseMatDescr_t *) malloc( 
    sizeof(cusparseMatDescr_t) );
  cusparseCreate(&hdl);
  CUDA_CHECK_ERR;
  cusparseCreateMatDescr(&descr);
  CUDA_CHECK_ERR;
  sparse_hdl->hdl = (void *) hdl;
  sparse_hdl->descr = (void *) descr;
}

void sp_destroy_handle(void * sparse_handle){
  ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;
  cusparseDestroy(*(cusparseHandle_t *) (
    (void *) sp_hdl->hdl));
  CUDA_CHECK_ERR;
  cusparseDestroyMatDescr(*(cusparseMatDescr_t *) (
    (void *) sp_hdl->hdl));
  CUDA_CHECK_ERR;
  ok_free(sp_hdl->descr);
  ok_free(sp_hdl->hdl);
  ok_free(sparse_handle);
}

void sp_matrix_alloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order) {
  /* Stored forward and adjoint operators */
  A->size1 = m;
  A->size2 = n;
  A->nnz = nnz;
  A->ptrlen = (order == CblasColMajor) ? n + 1 : m + 1;
  ok_alloc_gpu(A->val, 2 * nnz * sizeof(ok_float));
  ok_alloc_gpu(A->ind, 2 * nnz * sizeof(ok_int));
  ok_alloc_gpu(A->ptr, 2 * A->ptrlen * sizeof(ok_int));
}

void sp_matrix_calloc(sp_matrix * A, size_t m, size_t n, 
  size_t nnz, CBLAS_ORDER_t order){
  sp_matrix_alloc(A, m, n, nnz, order); 
  __float_set_all(A->val, (ok_float) 0, 1, nnz); 
  __int_set_all(A->ind, (ok_int) 0, 1, nnz); 
  __int_set_all(A->ptr, (ok_int) 0, 1, ptrlen); 
}

void sp_matrix_free(sp_matrix * A) {
  ok_free_gpu(A->val);
  ok_free_gpu(A->ind);
  ok_free_gpu(A->ptr);
}

void sp_matrix_memcpy_ma(void * sparse_handle, sp_matrix * A, 
  const ok_float * val, const ok_int * ind, const ok_int * ptr){

  ok_memcpy_gpu(A->val, val, A->nnz * sizeof(ok_float));
  ok_memcpy_gpu(A->len, ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(A->ptr, ptr, A->ptrlen * sizeof(ok_int));
  __transpose_inplace(sparse_handle, A);
}

void sp_matrix_memcpy_am(ok_float * val, ok_int * ind, ok_int * ptr, 
  const sp_matrix * A){

  ok_memcpy_gpu(val, A->val, A->nnz * sizeof(ok_float));
  ok_memcpy_gpu(len, A->ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(ptr, A-> ptr, A->ptrlen * sizeof(ok_int));
}

void sp_matrix_print(const sp_matrix * A){
  uint i;
  ok_float val_host[A->nnz];
  ok_int ind_host[A->nnz];
  ok_int ptr_host[A->ptrlen];

  ok_memcpy_gpu(val_host, A->val, A->nnz * sizeof(ok_float));
  ok_memcpy_gpu(ind_host, A->ind, A->nnz * sizeof(ok_int));
  ok_memcpy_gpu(ptr_host, A->ptr, A->ptrlen * sizeof(ok_int));

  if (A->rowmajor == CblasRowMajor)
    printf("sparse CSR matrix:\n");
  else
    printf("sparse CSC matrix:\n");
  
  printf("dims: %u, %u\n", (uint) A->size1, (uint) A->size2);
  printf("# nonzeros: %u\n", (uint) A->nnz);
  printf("\n\tpointers:\n");
  for(i = 0; i < A->ptrlen; ++i)
    printf("%i ", ptr_host[i]);
  printf("\n");
  printf("\n\tindices:\n");
  for(i = 0; i < A->nnz; ++i)
    printf("%i ", ind_host[i]);
  printf("\n");  
  printf("\n\tvalues:\n");
  for(i = 0; i < A->nnz; ++i)
    printf("%e ", val_host[i]);
  printf("\n");
}


void sp_blas_gemv(void * sparse_handle, 
  CBLAS_TRANSPOSE_t transA, ok_float alpha, sp_matrix * A, 
  vector * x, ok_float beta, vector * y){

  cusparseStatus_t err;
  ok_sparse_handle * sp_hdl = (ok_sparse_handle *) sparse_handle;

  /* Always perform forward (non-transpose) operations */
  /* cusparse uses csr, so:
    csr, forward op -> forward
    csr, adjoint op -> adjoint
    csc, forward op -> adjoint
    csc, adjoint op -> forward */

  if ((A->rowmajor == CblasRowMajor) != (transA == CblasTrans))
    /* Use forward operator stored in A */
    err = CUSPARSE(csrmv)(
      *(cusparseHandle_t *) sp_hdl->hdl, 
      CUSPARSE_OPERATION_NON_TRANSPOSE, 
      (int) A->size1, (int) A->size2, (int) A->nnz, &alpha, 
      *(cusparseMatDescr_t *) sp_hdl->descr,        
      A->val, A->ptr, A->ind, x->data, &beta, y->data);
  else
    /* Use adjoint operator stored in A */
    err = CUSPARSE(csrmv)(
      *(cusparseHandle_t *) sp_hdl->hdl, 
      CUSPARSE_OPERATION_NON_TRANSPOSE, 
      (int) A->size2, (int) A->size1, (int) A->nnz, &alpha, 
      *(cusparseMatDescr_t *) sp_hdl->descr,        
      A->val + A->nnz, A->ptr + A->ptrlen, A->ind + A->nnz, 
      x->data, &beta, y->data);
  
  // CusparseCheckError(err);
  // return err;
}


#ifdef __cplusplus
}
#endif