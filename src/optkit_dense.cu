#include "optkit_dense.h"
#include "gsl_cblas.h"
#include "optkit_defs_gpu.h"


/* thrust:: methods */
strided_range_t
__make_strided_range(vector * v){
  return strided_range_t(
    thrust::device_pointer_cast(v->data),
    thrust::device_pointer_cast(v->data + v->stride * v->size), v->stride);
}

strided_range_t
__make_const_strided_range(const vector * v){
  return strided_range_t(
    thrust::device_pointer_cast(v->data),
    thrust::device_pointer_cast(v->data + v->stride * v->size), v->stride);
}

void
__transform_rr(strided_range_t r1, strided_range_t r2, binary_function_t f){
  thrust::transform(r1.begin(), r1.end(), r2.begin(), 
    r1.begin(), f);
}

void
__transform_rc(strided_range_t r, ok_float x, binary_function_t f){
  thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
    r.begin(), f);
}

void
__transform_cr(ok_float x, strided_range_t r, binary_function_t f){
  thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
    r.begin(), f);
}

void __thrust_vector_scale(vector * v, ok_float x) {
  strided_range_t r = __make_strided_range(v);
  transform_rc(r, x, thrust::multiplies<ok_float>());
}

void __thrust_vector_add(vector * v1, const vector * v2) {
  strided_range_t r1, r2;
  r1 = __make_strided_range(v1);
  r2 = __make_const_strided_range(v2);
  transform_rr(r1, r2, thrust::plus<ok_float>());
}

void __thrust_vector_sub(vector * v1, const vector * v2) {
  strided_range_t r1, r2;
  r1 = __make_strided_range(v1);
  r2 = __make_const_strided_range(v2);
  transform_rr(r1, r2, thrust::minus<ok_float>());
}

void __thrust_vector_mul(vector * v1, const vector * v2) {
  strided_range_t r1, r2;
  r1 = __make_strided_range(v1);
  r2 = __make_const_strided_range(v2);
  transform_rr(r1, r2, thrust::multiplies<ok_float>());
}

void __thrust_vector_div(vector * v1, const vector * v2) {
  strided_range_t r1, r2;
  r1 = __make_strided_range(v1);
  r2 = __make_const_strided_range(v2);
  transform_rr(r1, r2, thrust::divides<ok_float>());
}

void __thrust_vector_add_constant(vector * v, const ok_float x) {
  strided_range_t r = __make_strided_range(v);
  transform_rc(r, x, thrust::plus<ok_float>());
}

void __thrust_vector_pow(vector * v, const ok_float p) {
  r1 = __make_strided_range(v);
  struct pow_op{
    __device__ void  
    operator()(ok_float x){
      return pow(x, p)
    }
  };
  thrust::for_each(r1.begin(), v->size, pow_op());
}


#ifdef __cplusplus
extern "C" {
#endif

/* VECTOR helper methods for CUDA */
__global__ void 
__set_vector(ok_float * data, ok_float val, size_t stride, size_t size) {
  uint i, tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (i = tid; i < size; i += gridDim.x * blockDim.x)
    data[i * stride] = val;
}

void 
__set_vector_all(veector * v, ok_float x){
  uint grid_dim = calc_grid_dim(v->size);
  __set_vector<<<grid_dim, kBlockSize>>>(v->data, x, v->stride, v->size);
}

__global__ void 
__strided_memcpy(ok_float * x, size_t stride_x, 
  const ok_float *y, size_t stride_y, size_t size) {
  uint i, tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (i = tid; i < size; i += gridDim.x * blockDim.x)
    x[i * stride_x] = y[i * stride_y];
}


/* VECTOR methods */
inline int 
__vector_exists(vector * v) {
  if (v == OK_NULL){
    printf("Error: cannot write to uninitialized vector pointer\n");
    return 0;
  }
  else
    return 1;
}

void 
__vector_alloc(vector * v, size_t n) {
  if (!__vector_exists(v)) return;
  v->size=n;
  v->stride=1;
  ok_alloc_gpu(v->data, n * sizeof(T));
}

void 
__vector_calloc(vector * v, size_t n) {
  __vector_alloc(v, n);
  __set_vector_all(v, ok_float(0));
}

void 
__vector_free(vector * v) {
  if (v->data != OK_NULL) ok_free(v->data);
}


void 
__vector_set_all(vector * v, ok_float x) {
  __set_vector_all(v, x);
}

void 
__vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n) {
  if (!__vector_exists(v_out)) return;
  v_out->size=n;
  v_out->stride=v_in->stride;
  v_out->data=v_in->data + offset * v_in->stride;
}

void 
__vector_view_array(vector * v, ok_float * base, size_t n) {
  if (!__vector_exists(v)) return;
  v->size=n;
  v->stride=1;
  v->data=base;
}


void 
__vector_memcpy_vv(vector * v1, const vector * v2) {
  uint grid_dim;
  if ( v1->stride == 1 && v2->stride == 1) {
    cudaMemcpy(v1->data, v2->data, v1->size * sizeof(ok_float),
      cudaMemcpyDefault);
  } else {
    grid_dim = calc_grid_dim(v1->size, kBlockSize);
    __strided_memcpy<<<grid_dim, kBlockSize>>>(v1->data, v1->stride,
      v2->data, v2->stride, v1->size);
  }
}

void 
__vector_memcpy_va(vector * v, const ok_float *y) {
  uint i;
  if (v->stride == 1) {
    memcpy(v->data, y, v->size * sizeof(ok_float));
  } else {
    for (i = 0; i < v->size; ++i)
      cudaMemcpy(v->data + i, y + i, v->size * sizeof(ok_float),
       cudaMemcpyDefault);
  }
}

void 
__vector_memcpy_av(ok_float *x, const vector *v) {
  uint i;  
  if (v->stride ==1) {
    memcpy(x, v->data, v->size * sizeof(ok_float));
  } else {
    for (i = 0; i < v->size; ++i)
      cudaMemcpy(y + i, v->data + i, v->size * sizeof(ok_float),
       cudaMemcpyDefault);      
  }
}


void 
__vector_print(const vector * v) {
  uint i;
  ok_float * v_host = (ok_float *) malloc(v->size * sizeof(ok_float));
  vector_memcpy(v_host, v);
  for (i = 0; i < v->size; ++i)
    printf("%e ", v_host[i * v->stride];
  printf("\n");
  ok_free(v_host);
}

void 
__vector_scale(vector * v, ok_float x) {
  __thrust_vector_scale(v, x);
}

void 
__vector_add(vector * v1, const vector * v2) {
  __thrust_vector_add(v1, v2);
}

void 
__vector_sub(vector * v1, const vector * v2) {
  __thrust_vector_sub(v1, v2);
}

void 
__vector_mul(vector * v1, const vector * v2) {
  __thrust_vector_mul(v1, v2);
}

void 
__vector_div(vector * v1, const vector * v2) {
  __thrust_vector_div(v1, v2);
}

void 
__vector_add_constant(vector * v, const ok_float x) {
  __thrust_vector_add_constant(v1, x);
}

void 
__vector_pow(vector * v, const ok_float x) {
  __thrust_vector_pow(v, x);
}


/* MATRIX CUDA helper methods */
__global__ void 
__set_matrix(ok_float * data, ok_float x, size_t tda, 
  size_t size1, size_t size2, CBLAS_ORDER_t rowmajor){
  uint i, j;
  uint tid_row = blockIdx.x * blockDim.x + threadIdx.x;
  uint tid_col = blockIdx.y * blockDim.y + threadIdx.y;
  if (rowmajor == CblasRowMajor)
    for (i = tid_row; i < size1; i += gridDim.x * blockDim.x)
      for (j = tid_col; j < size2; j += gridDim.y * blockDim.y)
        data[i * tda + j] = val;
  else
    for (j = tid_col; j < size2; j += gridDim.y * blockDim.y)
      for (i = tid_row; i < size1; i += gridDim.x * blockDim.x)
        data[i + j * tda] = val;
}

void 
__set_matrix_all(matrix * A, ok_float x) {
  uint grid_dimx = calc_grid_dim(A->size1, kBlockSize);
  uint grid_dimy = calc_grid_dim(A->size2, kBlockSize);
  dim3 grid_dim(grid_dimx, grid_dimy, 1u);
  dim3 block_dim(kBlockSize, kBlockSize, 1u);
  __set_matrix<<<grid_dim, block_dim>>>(A->data, x, 
    A->size1, A->size2, A->rowmajor);
}

__global__ void
__matrix_add_constant_diag(ok_float * data, ok_float x, size_t tda){
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i * tda + i += val];
}


/* CUDA helper kernels */
__device__ inline ok_float& 
__get_matrix(ok_float * A, uint i, uint j, uint tda, uint rowmajor) {
  if (rowmajor) return &A[i + j * tda];
  else return &A[i * tda + j];
}




/* MATRIX methods */
inline int 
__matrix_exists(matrix * A) {
  if (A == OK_NULL){
    printf("Error: cannot write to uninitialized matrix pointer\n");
    return 0;
  }
  else
    return 1;
}


void 
__matrix_alloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  A->size1 = m;
  A->size2 = n;
  A->tda = (ord == CblasRowMajor) ? n : m;
  ok_alloc_gpu(A->data, m * n * sizeof(ok_float));
  A->rowmajor = ord;
}

void 
__matrix_calloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  __matrix_alloc(A, m, n, ord);
  __set_matrix_all(A, (ok_float) 0);
}

void 
__matrix_free(matrix * A) {
  if (A == OK_NULL || A->data != OK_NULL) return;
  ok_free_gpu(A->data);
}

void
__matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1, size_t n2){
  __matrix_exists(A_sub);
  A_sub->size1 = n1;
  A_sub->size2 = n2;
  A_sub->tda = A->tda;
  A_sub->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) + j : A->data + i + (j * A->tda);
  A_sub->rowmajor = A->rowmajor;
}


void 
__matrix_row(vector * row, matrix * A, size_t i) {
  if (!__vector_exists(row)) return;
  row->size = A->size2;
  row->stride = (A->rowmajor == CblasRowMajor) ? 1 : A->tda;
  row->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) : A->data + i;
}

void 
__matrix_column(vector * col, matrix *A, size_t j) {
  if (!__vector_exists(col)) return;
  col->size = A->size1;
  col->stride = (A->rowmajor == CblasRowMajor) ? A->tda : 1, 
  col->data = (A->rowmajor == CblasRowMajor) ? A->data + j : A->data + (j * A->tda);
}

void 
__matrix_diagonal(vector * diag, matrix *A) {
  if (!__vector_exists(diag)) return;
  diag->data = A->data;
  diag->stride = A->tda + 1;
  diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

void 
__matrix_view_array(matrix * A, const ok_float *base, size_t n1, 
  size_t n2, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  A->size1 = n1;
  A->size2 = n2;
  A->tda = (ord == CblasRowMajor) ? n2 : n1;
  A->data = (ok_float *) base;
  A->rowmajor = ord;
}


void 
__matrix_memcpy_mm(matrix * A, const matrix * B) {
  uint i, j, grid_dim;
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n");
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n");
  else{ 
    if (A->rowmajor == B->rowmajor)  
      memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(ok_float));
    else if (A->rowmajor == CblasRowMajor){
      /* A row major, B column major */
      grid_dim = calc_grid_dim(A->size1, kBlockSize);
      for (i = 0; i < A->size1; ++i)
        __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + i * A->size2, 
          1, B->data + i, A->tda, A->size2);
    } else {
      /* A column major, B row major */
      grid_dim = calc_grid_dim(A->size2, kBlockSize);
      for (j= 0; j < A->size2; ++j)
        __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + j * A->size1, 
          1, B->data + j, A->tda, A->size1);
    }
    CUDA_CHECK_ERR;
  }
}

void 
__matrix_memcpy_ma(matrix * A, const ok_float * B, 
  const CBLAS_ORDER_t rowmajor) {
  uint i, j, grid_dim;
  ok_float * row, col;

  if (rowmajor == A->rowmajor) {
    cudaMemcpy(A->data, B, A->size1 * A->size2 * sizeof(ok_float),
      cudaMemcpyDefault);
  } else if (rowmajor == CblasColMajor) {
    /* A row major, B column major */
    ok_alloc_gpu(col, A->size1);
    grid_dim = calc_grid_dim(A->size1, kBlockSize);
    for (j = 0; j < A->size2; ++j){
      cudaMemcpy(col, B + j * A->size1, A->size1, cudaMemcpyDefault);
      __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + j,
        A->tda, col, 1, A->size1);
    }
  } else {
    /* A column major, B row major */
    ok_alloc_gpu(row, A->size2);
    grid_dim = calc_grid_dim(A->size2, kBlockSize);
    for (i = 0; i < A->size1; ++j){
      cudaMemcpy(col, B + i * A->size1, A->size1, cudaMemcpyDefault);
      __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + i,
        A->tda, row, 1, A->size2);
    }
  }
  CUDA_CHECK_ERR;
}

void 
__matrix_memcpy_am(ok_float * A, const matrix * B, 
  const CBLAS_ORDER_t rowmajor) {
  uint i, j, grid_dim;
  ok_float * row, col;
  if (rowmajor == B->rowmajor) {
    cudaMemcpy(A, B->data, B->size1 * B->size2 * sizeof(ok_float),
      cudaMemcpyDefault);
  } else if (rowmajor == CblasRowMajor) {
    /* A row major, B column major */
    ok_alloc_gpu(row, B->size2);
    grid_dim = calc_grid_dim(B->size2, kBlockSize);
    for (i = 0; i < B->size1; ++i){
      __strided_memcpy<<<grid_dim, kBlockSize>>>(row, 1, 
        B->data + i, B->tda, B->size2);
      cudaMemcpy(A + i * B->size2, row, B->size2, cudaMemcpyDefault);
    }
    ok_free_gpu(row);
  } else {
    /* A column major, B row major */
    ok_alloc_gpu(col, B->size1);
    grid_dim = calc_grid_dim(B->size1, kBlockSize);
    for (j = 0; j < B->size2; ++j){
      __strided_memcpy<<<grid_dim, kBlockSize>>>(col, 1,
        B->data + j, B->tda, B->size1);
      cudaMemcpy(A + j * B->size1, col, B->size1, cudaMemcpyDefault);
    }
  }
  CUDA_CHECK_ERR;
}

void 
__matrix_print(const matrix * A) {
  ok_float * A_;
  A_ = (ok_float *) malloc(A->size1 * A->size2 * sizeof(ok_float));
  __matrix_memcpy_am(A_, A);
  for (uint i = 0; i < A->size1; ++i) {
    for (uint j = 0; j < A->size2; ++j)
      if (A->rowmajor == CblasRowMajor)
        printf("%e ", A_[i * A->tda + j]);
      else
        printf("%e ", A_[i + j * A->tda]);
    printf("\n");
  }
  printf("\n");
  ok_free(A_);
}


void 
__matrix_scale(matrix *A, ok_float x) {
  vector * row_col = OK_NULL;
  row_col = &(vector){0,0,OK_NULL};
  size_t i;
  if (A->rowmajor == CblasRowMajor)
    for(i = 0; i < A->size1; ++i){
      __matrix_row(row_col, A, i);
      __vector_scale(row_col, x);
    }
  else{
    for(i = 0; i < A->size2; ++i){
      __matrix_column(row_col, A, i);
      __vector_scale(row_col, x);
    }
  }
}


/* BLAS routines */

inline int 
__blas_check_handle(void * handle){
  if (handle == OK_NULL) return 0;
  else return 1; 
}

void 
__blas_make_handle(void * handle){
  cublasHandle_t * hdl = (cublasHandle_t *) handle;
  cublasCreate(hdl);
}

void 
__blas_destroy_handle(void * handle){
  cublasDestroy(*(cublasHandle_t *) handle);
  handle = OK_NULL;
}


/* BLAS LEVEL 1 */
void 
__blas_axpy(void * blas_handle, ok_float alpha, 
                 const vector *x, vector *y) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CUBLAS(axpy)(*(cublasHandle_t *) linalg_handle,
   (int) x->size, &alpha, x->data, (int) x->stride, 
   y->data, (int) y->stride);
}

ok_float 
__blas_nrm2(void * blas_handle, const vector *x) {
  ok_float * result;
  if ( !__blas_check_handle(linalg_handle) ) return (ok_float) nan;
  CUBLAS(nrm2)(*(cublasHandle_t *) linalg_handle, 
    (int) x->size, x->data, (int) x->stride, result);
  return *result;
}

void 
__blas_scal(void * blas_handle, const ok_float alpha, vector *x) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CUBLAS(scal)(*(cublasHandle_t *) linalg_handle, 
    (int) x->size, &alpha, x->data, (int) x->stride);
}

ok_float 
__blas_asum(void * linalg_handle, const vector * x) {
  ok_float result;
  if ( !__blas_check_handle(linalg_handle) ) return (ok_float) nan;
  CUBLAS(asum)(*(cublasHandle_t *) linalg_handle, 
    (int) x->size, x->data, (int) x->stride, result);
  return *result;
}

ok_float 
__blas_dot(void * linalg_handle, 
                    const vector * x, const vector * y) {
  
  ok_float * result;
  if ( !__blas_check_handle(linalg_handle) ) return (ok_float) nan;
  CUBLAS(dot)(*(cublasHandle_t *) linalg_handle,
    (int) x->size, x->data, (int) x->stride, 
    y->data, (int) y->stride, result);
  return *result;
}

/* BLAS LEVEL 2 */

void 
__blas_gemv(void * linalg_handle, CBLAS_TRANSPOSE_t TransA, 
                ok_float alpha, const matrix *A, 
               const vector *x, ok_float beta, vector *y){

  cublasHandle_t hdl;
  cublasOperation_t tA;
  int s1, s2;

  tA = (Trans==CblasTrans) != (A->rowmajor==CblasRowMajor) ? \
      CUBLAS_OP_T : CUBLAS_OP_N; 
  s1 = (A->rowmajor==CblasRowMajor) ? C->size2 : C->size1;
  s2 = (A->rowmajor==CblasRowMajor) ? C->size1 : C->size2;

  if ( !__blas_check_handle(linalg_handle) ) return;
  CUBLAS(gemv)(*(cublasHandle_t *) linalg_handle, tA, s1, s2, 
    &alpha, A->data, (int) A->tda, x->data, (int) x->stride, 
    &beta, y->data, (int) y->stride);
}

void 
__blas_trsv(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
                 const matrix *A, vector *x){

  cublasHandle_t hdl;
  cublasOperation_t tA;
  cublasDiagType_t di;
  cublasFillMode_t ul;

  // transpose = transpose xor A->rowmajor
  // uplo = uplo xor A->rowmajor 
  tA = (Trans==CblasTrans) != (A->rowmajor==CblasRowMajor) ? \
      CUBLAS_OP_T : CUBLAS_OP_N;
  ul = (Uplo==CblasLower) != (A->rowmajor==CblasRowMajor) ? \
      CUBLAS_FILLMODE_LOWER : CUBLAS_FILLMODE_UPPER;
  di = Diag==CblasNonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;  


  if ( !__blas_check_handle(linalg_handle) ) return;
  hdl = *(cublasHandle_t *) linalg_handle;

  CUBLAS(trsv)(*(cublasHandle_t *) linalg_handle, ul, tA, di, 
    (int) A->size1, A->data, (int) A->tda, x->data, (int) x->stride); 
}

/* BLAS LEVEL 3 */

void 
__blas_syrk(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t Trans, ok_float alpha, 
                 const matrix * A, ok_float beta, matrix * C) {

  cublasOperation_t tA;
  cublasFillMode_t ul;

  const int K = (Trans == CblasNoTrans) ? (int) A->size2 : (int) A->size1;

  // transpose = transpose xor A->rowmajor
  // uplo = uplo xor A->rowmajor 
  tA = (Trans==CblasTrans) != (A->rowmajor==CblasRowMajor) ? \
      CUBLAS_OP_T : CUBLAS_OP_N;
  ul = (Uplo==CblasLower) != (A->rowmajor==CblasRowMajor) ? \
      CUBLAS_FILLMODE_LOWER : CUBLAS_FILLMODE_UPPER;

  if ( !__blas_check_handle(linalg_handle) ) return;
  if ( matrix_order_compat(A, C, "A", "C", "blas_syrk") ){
    CUBLAS(syrk)(*(cublasHandle_t *) linalg_handle, ul, tA, 
      (int) C->size2 , K, &alpha, A->data, (int) A->tda,
      &beta, C->data, (int) C->tda);
  }
  CUDA_CHECK_ERR;
}

void 
__blas_gemm(void * linalg_handle, CBLAS_TRANSPOSE_t TransA, 
                 CBLAS_TRANSPOSE_t TransB, ok_float alpha, 
                 const matrix * A, const matrix * B, 
                 ok_float beta, matrix * C){

  cublasOperation_t tA, tB;
  int s1, s2;

  const int NA = (TransA == CblasNoTrans) ? (int) A->size2 : (int) A->size1; 
  s1 = (A->rowmajor==CblasRowMajor) ? C->size2 : C->size1;
  s2 = (A->rowmajor==CblasRowMajor) ? C->size1 : C->size2;

  if (A->rowmajor==CblasRowMajor){
    tA = TransA == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
    tB = TransB == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  } else {
    tA = TransB == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
    tB = TransA == CblasTrans ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  if ( !__blas_check_handle(linalg_handle) ) return;
  if ( matrix_order_compat(A, B, "A", "B", "gemm") && 
        matrix_order_compat(A, C, "A", "C", "blas_gemm") ){
    CUBLAS(gemm)(*(cublasHandle_t *) linalg_handle, tA, tB, 
      s1, s2, NA, &alpha, A->data, (int) A->tda, 
      B->data, (int) B->tda, &beta, C->data, (int) C->tda);
  }
  CUDA_CHECK_ERR;
}

void 
__blas_trsm(void * linalg_handle, CBLAS_SIDE_t Side, 
                 CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
                 CBLAS_DIAG_t Diag, ok_float alpha, 
                 const matrix *A, matrix *B) {

  cublasOperation_t tA;
  cublasFillMode_t ul;
  cublasSideMode_t si;
  cublasDiagType_t di;

  tA = (Trans==CblasTrans) != (A->rowmajor==CblasRowMajor) ? \
      CUBLAS_OP_T : CUBLAS_OP_N;
  ul = (Uplo==CblasLower) != (A->rowmajor==CblasRowMajor) ? \
      CUBLAS_FILLMODE_LOWER : CUBLAS_FILLMODE_UPPER;
  si = Side==CblasLeft ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  di = Diag==CblasNonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;  

  if ( !__blas_check_handle(linalg_handle) ) return;
  if ( matrix_order_compat(A, B, "A", "B", "blas_trsm") )
    CUBLAS(trsm)(*(cublasHandle_t *) linalg_handle, si, ul, tA, di, 
      (int) B->size1, (int) B->size2, &alpha, A->data, 
      (int) A->tda, B->data, (int) B->tda);
  CUDA_CHECK_ERR;
}

/* LINEAR ALGEBRA routines */

/* cholesky decomposition of a single block */
__global__ void 
__block_chol(ok_float * A, uint iter, uint tda, uint rowmajor) {
  
  uint col, row, mat_dim, global_col, global_row, i;
  const uint kSmTda = kTileSize + 1u;
  __shared__ ok_float L[kSmTda * kTileSize];
  ok_float rl11;

  col = threadIdx.x;
  row = threadIdx.y;
  mat_dim = blockDim.x;

  global_col = iter * kTileSize + col;
  global_row = iter * kTileSize + row;

  __get_matrix(L, row, col, kSmTda, rowmajor) = 
      __get_matrix(A, global_row, global_col, tda, rowmajor);
  __syncthreads();

  for (i = 0; i < mat_dim; ++i) {
    /* l11 = sqrt(a11) */
    rl11 = math_rsqrt(__get_matrix(L, i, i, kSmTda, rowmajor));
    __syncthreads();


    /* l21 = a21 / l11 */
    if (row >= i && col == 0)
      __get_matrix(L, row, i, kSmTda, rowmajor) *= rl11;
    __syncthreads();


    /* a22 -= l21 * l21' */
    if (row >= col && col > i)
      __get_matrix(L, row, col, kSmTda, rowmajor) -=
          __get_matrix(L, col, i, kSmTda, rowmajor) * 
          __get_matrix(L, row, i, kSmTda, rowmajor);
    __syncthreads();
  }

  if (row >= col)
    __get_matrix(A, global_row, global_col, tda, rowmajor) = 
        __get_matrix(L, row, col, kSmTda, rowmajor);
}

__global__ void 
__block_trsv(ok_float * A, uint iter, uint n, 
                             uint tda, uint rowmajor) {
  
  uint tile_idx, row, global_row, global_col, i, j;
  const uint kSmTda = kTileSize + 1u;
  __shared__ ok_float L[kSmTda * kTileSize];
  __shared__ ok_float A12[kSmTda * kTileSize];

  tile_idx = blockIdx.x;
  row = threadIdx.x;
  global_col = iter * kTileSize;
  global_row = iter * kTileSize + row;

  // Load A -> L column-wise.
  for (i = 0; i < kTileSize; ++i)
    __get_matrix(L, row, i, kSmTda, rowmajor) =
        __get_matrix(A, global_row, global_col + i, tda, rowmajor);

  global_row = row + (iter + tile_idx + 1u) * kTileSize;

  if (global_row < n) {
    for (i = 0; i < kTileSize; ++i)
      __get_matrix(A12, row, i, kSmTda, rowmajor) = 
          __get_matrix(A, global_row, global_col + i, tda, rowmajor);
  }
  __syncthreads();

  if (global_row < n) {
    for (i = 0; i < kTileSize; ++i) {
      for (j = 0; j < i; ++j)
        __get_matrix(A12, row, i, kSmTda) -=
            __get_matrix(A12, row, j, kSmTda, rowmajor) * 
            __get_matrix(L, i, j, kSmTda, rowmajor);
      __get_matrix(A12, row, i, kSmTda, rowmajor) /= 
        __get_matrix(L, i, i, kSmTda, rowmajor);
    }
  }
  __syncthreads();

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      __get_matrix(A, global_row, global_col + i, tda, rowmajor) =
          __get_matrix(A12, row, i, kSmTda, rowmajor);
  }
}





/*
// Block Cholesky.
//   l11 l11^T = a11
//   l21 = a21 l11^(-T)
//   a22 = a22 - l21 l21^T
//
// Stores result in Lower triangular part.
*/
cublasStatus_t 
__linalg_cholesky_decomp(void * linalg_handle, matrix * A) {

  cublasStatus_t err;
  cudaStream_t stm;
  matrix * L21, * A22;
  size_t n = A->size1, blk_dim, n11;
  uint num_tiles, blk_dim_1d, grid_dim, i;
  dim3 block_dim;


  L21= &(matrix){0,0,0,OK_NULL,CblasRowMajor};
  A22= &(matrix){0,0,0,OK_NULL,CblasRowMajor};

  err = cublasGetStream(linalg_handle, &stm);
  num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;

  for (i = 0; i < num_tiles; ++i) {
    if (err != CUBLAS_STATUS_SUCCESS) break;

    /* L11 = chol(A11) */
    block_dim_1d = std::min<uint>(kTileSize, A->size1 - i * kTileSize);
    block_dim = (dim3){block_dim_1d, block_dim_1d};
    __block_chol<<<1, block_dim, 0, stm>>>(A->data, i, (uint) A->tda,
                                  (uint) A->rowmajor == CblasRowMajor);

    if (i < num_tiles - 1u) {

      /* L21 = A21 L21^-T */
      grid_dim = num_tiles - i - 1u;
      __block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data, i, 
                                  (uint) A->size1, (uint) A->tda,
                                  (uint) A->rowmajor == CblasRowMajor);

      err = __matrix_submatrix(L21, A, (i + 1) * kTileSize, i * kTileSize,
          A->size1 - (i + 1) * kTileSize, kTileSize);

      /* A22 -= L21*L21^T */
      err = __matrix_submatrix(A22, A, (i + 1) * kTileSize,
          (i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
          A->size1 - (i + 1) * kTileSize);
      err = __blas_syrk(linalg_handle, CblasLower, CblasNoTrans,
          (ok_float) -1, &L21, (ok_float) 1, &A22);
    }
  }
  // CublasCheckError(err);



}


/* Cholesky solve */
void 
__linalg_cholesky_svx(void * linalg_handle, 
                           const matrix * L, vector * x) {
  cublasStatus_t err;

  __blas_trsv(linalg_handle, 
                  CblasLower, CblasNoTrans, CblasNonUnit, L, x);
  // CublasCheckError(err);

  __blas_trsv(linalg_handle, 
                  CblasLower, CblasTrans, CblasNonUnit, L, x);
  // CublasCheckError(err);



}



#ifdef __cplusplus
}
#endif