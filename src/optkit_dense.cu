#include "optkit_dense.h"
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

template <typename BinaryFunction>
void 
__transform_rr(strided_range_t r1, strided_range_t r2, BinaryFunction f){
  thrust::transform(r1.begin(), r1.end(), r2.begin(), 
    r1.begin(), f);
}

template <typename BinaryFunction>
void
__transform_rc(strided_range_t r, ok_float x, BinaryFunction f){
  thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
    r.begin(), f);
}

template <typename BinaryFunction>
void
__transform_cr(ok_float x, strided_range_t r, BinaryFunction f){
  thrust::transform(r.begin(), r.end(), constant_iterator_t(x),
    r.begin(), f);
}

void __thrust_vector_scale(vector * v, ok_float x) {
  strided_range_t r = __make_strided_range(v);
  __transform_rc(r, x, thrust::multiplies<ok_float>());
}

void __thrust_vector_add(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::plus<ok_float>());
}

void __thrust_vector_sub(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::minus<ok_float>());
}

void __thrust_vector_mul(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::multiplies<ok_float>());
}

void __thrust_vector_div(vector * v1, const vector * v2) {
  strided_range_t r1 = __make_strided_range(v1);
  strided_range_t r2 = __make_const_strided_range(v2);
  __transform_rr(r1, r2, thrust::divides<ok_float>());
}

void __thrust_vector_add_constant(vector * v, const ok_float x) {
  strided_range_t r = __make_strided_range(v);
  __transform_rc(r, x, thrust::plus<ok_float>());
}


/* TODO */
/*
void __thrust_vector_pow(vector * v, const ok_float p) {}
*/


#ifdef __cplusplus
extern "C" {
#endif

/* VECTOR helper methods for CUDA */
__global__ void 
__vector_set(ok_float * data, ok_float val, size_t stride, size_t size) {
  uint i, thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (i = thread_id; i < size; i += gridDim.x * blockDim.x)
    data[i * stride] = val;
}

void 
__vector_set_all(vector * v, ok_float x){
  uint grid_dim = calc_grid_dim(v->size);
  __vector_set<<<grid_dim, kBlockSize>>>(v->data, x, v->stride, v->size);
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
vector_alloc(vector * v, size_t n) {
  if (!__vector_exists(v)) return;
  v->size=n;
  v->stride=1;
  ok_alloc_gpu(v->data, n * sizeof(ok_float));
}

void 
vector_calloc(vector * v, size_t n) {
  vector_alloc(v, n);
  __vector_set_all(v, ok_float(0));
}

void 
vector_free(vector * v) {
  if (v != OK_NULL)
    if (v->data != OK_NULL) ok_free_gpu(v->data);
}


void 
vector_set_all(vector * v, ok_float x) {
  __vector_set_all(v, x);
}

void 
vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n) {
  if (!__vector_exists(v_out)) return;
  v_out->size=n;
  v_out->stride=v_in->stride;
  v_out->data=v_in->data + offset * v_in->stride;
}

vector
vector_subvector_gen(vector * v_in, size_t offset, size_t n) {
  return (vector){
    .size = n,
    .stride = v_in->stride,
    .data = v_in->data + offset * v_in->stride
  };
}

void 
vector_view_array(vector * v, ok_float * base, size_t n) {
  if (!__vector_exists(v)) return;
  v->size=n;
  v->stride=1;
  v->data=base;
}


void 
vector_memcpy_vv(vector * v1, const vector * v2) {
  uint grid_dim;
  if ( v1->stride == 1 && v2->stride == 1) {
    ok_memcpy_gpu(v1->data, v2->data, v1->size * sizeof(ok_float));
  } else {
    grid_dim = calc_grid_dim(v1->size);
    __strided_memcpy<<<grid_dim, kBlockSize>>>(v1->data, v1->stride,
      v2->data, v2->stride, v1->size);
  }
}

void 
vector_memcpy_va(vector * v, const ok_float *y, size_t stride_y) {
  uint i;
  if (v->stride == 1 && stride_y == 1)
    ok_memcpy_gpu(v->data, y, v->size * sizeof(ok_float));
  else
    for (i = 0; i < v->size; ++i)
      ok_memcpy_gpu(v->data + i * v->stride, y + i * stride_y, 
        sizeof(ok_float));
}

void 
vector_memcpy_av(ok_float *x, const vector *v, size_t stride_x) {
  uint i; 
  if (v->stride ==1 && stride_x == 1) 
    ok_memcpy_gpu(x, v->data, v->size * sizeof(ok_float));
  else 
    for (i = 0; i < v->size; ++i)
      ok_memcpy_gpu(x + i * stride_x, v->data + i * v->stride,
       sizeof(ok_float));  
}



void 
vector_print(const vector * v) {
  uint i;
  ok_float v_host[v->size];
  // ok_float * v_host = (ok_float *) malloc(v->size * sizeof(ok_float));
  vector_memcpy_av(v_host, v, 1);
  for (i = 0; i < v->size; ++i)
    printf("%e ", v_host[i]);
  printf("\n");
  // ok_free(v_host);
}

void 
vector_scale(vector * v, ok_float x) {
  __thrust_vector_scale(v, x);
  CUDA_CHECK_ERR;
}

void 
vector_add(vector * v1, const vector * v2) {
  __thrust_vector_add(v1, v2);
  CUDA_CHECK_ERR;
}

void 
vector_sub(vector * v1, const vector * v2) {
  __thrust_vector_sub(v1, v2);
  CUDA_CHECK_ERR;
}

void 
vector_mul(vector * v1, const vector * v2) {
  __thrust_vector_mul(v1, v2);
  CUDA_CHECK_ERR;
}

void 
vector_div(vector * v1, const vector * v2) {
  __thrust_vector_div(v1, v2);
  CUDA_CHECK_ERR;
}

void 
vector_add_constant(vector * v, const ok_float x) {
  __thrust_vector_add_constant(v, x);
  CUDA_CHECK_ERR;
}

void 
vector_pow(vector * v, const ok_float x) {
  /* __thrust_vector_pow(v, x); */
  /* CUDA_CHECK_ERR */
}


/* MATRIX CUDA helper methods */
__global__ void 
__matrix_set(ok_float * data, ok_float x, size_t stride, 
  size_t size1, size_t size2, CBLAS_ORDER_t rowmajor){
  uint i, j;
  uint thread_id_row = blockIdx.x * blockDim.x + threadIdx.x;
  uint thread_id_col = blockIdx.y * blockDim.y + threadIdx.y;
  if (rowmajor == CblasRowMajor)
    for (i = thread_id_row; i < size1; i += gridDim.x * blockDim.x)
      for (j = thread_id_col; j < size2; j += gridDim.y * blockDim.y)
        data[i * stride + j] = x;
  else
    for (j = thread_id_col; j < size2; j += gridDim.y * blockDim.y)
      for (i = thread_id_row; i < size1; i += gridDim.x * blockDim.x)
        data[i + j * stride] = x;
}

void 
__matrix_set_all(matrix * A, ok_float x) {
  uint grid_dimx = calc_grid_dim(A->size1);
  uint grid_dimy = calc_grid_dim(A->size2);
  dim3 grid_dim(grid_dimx, grid_dimy, 1u);
  dim3 block_dim(kBlockSize, kBlockSize - 1, 1u);
  __matrix_set<<<grid_dim, block_dim>>>(A->data, x, 
    A->tda, A->size1, A->size2, A->rowmajor);
  CUDA_CHECK_ERR;
}


__global__ void
__matrix_add_constant_diag(ok_float * data, ok_float x, size_t stride){
  uint i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i * stride + i] += x;
}


/* CUDA helper kernels */
__device__ inline ok_float& 
__matrix_get(ok_float * A, uint i, uint j, uint stride, uint rowmajor) {
  if (rowmajor) return A[i * stride + j];
  else return A[i + j * stride];
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
matrix_alloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  A->size1 = m;
  A->size2 = n;
  A->tda = (ord == CblasRowMajor) ? n : m;
  ok_alloc_gpu(A->data, m * n * sizeof(ok_float));
  A->rowmajor = ord;
}

void 
matrix_calloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  matrix_alloc(A, m, n, ord);
  cudaMemset(A->data, 0, m * n * sizeof(ok_float));
  CUDA_CHECK_ERR;
  // __matrix_set_all(A, (ok_float) 0);
}

void 
matrix_free(matrix * A) {
  if (A == OK_NULL || A->data != OK_NULL) return;
  ok_free_gpu(A->data);
}

void
matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1, size_t n2){
  __matrix_exists(A_sub);
  A_sub->size1 = n1;
  A_sub->size2 = n2;
  A_sub->tda = A->tda;
  A_sub->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) + j : A->data + i + (j * A->tda);
  A_sub->rowmajor = A->rowmajor;
}

matrix
matrix_submatrix_gen(matrix * A, size_t i, size_t j, size_t n1, size_t n2){
  return (matrix){
    .size1 = n1,
    .size2 = n2,
    .tda = A->tda,
    .data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) + j : A->data + i + (j * A->tda),
    .rowmajor = A->rowmajor
  };

}

void 
matrix_row(vector * row, matrix * A, size_t i) {
  if (!__vector_exists(row)) return;
  row->size = A->size2;
  row->stride = (A->rowmajor == CblasRowMajor) ? 1 : A->tda;
  row->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) : A->data + i;
}

void 
matrix_column(vector * col, matrix *A, size_t j) {
  if (!__vector_exists(col)) return;
  col->size = A->size1;
  col->stride = (A->rowmajor == CblasRowMajor) ? A->tda : 1, 
  col->data = (A->rowmajor == CblasRowMajor) ? A->data + j : A->data + (j * A->tda); 
}

void 
matrix_diagonal(vector * diag, matrix *A) {
  if (!__vector_exists(diag)) return;
  diag->data = A->data;
  diag->stride = A->tda + 1;
  diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

void 
matrix_view_array(matrix * A, const ok_float *base, size_t n1, 
  size_t n2, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  A->size1 = n1;
  A->size2 = n2;
  A->tda = (ord == CblasRowMajor) ? n2 : n1;
  A->data = (ok_float *) base;
  A->rowmajor = ord;
}

void matrix_set_all(matrix * A, ok_float x) {
  __matrix_set_all(A, x);
}


void 
matrix_memcpy_mm(matrix * A, const matrix * B) {
  uint i, j, grid_dim;
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n");
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n");
  else{ 
    if (A->rowmajor == B->rowmajor)  
      ok_memcpy_gpu(A->data, B->data, A->size1 * A->size2 * sizeof(ok_float));
    else if (A->rowmajor == CblasRowMajor){
      /* A row major, B column major */
      grid_dim = calc_grid_dim(A->size1);
      for (i = 0; i < A->size1; ++i)
        __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + i * A->size2, 
          1, B->data + i, A->tda, A->size2);
    } else {
      /* A column major, B row major */
      grid_dim = calc_grid_dim(A->size2);
      for (j= 0; j < A->size2; ++j)
        __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + j * A->size1, 
          1, B->data + j, A->tda, A->size1);
    }
    CUDA_CHECK_ERR;
  }
}

void 
matrix_memcpy_ma(matrix * A, const ok_float * B, 
  const CBLAS_ORDER_t rowmajor) {
  uint i, j, grid_dim;
  ok_float * row, * col;

  if (rowmajor == A->rowmajor) {
    ok_memcpy_gpu(A->data, B, A->size1 * A->size2 * sizeof(ok_float));
  } else if (rowmajor == CblasColMajor) {
    /* A row major, B column major */
    ok_alloc_gpu(col, A->size1 * sizeof(ok_float));
    grid_dim = calc_grid_dim(A->size1);
    for (j = 0; j < A->size2; ++j){
      ok_memcpy_gpu(col, B + j * A->size1, A->size1 * sizeof(ok_float));
      __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + j,
        A->tda, col, 1, A->size1);
    }
  } else {
    /* A column major, B row major */
    ok_alloc_gpu(row, A->size2 * sizeof(ok_float));
    grid_dim = calc_grid_dim(A->size2);
    for (i = 0; i < A->size1; ++j){
      ok_memcpy_gpu(col, B + i * A->size1, A->size1 * sizeof(ok_float));
      __strided_memcpy<<<grid_dim, kBlockSize>>>(A->data + i,
        A->tda, row, 1, A->size2);
    }
  }
  CUDA_CHECK_ERR;
}

void 
matrix_memcpy_am(ok_float * A, const matrix * B, 
  const CBLAS_ORDER_t rowmajor) {
  uint i, j, grid_dim;
  ok_float * row, * col;
  if (rowmajor == B->rowmajor) {
    ok_memcpy_gpu(A, B->data, B->size1 * B->size2 * sizeof(ok_float));
  } else if (rowmajor == CblasRowMajor) {
    /* A row major, B column major */
    ok_alloc_gpu(row, B->size2 * sizeof(ok_float));
    grid_dim = calc_grid_dim(B->size2);
    for (i = 0; i < B->size1; ++i){
      __strided_memcpy<<<grid_dim, kBlockSize>>>(row, 1, 
        B->data + i, B->tda, B->size2);
      ok_memcpy_gpu(A + i * B->size2, row, B->size2 * sizeof(ok_float));
    }
    ok_free_gpu(row);
  } else {
    /* A column major, B row major */
    ok_alloc_gpu(col, B->size1 * sizeof(ok_float));
    grid_dim = calc_grid_dim(B->size1);
    for (j = 0; j < B->size2; ++j){
      __strided_memcpy<<<grid_dim, kBlockSize>>>(col, 1,
        B->data + j, B->tda, B->size1);
      ok_memcpy_gpu(A + j * B->size1, col, B->size1 * sizeof(ok_float));
    }
  }
  CUDA_CHECK_ERR;
}

void 
matrix_print(matrix * A) {
  ok_float A_row_host[A->size2];

  for (uint i = 0; i < A->size1; ++i) {
    matrix A_row  = matrix_submatrix_gen(A, i, 0, 1, A->size2); 
    matrix_memcpy_am(A_row_host, &A_row, A->rowmajor);
    for (uint j = 0; j < A->size2; ++j)
      if (A->rowmajor == CblasRowMajor)
        printf("%0.2e ", A_row_host[j]);
      else
        printf("%0.2e ", A_row_host[j * A->tda]);
    printf("\n");
  }
  printf("\n");
}


void 
matrix_scale(matrix *A, ok_float x) {
  vector row_col = (vector){0,0,OK_NULL};
  size_t i;
  if (A->rowmajor == CblasRowMajor)
    for(i = 0; i < A->size1; ++i){
      matrix_row(&row_col, A, i);
      __vector_scale(&row_col, x);
    }
  else{
    for(i = 0; i < A->size2; ++i){
      matrix_column(&row_col, A, i);
      __vector_scale(&row_col, x);
    }
  }
}

// typedef struct ok_blas_handle {
//   cublasHandle_t h;
// } ok_blas_handle_t;


/* BLAS routines */

inline int 
__blas_check_handle(void * handle){
  if (handle == OK_NULL) return 0;
  else return 1; 
}

void 
blas_make_handle(void ** handle){
  cublasStatus_t status;
  cublasHandle_t * hdl = (cublasHandle_t *) malloc(sizeof(cublasHandle_t));
  status = cublasCreate(hdl);
  if (status != CUBLAS_STATUS_SUCCESS){
    printf("CUBLAS initialization failed\n");
    ok_free(hdl);
    * handle = OK_NULL;
  } else {
    * handle = (void *) hdl;
  }
}

void 
blas_destroy_handle(void * handle){
  cublasDestroy(*(cublasHandle_t *) handle);
  handle = OK_NULL;
}


/* BLAS LEVEL 1 */
void 
blas_axpy(void * linalg_handle, ok_float alpha, 
                 const vector *x, vector *y) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CUBLAS(axpy)(*(cublasHandle_t *) linalg_handle,
   (int) x->size, &alpha, x->data, (int) x->stride, 
   y->data, (int) y->stride);
}

ok_float 
blas_nrm2(void * linalg_handle, const vector *x) {
  ok_float result = (ok_float) 0;
  if ( !__blas_check_handle(linalg_handle) ) return get_cuda_nan();
  CUBLAS(nrm2)(*(cublasHandle_t *) linalg_handle, 
    (int) x->size, x->data, (int) x->stride, &result);
  return result;
}

void 
blas_scal(void * linalg_handle, const ok_float alpha, vector *x) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CUBLAS(scal)(*(cublasHandle_t *) linalg_handle, 
    (int) x->size, &alpha, x->data, (int) x->stride);
}

ok_float 
blas_asum(void * linalg_handle, const vector * x) {
  ok_float result = (ok_float) 0;
  if ( !__blas_check_handle(linalg_handle) ) return get_cuda_nan();
  CUBLAS(asum)(*(cublasHandle_t *) linalg_handle, 
    (int) x->size, x->data, (int) x->stride, &result);
  return result;
}

ok_float 
blas_dot(void * linalg_handle, 
                    const vector * x, const vector * y) {
  
  ok_float result = (ok_float) 0;
  if ( !__blas_check_handle(linalg_handle) ) return get_cuda_nan();
  CUBLAS(dot)(*(cublasHandle_t *) linalg_handle,
    (int) x->size, x->data, (int) x->stride, 
    y->data, (int) y->stride, &result);
  return result;
}

/* BLAS LEVEL 2 */

void 
blas_gemv(void * linalg_handle, CBLAS_TRANSPOSE_t Trans, 
                ok_float alpha, const matrix *A, 
               const vector *x, ok_float beta, vector *y){

  cublasOperation_t tA;
  int s1, s2;

  if (A->rowmajor==CblasColMajor){
    tA = (Trans == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  } else {
    tA = (Trans == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  }
  s1 = (A->rowmajor==CblasRowMajor) ? (int) A->size2 : (int) A->size1;
  s2 = (A->rowmajor==CblasRowMajor) ? (int) A->size1 : (int) A->size2;

  if ( !__blas_check_handle(linalg_handle) ) return;
  CUBLAS(gemv)(*(cublasHandle_t *) linalg_handle, tA, s1, s2, 
    &alpha, A->data, (int) A->tda, x->data, (int) x->stride, 
    &beta, y->data, (int) y->stride);
}

void 
blas_trsv(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t Trans, CBLAS_DIAG_t Diag, 
                 const matrix *A, vector *x){

  cublasOperation_t tA;
  cublasDiagType_t di;
  cublasFillMode_t ul;

  if (A->rowmajor==CblasColMajor){
    tA = (Trans == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    ul = (Uplo == CblasLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  } else {
    tA = (Trans == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    ul = (Uplo == CblasLower) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  }
  di = Diag==CblasNonUnit ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;  


  if ( !__blas_check_handle(linalg_handle) ) return;
  CUBLAS(trsv)(*(cublasHandle_t *) linalg_handle, ul, tA, di, 
    (int) A->size1, A->data, (int) A->tda, x->data, (int) x->stride); 
}

/* BLAS LEVEL 3 */

void 
blas_syrk(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t Trans, ok_float alpha, 
                 const matrix * A, ok_float beta, matrix * C) {

  cublasOperation_t tA;
  cublasFillMode_t ul;

  const int k = (Trans == CblasNoTrans) ? (int) A->size2 : (int) A->size1;

  if (A->rowmajor==CblasColMajor){
    tA = (Trans == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    ul = (Uplo == CblasLower) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  } else {
    tA = (Trans == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    ul = (Uplo == CblasLower) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  }

  if ( !__blas_check_handle(linalg_handle) ) return;
  if ( matrix_order_compat(A, C, "A", "C", "blas_syrk") ){
    CUBLAS(syrk)(*(cublasHandle_t *) linalg_handle, ul, tA, 
      (int) C->size2 , k, &alpha, A->data, (int) A->tda,
      &beta, C->data, (int) C->tda);
  }
  CUDA_CHECK_ERR;
}

void 
blas_gemm(void * linalg_handle, CBLAS_TRANSPOSE_t TransA, 
                 CBLAS_TRANSPOSE_t TransB, ok_float alpha, 
                 const matrix * A, const matrix * B, 
                 ok_float beta, matrix * C){

  cublasOperation_t tA, tB;
  int s1, s2;

  const int k = (TransA == CblasNoTrans) ? (int) A->size2 : (int) A->size1; 
  s1 = (A->rowmajor==CblasRowMajor) ? (int) C->size2 : (int) C->size1;
  s2 = (A->rowmajor==CblasRowMajor) ? (int) C->size1 : (int) C->size2;

  if (A->rowmajor==CblasColMajor){
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
      s1, s2, k, &alpha, A->data, (int) A->tda, 
      B->data, (int) B->tda, &beta, C->data, (int) C->tda);
  }
  CUDA_CHECK_ERR;
}


void 
blas_trsm(void * linalg_handle, CBLAS_SIDE_t Side, 
                 CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans,
                 CBLAS_DIAG_t Diag, ok_float alpha, 
                 const matrix *A, matrix *B) {

printf("Method `blas_trsm()` not implemented for GPU\n");
}

/* LINEAR ALGEBRA routines */

/* cholesky decomposition of a single block */
__global__ void 
__block_chol(ok_float * A, uint iter, uint tda, uint rowmajor) {
  
  uint col, row, mat_dim, global_col, global_row, i;
  const uint kSmTda = kTileSize + 1u;
  __shared__ ok_float L[kSmTda * kTileSize];
  ok_float a11;

  col = threadIdx.x;
  row = threadIdx.y;
  mat_dim = blockDim.x;

  global_col = iter * kTileSize + col;
  global_row = iter * kTileSize + row;

  __matrix_get(L, row, col, kSmTda, rowmajor) = 
      __matrix_get(A, global_row, global_col, tda, rowmajor);
  __syncthreads();


  for (i = 0; i < mat_dim; ++i) {
    /* l11 = sqrt(a11) */
    a11 = sqrt(__matrix_get(L, i, i, kSmTda, rowmajor));
    __syncthreads();


    /* l21 = a21 / l11 */
    if (row >= i && col == 0)
      __matrix_get(L, row, i, kSmTda, rowmajor) /= a11;
    __syncthreads();


    /* a22 -= l21 * l21' */
    if (row >= col && col > i)
      __matrix_get(L, row, col, kSmTda, rowmajor) -=
          __matrix_get(L, col, i, kSmTda, rowmajor) * 
          __matrix_get(L, row, i, kSmTda, rowmajor);
    __syncthreads();
  }

  if (row >= col)
    __matrix_get(A, global_row, global_col, tda, rowmajor) = 
        __matrix_get(L, row, col, kSmTda, rowmajor);
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
    __matrix_get(L, row, i, kSmTda, rowmajor) =
        __matrix_get(A, global_row, global_col + i, tda, rowmajor);

  global_row = row + (iter + tile_idx + 1u) * kTileSize;

  if (global_row < n) {
    for (i = 0; i < kTileSize; ++i)
      __matrix_get(A12, row, i, kSmTda, rowmajor) = 
          __matrix_get(A, global_row, global_col + i, tda, rowmajor);
  }
  __syncthreads();

  if (global_row < n) {
    for (i = 0; i < kTileSize; ++i) {
      for (j = 0; j < i; ++j)
        __matrix_get(A12, row, i, kSmTda, rowmajor) -=
            __matrix_get(A12, row, j, kSmTda, rowmajor) * 
            __matrix_get(L, i, j, kSmTda, rowmajor);
      __matrix_get(A12, row, i, kSmTda, rowmajor) /= 
        __matrix_get(L, i, i, kSmTda, rowmajor);
    }
  }
  __syncthreads();

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      __matrix_get(A, global_row, global_col + i, tda, rowmajor) =
          __matrix_get(A12, row, i, kSmTda, rowmajor);
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
void
linalg_cholesky_decomp(void * linalg_handle, matrix * A) {

  cublasStatus_t err;
  cudaStream_t stm;
  uint num_tiles, grid_dim, i;
#ifdef OK_DEBUG
  uint ii;
#endif

  err = cublasGetStream(*(cublasHandle_t *) linalg_handle, &stm);
  num_tiles = (A->size1 + kTileSize - 1u) / kTileSize;

  for (i = 0; i < num_tiles; ++i) {
    if (err != CUBLAS_STATUS_SUCCESS) break;

    /* L11 = chol(A11) */
    uint block_dim_1d = kTileSize < A->size1 - i * kTileSize ? \
                        kTileSize : A->size1 - i * kTileSize;
    dim3 block_dim(block_dim_1d, block_dim_1d);


    matrix L11 = matrix_submatrix_gen(A, i * kTileSize, i * kTileSize,
      block_dim_1d, block_dim_1d);

#ifdef OK_DEBUG
    printf("L11 before:\n");
      matrix_print(&L11);
#endif

    __block_chol<<<1, block_dim, 0, stm>>>(A->data, i, (uint) A->tda,
                                  (uint) A->rowmajor == CblasRowMajor);
    CUDA_CHECK_ERR;

#ifdef OK_DEBUG
    printf("L11 after:\n");
      matrix_print(&L11);
#endif

    if (i < num_tiles - 1u) {

      /* L21 = A21 L21^-T */
      grid_dim = num_tiles - i - 1u;

      matrix L21 = matrix_submatrix_gen(A, (i + 1) * kTileSize, 
        i * kTileSize, A->size1 - (i + 1) * kTileSize, kTileSize);

#ifdef OK_DEBUG
      printf("L21 before:\n");
      matrix_print(&L21);
#endif

      __block_trsv<<<grid_dim, kTileSize, 0, stm>>>(A->data, i, 
                                  (uint) A->size1, (uint) A->tda,
                                  (uint) A->rowmajor == CblasRowMajor);
      CUDA_CHECK_ERR;

#ifdef OK_DEBUG
      printf("L21 after:\n");
      matrix_print(&L21);
#endif

      /* A22 -= L21*L21^T */
      matrix A22 = matrix_submatrix_gen(A, (i + 1) * kTileSize,
          (i + 1) * kTileSize, A->size1 - (i + 1) * kTileSize,
          A->size1 - (i + 1) * kTileSize);

#ifdef OK_DEBUG
      printf("L22 before:\n");
      matrix_print(&A22);
#endif

      blas_syrk(linalg_handle, CblasLower, CblasNoTrans,
          (ok_float) -1, &L21, (ok_float) 1, &A22);

#ifdef OK_DEBUG
      printf("L22 before:\n");
      matrix_print(&A22);
#endif


    }
  }
  // CublasCheckError(err);
}




/* Cholesky solve */
void 
linalg_cholesky_svx(void * linalg_handle, 
                           const matrix * L, vector * x) {

  blas_trsv(linalg_handle, 
                  CblasLower, CblasNoTrans, CblasNonUnit, L, x);

  blas_trsv(linalg_handle, 
                  CblasLower, CblasTrans, CblasNonUnit, L, x);



}



#ifdef __cplusplus
}
#endif