#include "optkit_dense.h"
#include "optkit_dense.h"
#include "gsl_cblas.h"


/* TODO: CHANGE TRANSPOSE AND LU ENUMS */
/* TODO: FUNCTIONS TO CREATE AND DESTROY HANDLES */

#ifdef __cplusplus
extern "C" {
#endif


/* VECTOR methods */
inline int __vector_exists(vector * v) {
  if (v == OK_NULL){
    printf("Error: cannot write to uninitialized vector pointer\n");
    return 0;
  }
  else
    return 1;
}

void __vector_alloc(vector * v, size_t n) {
  if (!__vector_exists(v)) return;
  v->size=n;
  v->stride=1;
  v->data=(ok_float *) malloc(n * sizeof(ok_float));
}

void __vector_calloc(vector * v, size_t n) {
  __vector_alloc(v, n);
  memset(v->data, 0, n * sizeof(ok_float));
}

void __vector_free(vector * v) {
  if (v->data != OK_NULL) ok_free(v->data);
}

inline void __vector_set(vector * v, size_t i, ok_float x) {
  v->data[i * v->stride] = x;
}

ok_float __vector_get(const vector *v, size_t i) {
  return v->data[i * v->stride];
}

void __vector_set_all(vector * v, ok_float x) {
  for (uint i = 0; i < v->size; ++i)
    __vector_set(v, i, x);
}

void __vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n) {
  if (!__vector_exists(v_out)) return;
  v_out->size=n;
  v_out->stride=v_in->stride;
  v_out->data=v_in->data + offset * v_in->stride;
}

void __vector_view_array(vector * v, ok_float * base, size_t n) {
  if (!__vector_exists(v)) return;
  v->size=n;
  v->stride=1;
  v->data=base;
}


void __vector_memcpy_vv(vector * v1, const vector * v2) {
  if ( v1->stride == 1 && v2->stride == 1) {
    memcpy(v1->data, v2->data, v1->size * sizeof(ok_float));
  } else {
    for (uint i = 0; i < v1->size; ++i)
      __vector_set(v1, i, __vector_get(v2,i));
  }
}


void __vector_memcpy_va(vector * v, const ok_float *y) {
  if (v->stride == 1) {
    memcpy(v->data, y, v->size * sizeof(ok_float));
  } else {
    for (uint i = 0; i < v->size; ++i)
      __vector_set(v, i, y[i]);
  }
}

void __vector_memcpy_av(ok_float *x, const vector *v) {
  if (v->stride ==1) {
    memcpy(x, v->data, v->size * sizeof(ok_float));
  } else {
    for (uint i = 0; i < v->size; ++i)
      x[i] = __vector_get(v,i);
  }
}


void __vector_print(const vector * v) {
  for (uint i = 0; i < v->size; ++i)
    printf("%e ", __vector_get(v, i));
  printf("\n");
}

void __vector_scale(vector * v, ok_float x) {
  CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
}

void __vector_add(vector * v1, const vector * v2) {
  for (uint i = 0; i < v1->size; ++i)
    v1->data[i * v1->stride] += v2->data[i * v2->stride];
}

void __vector_sub(vector * v1, const vector * v2) {
  for (uint i = 0; i < v1->size; ++i)
    v1->data[i * v1->stride] -= v2->data[i * v2->stride];
}

void __vector_mul(vector * v1, const vector * v2) {
  for (uint i = 0; i < v1->size; ++i)
    v1->data[i * v1->stride] *= v2->data[i * v2->stride];
}

void __vector_div(vector * v1, const vector * v2) {
  for (uint i = 0; i < v1->size; ++i)
    v1->data[i * v1->stride] /= v2->data[i * v2->stride];
}

void __vector_add_constant(vector * v, const ok_float x) {
  for (uint i = 0; i < v->size; ++i)
    v->data[i * v->stride] += x;
}


/* MATRIX methods */
inline int __matrix_exists(matrix * A) {
  if (A == OK_NULL){
    printf("Error: cannot write to uninitialized matrix pointer\n");
    return 0;
  }
  else
    return 1;
}


void __matrix_alloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  A->size1 = m;
  A->size2 = n;
  A->tda = (ord == CblasRowMajor) ? n : m;
  A->data = (ok_float *) malloc(m * n * sizeof(ok_float));
  A->rowmajor = ord;
}

void __matrix_calloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  __matrix_alloc(A, m, n, ord);
  memset(A->data, 0, m * n * sizeof(ok_float));
}

void __matrix_free(matrix * A) {
  if (A->data != OK_NULL) ok_free(A->data);

}

void __matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1, size_t n2){
  __matrix_exists(A_sub);
  A_sub->size1 = n1;
  A_sub->size2 = n2;
  A_sub->tda = A->tda;
  A_sub->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) + j : A->data + i + (j * A->tda);
  A_sub->rowmajor = A->rowmajor;
}


void __matrix_row(vector * row, matrix * A, size_t i) {
  if (!__vector_exists(row)) return;
  row->size = A->size2;
  row->stride = (A->rowmajor == CblasRowMajor) ? 1 : A->tda;
  row->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) : A->data + i;
}

void __matrix_column(vector * col, matrix *A, size_t j) {
  if (!__vector_exists(col)) return;
  col->size = A->size1;
  col->stride = (A->rowmajor == CblasRowMajor) ? A->tda : 1, 
  col->data = (A->rowmajor == CblasRowMajor) ? A->data + j : A->data + (j * A->tda);
}

void __matrix_diagonal(vector * diag, matrix *A) {
  if (!__vector_exists(diag)) return;
  diag->data = A->data;
  diag->stride = A->tda + 1;
  diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

void __matrix_view_array(matrix * A, const ok_float *base, size_t n1, size_t n2, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  A->size1 = n1;
  A->size2 = n2;
  A->tda = (ord == CblasRowMajor) ? n2 : n1;
  A->data = (ok_float *) base;
  A->rowmajor = ord;
}

inline ok_float __matrix_get(const matrix * A, size_t i, size_t j) {
  if (A->rowmajor == CblasRowMajor)
    return A->data[i * A->tda + j];
  else
    return A->data[i + j * A->tda];
}

inline void __matrix_set(matrix *A, size_t i, size_t j, ok_float x){
  if (A->rowmajor == CblasRowMajor)
    A->data[i * A->tda + j] = x;
  else
    A->data[i + j * A->tda] = x;
}

void __matrix_set_all(matrix * A, ok_float x) {
  memset(A->data, x, A->size1 * A->size2 * sizeof(ok_float)); 
}

void __matrix_memcpy_mm(matrix * A, const matrix * B) {
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n");
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n");
  else{ 
    if (A->rowmajor == B->rowmajor)  
      memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(ok_float));
    else
      for (uint i = 0; i < A->size1; ++i)
        for (uint j = 0; j < A->size2; ++j)
          __matrix_set(A, i, j, __matrix_get(B, i , j));
  }
}

void __matrix_memcpy_ma(matrix * A, const ok_float * B) {
    memcpy(A->data, B, A->size1 * A->size2 * sizeof(ok_float));
}

void __matrix_memcpy_am(ok_float * A, const matrix * B) {
    memcpy(A, B->data, B->size1 * B->size2 * sizeof(ok_float));
}

void __matrix_print(const matrix * A) {
  for (uint i = 0; i < A->size1; ++i) {
    for (uint j = 0; j < A->size2; ++j)
      printf("%e ", __matrix_get(A, i, j));
    printf("\n");
  }
  printf("\n");
}


void __matrix_scale(matrix *A, ok_float x) {
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

inline int __blas_check_handle(void * handle){
  if (handle == OK_NULL) return 1;
  else { 
    printf("Error: CBLAS operations take no linear algebra handle.
            Non-void pointer provided for `linalg_handle.`\n");
    return 0;
  }
}

void __blas_make_handle(void * handle){
  handle = OK_NULL;
}

void __blas_destroy_handle(void * handle){
  handle = OK_NULL;
}


/* BLAS LEVEL 1 */
void __blas_axpy(void * blas_handle, ok_float alpha, 
                 const vector *x, vector *y) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CBLAS(axpy)( (int) x->size, alpha, x->data, (int) x->stride, 
               y->data, (int) y->stride);
}

ok_float __blas_nrm2(void * blas_handle, const vector *x) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  return CBLAS(nrm2)((int) x->size, x->data, (int) x->stride);
}

void __blas_scal(void * blas_handle, const ok_float alpha, vector *x) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CBLAS(scal)((int) x->size, alpha, x->data, (int) x->stride);
}

ok_float __blas_asum(void * linalg_handle, const vector * x) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  return CBLAS(asum)((int) x->size, x->data, (int) x->stride);
}

ok_float __blas_dot(void * linalg_handle, 
                    const vector * x, const vector * y) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  return CBLAS(dot)( (int) x->size, x->data, (int) x->stride, 
                     y->data, (int) y->stride);
}

/* BLAS LEVEL 2 */

void __blas_gemv(void * linalg_handle, CBLAS_TRANSPOSE_t TransA, 
                ok_float alpha, const matrix *A, 
               const vector *x, ok_float beta, vector *y){

  if ( !__blas_check_handle(linalg_handle) ) return;
  CBLAS(gemv)(A->rowmajor, TransA, (int) A->size1, (int) A->size2, 
              alpha, A->data, (int) A->tda, x->data, (int) x->stride, 
              beta, y->data, (int) y->stride);
}

void __blas_trsv(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
                 const matrix *A, vector *x){

  if ( !__blas_check_handle(linalg_handle) ) return;
  CBLAS(trsv)(A->rowmajor, Uplo, TransA, Diag, (int) A->size1, 
              A->data, (int) A->tda, x->data, (int) x->stride); 
}

/* BLAS LEVEL 3 */

void __blas_syrk(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t Trans, ok_float alpha, 
                 const matrix * A, ok_float beta, matrix * C) {

  if ( !__blas_check_handle(linalg_handle) ) return;
  if ( matrix_order_compat(A, C, "A", "C", "blas_syrk") ){
    const int K = (Trans == CblasNoTrans) ? (int) A->size2 : (int) A->size1;
    CBLAS(syrk)(A->rowmajor, Uplo, Trans, (int) C->size2 , K, alpha, 
                A->data, (int) A->tda, beta, C->data, (int) C->tda);
  }
}

void __blas_gemm(void * linalg_handle, CBLAS_TRANSPOSE_t TransA, 
                 CBLAS_TRANSPOSE_t TransB, ok_float alpha, 
                 const matrix * A, const matrix * B, 
                 ok_float beta, matrix * C){

  if ( !__blas_check_handle(linalg_handle) ) return;
  if ( matrix_order_compat(A, B, "A", "B", "gemm") && 
        matrix_order_compat(A, C, "A", "C", "blas_gemm") ){

    const int NA = (TransA == CblasNoTrans) ? (int) A->size2 : (int) A->size1; 
    CBLAS(gemm)(CblasRowMajor, TransA, TransB, (int) C->size1, (int) C->size2, NA, alpha, 
                A->data, (int) A->tda, B->data, (int) B->tda, beta, C->data, (int) C->tda);
  }
}

void __blas_trsm(void * linalg_handle, CBLAS_SIDE_t Side, 
                 CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
                 CBLAS_DIAG_t Diag, ok_float alpha, 
                 const matrix *A, matrix *B) {

  if ( !__blas_check_handle(linalg_handle) ) return;
  if ( matrix_order_compat(A, B, "A", "B", "blas_trsm") )
    CBLAS(trsm)(A->rowmajor, Side, Uplo, TransA, Diag,(int) B->size1, (int) B->size2, 
                alpha, A->data,(int) A->tda, B->data, (int) B->tda);
}

/* LINEAR ALGEBRA routines */

/* CUDA helper kernels */
__device__ inline ok_float& __Get(ok_float * A, uint i, 
                                   uint j, uint tda,
                                   uint rowmajor) {
  if (rowmajor)
    return &A[i + j * tda];
  else
    return &A[i * tda + j];
}

/* cholesky decomposition of a single block */
__global__ void __block_chol(ok_float * A, uint iter, 
                             uint tda, uint rowmajor) {
  
  uint col, row, mat_dim, global_col, global_row, i;
  const uint kSmTda = kTileSize + 1u;
  __shared__ ok_float L[kSmTda * kTileSize];
  ok_float rl11;

  col = threadIdx.x;
  row = threadIdx.y;
  mat_dim = blockDim.x;

  global_col = iter * kTileSize + col;
  global_row = iter * kTileSize + row;

  __Get(L, row, col, kSmTda, rowmajor) = 
      __Get(A, global_row, global_col, tda, rowmajor);
  __syncthreads();

  for (i = 0; i < mat_dim; ++i) {
    /* l11 = sqrt(a11) */
    rl11 = math_rsqrt(__Get(L, i, i, kSmTda, rowmajor));
    __syncthreads();


    /* l21 = a21 / l11 */
    if (row >= i && col == 0)
      __Get(L, row, i, kSmTda, rowmajor) *= rl11;
    __syncthreads();


    /* a22 -= l21 * l21' */
    if (row >= col && col > i)
      __Get(L, row, col, kSmTda, rowmajor) -=
          __Get(L, col, i, kSmTda, rowmajor) * 
          __Get(L, row, i, kSmTda, rowmajor);
    __syncthreads();
  }

  if (row >= col)
    __Get(A, global_row, global_col, tda, rowmajor) = 
        __Get(L, row, col, kSmTda, rowmajor);
}

__global__ void __block_trsv(ok_float * A, uint iter, uint n, 
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
    __Get(L, row, i, kSmTda, rowmajor) =
        __Get(A, global_row, global_col + i, tda, rowmajor);

  global_row = row + (iter + tile_idx + 1u) * kTileSize;

  if (global_row < n) {
    for (i = 0; i < kTileSize; ++i)
      __Get(A12, row, i, kSmTda, rowmajor) = 
          __Get(A, global_row, global_col + i, tda, rowmajor);
  }
  __syncthreads();

  if (global_row < n) {
    for (i = 0; i < kTileSize; ++i) {
      for (j = 0; j < i; ++j)
        __Get(A12, row, i, kSmTda) -=
            __Get(A12, row, j, kSmTda, rowmajor) * 
            __Get(L, i, j, kSmTda, rowmajor);
      __Get(A12, row, i, kSmTda, rowmajor) /= 
        __Get(L, i, i, kSmTda, rowmajor);
    }
  }
  __syncthreads();

  if (global_row < n) {
    for (uint i = 0; i < kTileSize; ++i)
      __Get(A, global_row, global_col + i, tda, rowmajor) =
          __Get(A12, row, i, kSmTda, rowmajor);
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
cublasStatus_t __linalg_cholesky_decomp(void * linalg_handle, matrix * A) {

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
void __linalg_cholesky_svx(void * linalg_handle, 
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