#include "optkit_dense.h"


#ifdef __cplusplus
extern "C" {
#endif

void 
denselib_version(int * maj, int * min, int * change, int * status){
  OPTKIT_VERSION(maj, min, change, status);
}

/* VECTOR methods */
inline int __vector_exists(vector * v) {
  if (v == OK_NULL){
    printf("Error: cannot write to uninitialized vector pointer\n");
    return 0;
  }
  else
    return 1;
}

void vector_alloc(vector * v, size_t n) {
  if (!__vector_exists(v)) return;
  v->size=n;
  v->stride=1;
  v->data=(ok_float *) malloc(n * sizeof(ok_float));
}

void vector_calloc(vector * v, size_t n) {
  vector_alloc(v, n);
  memset(v->data, 0, n * sizeof(ok_float));
}

void vector_free(vector * v) {
  if (v->data != OK_NULL) ok_free(v->data);
}

inline void __vector_set(vector * v, size_t i, ok_float x) {
  v->data[i * v->stride] = x;
}

ok_float __vector_get(const vector *v, size_t i) {
  return v->data[i * v->stride];
}

void vector_set_all(vector * v, ok_float x) {
	uint i;
 	for (i = 0; i < v->size; ++i)
		__vector_set(v, i, x);
}

void vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n) {
  if (!__vector_exists(v_out)) return;
  v_out->size=n;
  v_out->stride=v_in->stride;
  v_out->data=v_in->data + offset * v_in->stride;
}

void vector_view_array(vector * v, ok_float * base, size_t n) {
	if (!__vector_exists(v)) return;
	v->size=n;
	v->stride=1;
	v->data=base;
}


void vector_memcpy_vv(vector * v1, const vector * v2) {
	uint i;
 	if ( v1->stride == 1 && v2->stride == 1) {
		memcpy(v1->data, v2->data, v1->size * sizeof(ok_float));
  } else {
    for (i = 0; i < v1->size; ++i)
    			__vector_set(v1, i, __vector_get(v2,i));
	}
}

void vector_memcpy_va(vector * v, const ok_float *y, size_t stride_y) {
	uint i;
	if (v->stride == 1 && stride_y == 1) {
		memcpy(v->data, y, v->size * sizeof(ok_float));
	} else {
		for (i = 0; i < v->size; ++i)
			__vector_set(v, i, y[i * stride_y]);
	}
}

void vector_memcpy_av(ok_float *x, const vector *v, size_t stride_x) {
	uint i;
	if (v->stride ==1 && stride_x == 1) {
		memcpy(x, v->data, v->size * sizeof(ok_float));
	} else {
		for (i = 0; i < v->size; ++i)
			x[i * stride_x] = __vector_get(v,i);
	}
}

void vector_print(const vector * v) {
	uint i;
	for (i = 0; i < v->size; ++i)
		printf("%e ", __vector_get(v, i));
	printf("\n");
}

void vector_scale(vector * v, ok_float x) {
	CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
}

void vector_add(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] += v2->data[i * v2->stride];
}

void vector_sub(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] -= v2->data[i * v2->stride];
}

void vector_mul(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] *= v2->data[i * v2->stride];
}

void vector_div(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] /= v2->data[i * v2->stride];
}

void vector_add_constant(vector * v, const ok_float x) {
  uint i;
  for (i = 0; i < v->size; ++i)
    v->data[i * v->stride] += x;
}

void vector_abs(vector * v) {
  uint i;
  for (i = 0; i < v->size; ++i)
    v->data[i * v->stride] = MATH(fabs)(v->data[i * v->stride]);
}

void vector_recip(vector * v) {
  uint i;
  for (i = 0; i < v->size; ++i)
    v->data[i * v->stride] = (ok_float) 1 / v->data[i * v->stride];
}

void vector_sqrt(vector * v) {
  uint i;
  for (i = 0; i < v->size; ++i)
    v->data[i * v->stride] = MATH(sqrt)(v->data[i * v->stride]);
}

void vector_pow(vector * v, const ok_float x) {
  uint i;
  for (i = 0; i < v->size; ++i)
    v->data[i * v->stride] = MATH(pow)(v->data[i * v->stride], x);
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


void matrix_alloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  A->size1 = m;
  A->size2 = n;
  A->data = (ok_float *) malloc(m * n * sizeof(ok_float));
#ifndef OPTKIT_ORDER
  A->ld = (ord == CblasRowMajor) ? n : m;
  A->rowmajor = ord;
#elif OPTKIT_ORDER == 101
  A->ld = n;
  A->rowmajor = CblasRowMajor;
#else
  A->ld = m;
  A->rowmajor = CblasColMajor;
#endif
}

void matrix_calloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  matrix_alloc(A, m, n, ord);
  memset(A->data, 0, m * n * sizeof(ok_float));
}

void matrix_free(matrix * A) {
  if (!__matrix_exists(A)) return;
  if (A->data != OK_NULL) ok_free(A->data);

}

void matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1, size_t n2){
  if (!__matrix_exists(A_sub)) return;
  if (!__matrix_exists(A)) return;
  A_sub->size1 = n1;
  A_sub->size2 = n2;
  A_sub->ld = A->ld;
  #ifndef OPTKIT_ORDER
  A_sub->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->ld) + j : A->data + i + (j * A->ld);
  #elif OPTKIT_ORDER == 101
  A_sub->data = A->data + (i * A->ld) + j;
  #else
  A_sub->data = A->data + i + (j * A->ld);
  #endif
  A_sub->rowmajor = A->rowmajor;
}


void matrix_row(vector * row, matrix * A, size_t i) {
  if (!__vector_exists(row)) return;
  if (!__matrix_exists(A)) return;
  row->size = A->size2;
  #ifndef OPTKIT_ORDER
  row->stride = (A->rowmajor == CblasRowMajor) ? 1 : A->ld;
  row->data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->ld) : A->data + i;
  #elif OPTKIT_ORDER == 101
  row->stride = 1;
  row->data = A->data + (i * A->ld);
  #else
  row->stride = A->ld;
  row->data = A->data + i;
  #endif
}

void matrix_column(vector * col, matrix *A, size_t j) {
  if (!__vector_exists(col)) return;
  if (!__matrix_exists(A)) return;
  col->size = A->size1;
  #ifndef OPTKIT_ORDER
  col->stride = (A->rowmajor == CblasRowMajor) ? A->ld : 1;
  col->data = (A->rowmajor == CblasRowMajor) ? A->data + j : A->data + (j * A->ld);
  #elif OPTKIT_ORDER == 101
  col->stride = A->ld;
  col->data = A->data + j;
  #else
  col->stride = 1;
  col->data = A->data + (j * A->ld);
  #endif  

}

void matrix_diagonal(vector * diag, matrix * A) {
  if (!__vector_exists(diag)) return;
  if (!__matrix_exists(A)) return;
  diag->data = A->data;
  diag->stride = A->ld + 1;
  diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

void matrix_cast_vector(vector * v, matrix * A) {
  if (!__vector_exists(v)) return;
  if (!__matrix_exists(A)) return;
  v->size = A->size1 * A->size2;
  v->stride = 1;
  v->data = A->data;
}

void matrix_view_array(matrix * A, const ok_float *base, size_t n1, size_t n2, CBLAS_ORDER_t ord) {
  if (!__matrix_exists(A)) return;
  A->size1 = n1;
  A->size2 = n2;
  A->data = (ok_float *) base;
  #ifndef OPTKIT_ORDER
  A->ld = (ord == CblasRowMajor) ? n2 : n1;
  A->rowmajor = ord;
  #elif OPTKIT_ORDER == 101
  A->ld = n2;
  A->rowmajor = CblasRowMajor;
  #else
  A->ld = n1;
  A->rowmajor = CblasColMajor;
  #endif
}

inline ok_float __matrix_get(const matrix * A, size_t i, size_t j) {
  #ifndef OPTKIT_ORDER
  if (A->rowmajor == CblasRowMajor)
    return A->data[i * A->ld + j];
  else
    return A->data[i + j * A->ld];
  #elif OPTKIT_ORDER == 101
  return A->data[i * A->ld + j];
  #else
  return A->data[i + j * A->ld];
  #endif

}

inline void __matrix_set(matrix *A, size_t i, size_t j, ok_float x){
  #ifndef OPTKIT_ORDER
  if (A->rowmajor == CblasRowMajor)
    A->data[i * A->ld + j] = x;
  else
    A->data[i + j * A->ld] = x;
  #elif OPTKIT_ORDER == 101
  A->data[i * A->ld + j] = x;
  #else
  A->data[i + j * A->ld] = x;
  #endif

}

void matrix_set_all(matrix * A, ok_float x) {
  if (!__matrix_exists(A)) return;
  memset(A->data, x, A->size1 * A->size2 * sizeof(ok_float));
}

void matrix_memcpy_mm(matrix * A, const matrix * B) {
  #ifndef OPTKIT_ORDER
  uint i,j;
  #endif
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n");
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n");
  else{
    #ifndef OPTKIT_ORDER
    if (A->rowmajor == B->rowmajor)  
      memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(ok_float));
    else    
      for (i = 0; i < A->size1; ++i)
        for (j = 0; j < A->size2; ++j)
          __matrix_set(A, i, j, __matrix_get(B, i , j));
    #else
    memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(ok_float));
    #endif
  }
}

void matrix_memcpy_ma(matrix * A, const ok_float * B, 
  const CBLAS_ORDER_t rowmajor) {

  uint i,j;
  if (A->rowmajor == rowmajor)
    memcpy(A->data, B, A->size1 * A->size2 * sizeof(ok_float));
  else {
    if (rowmajor == CblasRowMajor)
      for (i = 0; i < A->size1; ++i)
        for (j = 0; j < A->size2; ++j)
          __matrix_set(A,i,j, B[i * A->size2 + j]);
    else
      for (i = 0; i < A->size1; ++i)
        for (j = 0; j < A->size2; ++j)
            __matrix_set(A,i,j,B[i + j * A->size1]);
  }
}

void matrix_memcpy_am(ok_float * A, const matrix * B,
  const CBLAS_ORDER_t rowmajor) {
  uint i,j;
  if (B->rowmajor == rowmajor)
    memcpy(A, B->data, B->size1 * B->size2 * sizeof(ok_float));
  else {
    if (rowmajor)
      for (i = 0; i < B->size1; ++i)
        for (j = 0; j < B->size2; ++j)
          A[i + j * B->size1] = __matrix_get(B, i, j);
  else
      for (j = 0; j < B->size2; ++j)
        for (i = 0; i < B->size1; ++i)
          A[i * B->size2 + j] = __matrix_get(B, i, j);
  }
}


void matrix_print(matrix * A) {
  uint i,j;
  for (i = 0; i < A->size1; ++i) {
    for (j = 0; j < A->size2; ++j)
      printf("%e ", __matrix_get(A, i, j));
    printf("\n");
  }
  printf("\n");
}


void matrix_scale(matrix *A, ok_float x) {
  size_t i;
  #ifndef OPTKIT_ORDER
  vector row_col = (vector){0,0,OK_NULL};
  if (A->rowmajor == CblasRowMajor)
    for(i = 0; i < A->size1; ++i){
      matrix_row(&row_col, A, i);
      vector_scale(&row_col, x);
    }
  else{
    for(i = 0; i < A->size2; ++i){
      matrix_column(&row_col, A, i);
      vector_scale(&row_col, x);
    }
  }
  #elif OPTKIT_ORDER == 101
  vector row = (vector){0,0,OK_NULL};
  for(i = 0; i < A->size1; ++i){
    matrix_row(&row, A, i);
    vector_scale(&row, x);
  }
  #else
  vector col = (vector){0,0,OK_NULL};
  for(i = 0; i < A->size2; ++i){
    matrix_column(&col, A, i);
    vector_scale(&col, x);
  }
  #endif
}

void matrix_abs(matrix * A) {
  size_t i;
  #ifndef OPTKIT_ORDER
  vector row_col = (vector){0,0,OK_NULL};
  if (A->rowmajor == CblasRowMajor)
    for(i = 0; i < A->size1; ++i){
      matrix_row(&row_col, A, i);
      vector_abs(&row_col);
    }
  else{
    for(i = 0; i < A->size2; ++i){
      matrix_column(&row_col, A, i);
      vector_abs(&row_col);
    }
  }
  #elif OPTKIT_ORDER == 101
  vector row = (vector){0,0,OK_NULL};
  for(i = 0; i < A->size1; ++i){
    matrix_row(&row, A, i);
    vector_abs(&row);
  }
  #else
  vector col = (vector){0,0,OK_NULL};
  for(i = 0; i < A->size2; ++i){
    matrix_column(&col, A, i);
    vector_abs(&col);
  }
  #endif
}


#ifndef OPTKIT_ORDER
int __matrix_order_compat(const matrix * A, const matrix * B, 
  const char * nm_A, const char * nm_B, const char * nm_routine){

  if (A->rowmajor == B->rowmajor) return 1;
  printf("OPTKIT ERROR (%s) matrices %s and %s must have same layout.\n", 
         nm_routine, nm_A, nm_B);
  return 0;
}
#endif


/* BLAS routines */

inline int __blas_check_handle(void * linalg_handle){
  if (linalg_handle == OK_NULL) return 1;
  else { 
    printf("%s\n",
      "Error: CBLAS operations take no linear algebra handle.\
       Non-void pointer provided for argument \'linalg_handle\'");
    return 0;
  }
}

void blas_make_handle(void ** linalg_handle){
  *linalg_handle = OK_NULL;
}

void blas_destroy_handle(void * linalg_handle){
  linalg_handle = OK_NULL;
}


/* BLAS LEVEL 1 */
void blas_axpy(void * linalg_handle, ok_float alpha, 
                 const vector *x, vector *y) {
  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  CBLAS(axpy)( (int) x->size, alpha, x->data, (int) x->stride, 
               y->data, (int) y->stride);
}

ok_float blas_nrm2(void * linalg_handle, const vector *x) {
  if ( !__blas_check_handle(linalg_handle) ) return NAN;
  return CBLAS(nrm2)((int) x->size, x->data, (int) x->stride);
}

void 
blas_scal(void * linalg_handle, const ok_float alpha, vector *x) {
  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  CBLAS(scal)((int) x->size, alpha, x->data, (int) x->stride);
}

ok_float 
blas_asum(void * linalg_handle, const vector * x) {
  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return NAN;
  #endif
  return CBLAS(asum)((int) x->size, x->data, (int) x->stride);
}

ok_float 
blas_dot(void * linalg_handle, const vector * x, const vector * y) {
  if ( !__blas_check_handle(linalg_handle) ) return NAN;
  return CBLAS(dot)( (int) x->size, x->data, (int) x->stride, 
                     y->data, (int) y->stride);
}

void
blas_dot_inplace(void * linalg_handle, const vector * x, const vector * y,
  ok_float * deviceptr_result){
  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  * deviceptr_result = CBLAS(dot)( (int) x->size, x->data, (int) x->stride, 
                     y->data, (int) y->stride);
}

/* BLAS LEVEL 2 */

void blas_gemv(void * linalg_handle, CBLAS_TRANSPOSE_t transA, 
                ok_float alpha, const matrix *A, 
               const vector *x, ok_float beta, vector *y){

  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  CBLAS(gemv)(A->rowmajor, transA, (int) A->size1, (int) A->size2, 
              alpha, A->data, (int) A->ld, x->data, (int) x->stride, 
              beta, y->data, (int) y->stride);
}

void blas_trsv(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t transA, CBLAS_DIAG_t Diag, 
                 const matrix *A, vector *x){

  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  CBLAS(trsv)(A->rowmajor, Uplo, transA, Diag, (int) A->size1, 
              A->data, (int) A->ld, x->data, (int) x->stride); 
}

/* BLAS LEVEL 3 */
void blas_syrk(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t transA, ok_float alpha, 
                 const matrix * A, ok_float beta, matrix * C) {


  const int k = (transA == CblasNoTrans) ? (int) A->size2 : (int) A->size1;

  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  #ifndef OPTKIT_ORDER
  if ( __matrix_order_compat(A, C, "A", "C", "blas_syrk") )
  #endif
    CBLAS(syrk)(A->rowmajor, Uplo, transA, (int) C->size2 , k, alpha, 
                A->data, (int) A->ld, beta, C->data, (int) C->ld);
  
}

void blas_gemm(void * linalg_handle, CBLAS_TRANSPOSE_t transA, 
                 CBLAS_TRANSPOSE_t transB, ok_float alpha, 
                 const matrix * A, const matrix * B, 
                 ok_float beta, matrix * C){


  const int NA = (transA == CblasNoTrans) ? (int) A->size2 : (int) A->size1; 

  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  #ifndef OPTKIT_ORDER
  if ( __matrix_order_compat(A, B, "A", "B", "gemm") && 
        __matrix_order_compat(A, C, "A", "C", "blas_gemm") )
  #endif
    CBLAS(gemm)(A->rowmajor, transA, transB, (int) C->size1, (int) C->size2, NA, alpha, 
                A->data, (int) A->ld, B->data, (int) B->ld, beta, C->data, (int) C->ld);

}

void blas_trsm(void * linalg_handle, CBLAS_SIDE_t Side, 
                 CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t transA,
                 CBLAS_DIAG_t Diag, ok_float alpha, 
                 const matrix *A, matrix *B) {

  #ifdef OK_DEBUG
  if ( !__blas_check_handle(linalg_handle) ) return;
  #endif
  #ifndef OPTKIT_ORDER
  if ( __matrix_order_compat(A, B, "A", "B", "blas_trsm") )
  #endif
    CBLAS(trsm)(A->rowmajor, Side, Uplo, transA, Diag,(int) B->size1, (int) B->size2, 
                alpha, A->data,(int) A->ld, B->data, (int) B->ld);

}

/* LINEAR ALGEBRA routines */

/* Non-Block Cholesky. */
void __linalg_cholesky_decomp_noblk(void * linalg_handle, matrix *A) {
  ok_float l11;
  matrix l21, a22;
  size_t n = A->size1, i;



  l21= (matrix){0,0,0,OK_NULL,CblasRowMajor};
  a22= (matrix){0,0,0,OK_NULL,CblasRowMajor};

  for (i = 0; i < n; ++i) {
    /* L11 = sqrt(A11) */
    l11 = (ok_float) MATH(sqrt)(__matrix_get(A, i, i));
    __matrix_set(A, i, i, l11);

    if (i + 1 < n) {
      /* L21 = A21 / L11 */
      matrix_submatrix(&l21, A, i + 1, i, n - i - 1, 1);
      matrix_scale(&l21, kOne / l11);
     

      /* A22 -= L12*L12'*/
      matrix_submatrix(&a22, A, i + 1, i + 1, n - i - 1, n - i - 1);
      blas_syrk(linalg_handle, CblasLower, CblasNoTrans, 
                  -kOne, &l21, kOne, &a22);      
    }
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
void linalg_cholesky_decomp(void * linalg_handle, matrix * A) {
  matrix L11, L21, A22;
  size_t n = A->size1, blk_dim, i, n11;

  L11= (matrix){0,0,0,OK_NULL,CblasRowMajor};
  L21= (matrix){0,0,0,OK_NULL,CblasRowMajor};
  A22= (matrix){0,0,0,OK_NULL,CblasRowMajor};

  /* block dimension borrowed from Eigen. */
  blk_dim = ((n / 128) * 16) < 8 ? (n / 128) * 16 : 8;
  blk_dim = blk_dim > 128 ? blk_dim : 128;

  for (i = 0; i < n; i += blk_dim) {
    n11 = blk_dim < n - i ? blk_dim : n - i;

    /* L11 = chol(A11) */
    matrix_submatrix(&L11, A, i, i, n11, n11);
    __linalg_cholesky_decomp_noblk(linalg_handle, &L11);

    if (i + blk_dim < n) {
      /* L21 = A21 L21^-T */
      matrix_submatrix(&L21, A, i + n11, i, n - i - n11, n11);
      blas_trsm(linalg_handle, CblasRight, CblasLower, CblasTrans, 
                  CblasNonUnit, kOne, &L11, &L21);

      /* A22 -= L21*L21^T */
      matrix_submatrix(&A22, A, i + blk_dim, i + blk_dim, 
                         n - i - blk_dim, n - i - blk_dim);
      blas_syrk(linalg_handle, CblasLower, CblasNoTrans, 
                  (ok_float) -kOne, &L21, (ok_float) kOne, &A22);
    }
  }
}


/* Cholesky solve */
void linalg_cholesky_svx(void * linalg_handle, 
                           const matrix * L, vector * x) {
  blas_trsv(linalg_handle, CblasLower, CblasNoTrans, CblasNonUnit, L, x);
  blas_trsv(linalg_handle, CblasLower, CblasTrans, CblasNonUnit, L, x);
}

/* device reset */
int ok_device_reset(){
  return 0;
}



#ifdef __cplusplus
}
#endif
