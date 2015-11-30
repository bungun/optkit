#include "optkit_dense.h"
#include "gsl_cblas.h"

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
	uint i;
 	for (i = 0; i < v->size; ++i)
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
	uint i;
 	if ( v1->stride == 1 && v2->stride == 1) {
		memcpy(v1->data, v2->data, v1->size * sizeof(ok_float));
	} else {
  		for (i = 0; i < v1->size; ++i)
      			__vector_set(v1, i, __vector_get(v2,i));
  	}
}


void __vector_memcpy_va(vector * v, const ok_float *y) {
	uint i;
	if (v->stride == 1) {
		memcpy(v->data, y, v->size * sizeof(ok_float));
	} else {
		for (i = 0; i < v->size; ++i)
			__vector_set(v, i, y[i]);
	}
}

void __vector_memcpy_av(ok_float *x, const vector *v) {
	uint i;
	if (v->stride ==1) {
		memcpy(x, v->data, v->size * sizeof(ok_float));
	} else {
		for (i = 0; i < v->size; ++i)
			x[i] = __vector_get(v,i);
	}
}


void __vector_print(const vector * v) {
	uint i;
	for (i = 0; i < v->size; ++i)
		printf("%e ", __vector_get(v, i));
	printf("\n");
}

void __vector_scale(vector * v, ok_float x) {
	CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
}

void __vector_add(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] += v2->data[i * v2->stride];
}

void __vector_sub(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] -= v2->data[i * v2->stride];
}

void __vector_mul(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] *= v2->data[i * v2->stride];
}

void __vector_div(vector * v1, const vector * v2) {
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] /= v2->data[i * v2->stride];
}

void __vector_add_constant(vector * v, const ok_float x) {
  uint i;
  for (i = 0; i < v->size; ++i)
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
  uint i,j;
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n");
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n");
  else{ 
    if (A->rowmajor == B->rowmajor)  
      memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(ok_float));
    else
      for (i = 0; i < A->size1; ++i)
        for (j = 0; j < A->size2; ++j)
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
  uint i,j;
  for (i = 0; i < A->size1; ++i) {
    for (j = 0; j < A->size2; ++j)
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

inline int __blas_check_handle(void * linalg_handle){
  if (linalg_handle == OK_NULL) return 1;
  else { 
    // printf("%s\n","Error: CBLAS operations take no linear algebra handle. 
            // Non-void pointer provided for argument \'linalg_handle\'");
    return 0;
  }
}

void __blas_make_handle(void * linalg_handle){
  linalg_handle = OK_NULL;
}

void __blas_destroy_handle(void * linalg_handle){
  linalg_handle = OK_NULL;
}


/* BLAS LEVEL 1 */
void __blas_axpy(void * linalg_handle, ok_float alpha, 
                 const vector *x, vector *y) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CBLAS(axpy)( (int) x->size, alpha, x->data, (int) x->stride, 
               y->data, (int) y->stride);
}

ok_float __blas_nrm2(void * linalg_handle, const vector *x) {
  if ( !__blas_check_handle(linalg_handle) ) return NAN;
  return CBLAS(nrm2)((int) x->size, x->data, (int) x->stride);
}

void __blas_scal(void * linalg_handle, const ok_float alpha, vector *x) {
  if ( !__blas_check_handle(linalg_handle) ) return;
  CBLAS(scal)((int) x->size, alpha, x->data, (int) x->stride);
}

ok_float __blas_asum(void * linalg_handle, const vector * x) {
  if ( !__blas_check_handle(linalg_handle) ) return NAN;
  return CBLAS(asum)((int) x->size, x->data, (int) x->stride);
}

ok_float __blas_dot(void * linalg_handle, 
                    const vector * x, const vector * y) {
  if ( !__blas_check_handle(linalg_handle) ) return NAN;
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

/* Non-Block Cholesky. */
void __linalg_cholesky_decomp_noblk(void * linalg_handle, matrix *A) {
  ok_float l11;
  matrix * l21, * a22;
  size_t n = A->size1, i;

  l21= &(matrix){0,0,0,OK_NULL,CblasRowMajor};
  a22= &(matrix){0,0,0,OK_NULL,CblasRowMajor};

  for (i = 0; i < n; ++i) {
    /* L11 = sqrt(A11) */
    l11 = (ok_float) sqrt(__matrix_get(A, i, i));
    __matrix_set(A, i, i, l11);

    if (i + 1 < n) {
      /* L21 = A21 / L11 */
      __matrix_submatrix(l21, A, i + 1, i, n - i - 1, 1);
      __matrix_scale(l21, (ok_float) 1. / l11);
     
      /* A22 -= L12*L12'*/
      __matrix_submatrix(a22, A, i + 1, i + 1, n - i - 1, n - i - 1);
      __blas_syrk(linalg_handle, CblasLower, CblasNoTrans, 
                  (ok_float) -1, l21, (ok_float) 1, a22);
      
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
void __linalg_cholesky_decomp(void * linalg_handle, matrix * A) {
  matrix * L11, * L21, * A22;
  size_t n = A->size1, blk_dim, i, n11;


  L11= &(matrix){0,0,0,OK_NULL,CblasRowMajor};
  L21= &(matrix){0,0,0,OK_NULL,CblasRowMajor};
  A22= &(matrix){0,0,0,OK_NULL,CblasRowMajor};

  /* block dimension borrowed from Eigen. */
  blk_dim = (size_t) fmax(fmin( (n / 128) * 16, 8), 128);
  for (i = 0; i < n; i += blk_dim) {
    n11 = (size_t) fmin(blk_dim, n - i);

    /* L11 = chol(A11) */
    __matrix_submatrix(L11, A, i, i, n11, n11);
    __linalg_cholesky_decomp_noblk(linalg_handle, L11);

    if (i + blk_dim < n) {
      /* L21 = A21 L21^-T */
      __matrix_submatrix(L21, A, i + n11, i, n - i - n11, n11);
      __blas_trsm(linalg_handle, CblasRight, CblasLower, CblasTrans, 
                  CblasNonUnit, (ok_float) 1, L11, L21);

      /* A22 -= L21*L21^T */
      __matrix_submatrix(A22, A, i + blk_dim, i + blk_dim, 
                         n - i - blk_dim, n - i - blk_dim);
      __blas_syrk(linalg_handle, CblasLower, CblasNoTrans, 
                  (ok_float) -1, L21, (ok_float) 1, A22);
    }
  }
}


/* Cholesky solve */
void __linalg_cholesky_svx(void * linalg_handle, 
                           const matrix * L, vector * x) {
  __blas_trsv(linalg_handle, CblasLower, CblasNoTrans, CblasNonUnit, L, x);
  __blas_trsv(linalg_handle, CblasLower, CblasTrans, CblasNonUnit, L, x);
}



#ifdef __cplusplus
}
#endif
