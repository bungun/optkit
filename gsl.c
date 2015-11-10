#include "gsl.h"

#ifdef __cplusplus
extern "C" {
#endif




/* VECTOR methods */
void __vector_clear(vector * v) {
  if (v == GSL_NULL) v = &(vector){0,0,0};
  if (v->data != GSL_NULL) __vector_free(v);
}

void __vector_alloc(vector * v, size_t n) {
  __vector_clear(v);
  v->size=n;
  v->stride=1;
  v->data=(gsl_float *) malloc(n * sizeof(gsl_float));
}

void vector_alloc(void * vec, size_t n) {
  __vector_alloc( (vector *) vec, n);
}

void __vector_calloc(vector * v, size_t n) {
  vector_alloc(v, n);
  memset(v->data, 0, n * sizeof(gsl_float));
}

void vector_calloc(void * vec, size_t n) {
  __vector_calloc( (vector *) vec, n);
}

void __vector_free(vector * v) {
  gsl_free(v->data);
}

void vector_free(void * vec) {
  printf("Pointer v: %p\n", vec);
  __vector_free( (vector *) vec);
}

inline void __vector_set(vector * v, size_t i, gsl_float x) {
  v->data[i * v->stride] = x;
}

inline void vector_set(void * vec, size_t i, gsl_float x) {
  __vector_set( (vector *) vec, i, x);
}

gsl_float __vector_get(const vector *v, size_t i) {
  return v->data[i * v->stride];
}

gsl_float vector_get(const void *vec, size_t i) {
  return __vector_get( (const vector *) vec, i);
}

void __vector_set_all(vector * v, gsl_float x) {
  for (unsigned int i = 0; i < v->size; ++i)
    vector_set(v, i, x);
}

void vector_set_all(void * vec, gsl_float x) {
  __vector_set_all( (vector *) vec, x);
}

void __vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n) {
  __vector_clear(v_out);
  v_out->size=n;
  v_out->stride=v_in->stride;
  v_out->data=v_in->data + offset * v_in->stride;
}

void vector_subvector(void * vec_out, void * vec_in, size_t offset, size_t n) {
  __vector_subvector( (vector *) vec_out, (vector *) vec_in, offset, n);
}

void __vector_view_array(vector * v, gsl_float * base, size_t n) {
  __vector_clear(v);
	v->size=n;
  v->stride=1;
  v->data=base;
}

void vector_view_array(void * vec, gsl_float * base, size_t n) {
  __vector_view_array( (vector *) vec, base, n);
}

void __vector_memcpy_vv(vector * v1, const vector * v2) {
  if ( v1->stride == 1 && v2->stride == 1) {
    memcpy(v1->data, v2->data, v1->size * sizeof(gsl_float));
  } else {
    for (unsigned int i = 0; i < v1->size; ++i)
      vector_set(v1, i, vector_get(v2,i));
  }
}


void vector_memcpy_vv(void * vec1, const void * vec2) {
  __vector_memcpy_vv( (vector *) vec1, (const vector *) vec2);
}


void __vector_memcpy_va(vector * v, const gsl_float *y) {
  if (v->stride == 1) {
		memcpy(v->data, y, v->size * sizeof(gsl_float));
	} else {
		for (unsigned int i = 0; i < v->size; ++i)
			vector_set(v, i, y[i]);
	}
}

void vector_memcpy_va(void *vec, const gsl_float *y) {
  __vector_memcpy_va( (vector *) vec, y);
}

void __vector_memcpy_av(gsl_float *x, const vector *v) {
	if (v->stride ==1) {
		memcpy(x, v->data, v->size * sizeof(gsl_float));
	} else {
		for (unsigned int i = 0; i < v->size; ++i)
			x[i] = vector_get(v,i);
	}
}

void vector_memcpy_av(gsl_float *x, const void *vec) {
  __vector_memcpy_av( x, (const vector *)  vec);
}


void __vector_print(const vector * v) {
	for (unsigned int i = 0; i < v->size; ++i)
		printf("%e ", vector_get(v, i));
	printf("\n");
}

void vector_print(const void * vec) {
  __vector_print( (const vector *) vec);
}

void __vector_scale(vector * v, gsl_float x) {
	CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
}

void vector_scale(void * vec, gsl_float x) {
  __vector_scale( (vector *) vec, x);
}

void __vector_add(vector * v1, const vector * v2) {
	for (unsigned int i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] += v2->data[i * v2->stride];
}

void vector_add(void * vec1, const void * vec2) {
  __vector_add( (void *) vec1, (const void *) vec2);
}

void __vector_sub(vector * v1, const vector * v2) {
	for (unsigned int i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] -= v2->data[i * v2->stride];
}

void vector_sub(void * vec1, const void * vec2) {
  __vector_sub( (void *) vec1, (const void *) vec2);
}

void __vector_mul(vector * v1, const vector * v2) {
	for (unsigned int i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] *= v2->data[i * v2->stride];
}

void vector_mul(void * vec1, const void * vec2) {
  __vector_mul( (void *) vec1, (const void *) vec2);
}

void __vector_div(vector * v1, const vector * v2) {
	for (unsigned int i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] /= v2->data[i * v2->stride];
}

void vector_div(void * vec1, const void * vec2) {
  __vector_div( (void *) vec1, (const void *) vec2);
}

void __vector_add_constant(vector * v, const gsl_float x) {
  for (unsigned int i = 0; i < v->size; ++i)
    v->data[i * v->stride] += x;
}

void vector_add_constant(void * vec, const gsl_float x) {
  __vector_add_constant( (vector *) vec, x);
}


/* MATRIX methods */

matrix * __matrix_alloc(size_t m, size_t n, CBLAS_ORDER_t ord) {
  return &(matrix){
    .size1 = m,
    .size2 = n,
    .tda = ord == CblasRowMajor ? n : m,
    .data = (gsl_float *) malloc(m * n * sizeof(gsl_float)),
    .rowmajor = ord
  };
}

void * matrix_alloc(size_t m, size_t n, CBLAS_ORDER_t ord) {
  return (void *) __matrix_alloc(m, n, ord);
}

matrix * __matrix_calloc(size_t m, size_t n, CBLAS_ORDER_t ord) {
  matrix * mat = matrix_alloc(m, n, ord);
  memset(mat->data, 0, m * n * sizeof(gsl_float));
  return mat;
}

void * matrix_calloc(size_t m, size_t n, CBLAS_ORDER_t ord) {
  return (void *) __matrix_calloc(m, n, ord);
}

void __matrix_free(matrix * A) {
  gsl_free(A->data);
}

void matrix_free(void * A) {
  __matrix_free( (matrix *) A);
}

matrix * __matrix_submatrix(matrix * A, size_t i, size_t j, size_t n1, size_t n2){
  return &(matrix){
    .size1 = n1,
    .size2 = n2, 
    .tda = A->tda,
    .data = (A->rowmajor == CblasRowMajor) ? A->data + i * A->tda + j : A->data + i + j * A->tda,
    .rowmajor = A->rowmajor
  };
}

void * matrix_submatrix(void * A, size_t i, size_t j, size_t n1, size_t n2){
  return (void *) __matrix_submatrix( (matrix *) A, i, j, n1, n2);
}

vector * __matrix_row(matrix * A, size_t i) {
  return &(vector){
    .size = A->size2,
    .stride = (A->rowmajor == CblasRowMajor) ? 1 : A->tda,
    .data = (A->rowmajor == CblasRowMajor) ? A->data + (i * A->tda) : A->data + i
  };
}

void * matrix_row(void * A, size_t i) {
  return (void *) __matrix_row( (matrix *) A, i);
}

vector * __matrix_column(matrix *A, size_t j) {
  return &(vector) {
    .size = A->size1,
    .stride = (A->rowmajor == CblasRowMajor) ? A->tda : 1, 
    .data = (A->rowmajor == CblasRowMajor) ? A->data + j : A->data + (j * A->tda) 
  };
}

void * matrix_column(void * A, size_t j) {
  return (void *) __matrix_column( (matrix *) A, j);
}

vector * __matrix_diagonal(matrix *A) {
  return &(vector){
    .data = A->data,
    .stride = A->tda + 1,
    .size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2
  };
}

void * matrix_diagonal(void * A) {
  return (void *) __matrix_diagonal( (matrix *) A);
}

matrix * __matrix_view_array(const gsl_float *base, size_t n1, size_t n2, CBLAS_ORDER_t ord) {
  return &(matrix) {
    .size1 = n1,
    .size2 = n2,
    .tda = (ord == CblasRowMajor) ? n2 : n1,
    .data = (gsl_float *) base,
    .rowmajor = ord
  };
}

void * matrix_view_array(const gsl_float *base, size_t n1, size_t n2, CBLAS_ORDER_t ord) {
  return (void *) __matrix_view_array(base,  n1, n2, ord);
}


inline gsl_float __matrix_get(const matrix * A, size_t i, size_t j) {
  if (A->rowmajor == CblasRowMajor)
    return A->data[i * A->tda + j];
  else
    return A->data[i + j * A->tda];
}


inline gsl_float matrix_get(const void * A, size_t i, size_t j) {
  return __matrix_get( (const matrix *) A, i, j);
}


inline void __matrix_set(matrix *A, size_t i, size_t j, gsl_float x){
  if (A->rowmajor == CblasRowMajor)
    A->data[i * A->tda + j] = x;
  else
    A->data[i + j * A->tda] = x;
}

inline void matrix_set(void *A, size_t i, size_t j, gsl_float x){
  __matrix_set( (matrix *) A, i, j, x);
}

void __matrix_set_all(matrix * A, gsl_float x) {
  memset(A->data, x, A->size1 * A->size2 * sizeof(gsl_float)); 
}

void matrix_set_all(void * A, gsl_float x) {
  __matrix_set_all( (matrix *) A, x);
}


void __matrix_memcpy_mm(matrix * A, const matrix * B) {
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n");
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n");
  else{ 
    if (A->rowmajor == B->rowmajor)  
      memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(gsl_float));
    else
      for (unsigned int i = 0; i < A->size1; ++i)
        for (unsigned int j = 0; j < A->size2; ++j)
          matrix_set(A, i, j, matrix_get(B, i , j));
  }
}

void matrix_memcpy_mm(void * A, const void * B) {
  __matrix_memcpy_mm( (matrix *) A, (const matrix *) B);
}

void __matrix_memcpy_ma(matrix * A, const gsl_float * B) {
    memcpy(A->data, B, A->size1 * A->size2 * sizeof(gsl_float));
}

void matrix_memcpy_ma(void * A, const gsl_float * B) {
  __matrix_memcpy_ma( (matrix *) A, B);
}

void __matrix_memcpy_am(gsl_float * A, const matrix * B) {
    memcpy(A, B->data, B->size1 * B->size2 * sizeof(gsl_float));
}

void matrix_memcpy_am(gsl_float * A, const void * B) {
    __matrix_memcpy_am( A, (matrix *) B);
}


void __matrix_print(const matrix * A) {
  for (unsigned int i = 0; i < A->size1; ++i) {
    for (unsigned int j = 0; j < A->size2; ++j)
      printf("%e ", matrix_get(A, i, j));
    printf("\n");
  }
  printf("\n");
}

void matrix_print(const void *A) {
  __matrix_print( (matrix *) A);
}


void __matrix_scale(matrix *A, gsl_float x) {
  CBLAS(scal)( (int) (A->size1 * A->size2), x, A->data, 1);
}

void matrix_scale(void *A, gsl_float x) {
  __matrix_scale( (matrix *) A, x);
}


/* BLAS routines */

/* BLAS LEVEL 1 */



void __blas_axpy(gsl_float alpha, const vector *x, vector *y) {
  CBLAS(axpy)( (int) x->size, alpha, x->data, (int) x->stride, y->data, (int) y->stride);
}

void blas_axpy(gsl_float alpha, const void * vec_x, void * vec_y) {
  __blas_axpy(alpha, (const vector *) vec_x, (vector *) vec_y);
}

gsl_float __blas_nrm2(vector *x) {
  return CBLAS(nrm2)((int) x->size, x->data, (int) x->stride);
}

gsl_float blas_nrm2(void *vec) {
  return __blas_nrm2( (vector *) vec);
}

void __blas_scal(const gsl_float alpha, vector *x) {
  CBLAS(scal)((int) x->size, alpha, x->data, (int) x->stride);
}

void blas_scal(const gsl_float alpha, void * vec) {
  __blas_scal( alpha, (vector *) vec);
}

gsl_float __blas_asum(const vector * x) {
  return CBLAS(asum)((int) x->size, x->data, (int) x->stride);
}

gsl_float blas_asum(const void * vec) {
  return __blas_asum( (const vector *) vec);
}

void __blas_dot(const vector * x, const vector * y, gsl_float *result) {
  *result = CBLAS(dot)((int) x->size, x->data, (int) x->stride, y->data, (int) y->stride);
}
void blas_dot(const void * vec1, const void * vec2, gsl_float * result) {
  __blas_dot( (const vector *) vec1, (const vector *) vec2, result);
}

/* BLAS LEVEL 2 */

void __blas_gemv(CBLAS_TRANSPOSE_t TransA, gsl_float alpha, const matrix *A, 
               const vector *x, gsl_float beta, vector *y){

    CBLAS(gemv)(A->rowmajor, TransA, (int) A->size1, (int) A->size2, alpha, A->data, 
                (int) A->tda, x->data, (int) x->stride, beta, y->data, (int) y->stride);
}

void blas_gemv(CBLAS_TRANSPOSE_t TransA, gsl_float alpha, const void *A, 
               const void *x, gsl_float beta, void *y){

  __blas_gemv(TransA, alpha, (const matrix *) A, (const vector *) x, beta, (vector *) y);
}


void __blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, 
               CBLAS_DIAG_t Diag, const matrix *A, vector *x){

     CBLAS(trsv)(A->rowmajor, Uplo, TransA, Diag, (int) A->size1, 
                 A->data, (int) A->tda, x->data, (int) x->stride); 
}
void blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, 
               CBLAS_DIAG_t Diag, const void *A, void *x){

  __blas_trsv(Uplo, TransA, Diag, (const matrix *) A, (vector *) x);
}



/* BLAS LEVEL 3 */

void __blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, gsl_float alpha, 
               const matrix * A, gsl_float beta, matrix * C) {

  if ( matrix_order_compat(A, C, "A", "C", "blas_syrk") ){
    const int K = (Trans == CblasNoTrans) ? (int) A->size2 : (int) A->size1;
    CBLAS(syrk)(A->rowmajor, Uplo, Trans, (int) C->size2 , K, alpha, 
                A->data, (int) A->tda, beta, C->data, (int) C->tda);
  }
}

void blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, gsl_float alpha, 
               const void * A, gsl_float beta, void * C) {
  __blas_syrk(Uplo, Trans, alpha, (const matrix *) A, beta, (matrix *) C);
}

void __blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, gsl_float alpha, 
               const matrix * A, const matrix * B, gsl_float beta, matrix * C){

  if ( matrix_order_compat(A, B, "A", "B", "gemm") && 
        matrix_order_compat(A, C, "A", "C", "blas_gemm") ){

    const int NA = (TransA == CblasNoTrans) ? (int) A->size2 : (int) A->size1; 
    CBLAS(gemm)(CblasRowMajor, TransA, TransB, (int) C->size1, (int) C->size2, NA, alpha, 
                A->data, (int) A->tda, B->data, (int) B->tda, beta, C->data, (int) C->tda);
  }
}


void blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, gsl_float alpha, 
               const void *A, const void *B, gsl_float beta, void *C){
  __blas_gemm(TransA, TransB, alpha, (const matrix *) A, (const matrix *) B, beta, (matrix *) C);
}


void __blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, gsl_float alpha, const matrix *A, matrix *B) {

  if ( matrix_order_compat(A, B, "A", "B", "blas_trsm") )
    CBLAS(trsm)(A->rowmajor, Side, Uplo, TransA, Diag,(int) B->size1, (int) B->size2, 
                alpha, A->data,(int) A->tda, B->data, (int) B->tda);
}

void blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, gsl_float alpha, const void *A, void *B) {

  __blas_trsm(Side, Uplo, TransA, Diag, alpha, (const matrix *) A, (matrix *) B);
}

/* LINEAR ALGEBRA routines */

/* Non-Block Cholesky. */
void __linalg_cholesky_decomp_noblk(matrix *A) {
  size_t n = A->size1;
  for (size_t i = 0; i < n; ++i) {
    gsl_float l11 = (gsl_float) sqrt(matrix_get(A, i, i));
    matrix_set(A, i, i, l11);
    if (i + 1 < n) {
      matrix * l21 = matrix_submatrix(A, i + 1, i, n - i - 1, 1);
      matrix_scale(l21, 1 / l11);
      matrix * a22 = matrix_submatrix(A, i + 1, i + 1, n - i - 1, n - i - 1);
      blas_syrk(CblasLower, CblasNoTrans, (gsl_float) -1, l21, (gsl_float) 1, a22);
    }
  }
}

void linalg_cholesky_decomp_noblk(void *A) {
  __linalg_cholesky_decomp_noblk( (matrix *) A);
}

/*
// Block Cholesky.
//   l11 l11^T = a11
//   l21 = a21 l11^(-T)
//   a22 = a22 - l21 l21^T
//
// Stores result in Lower triangular part.
*/
void __linalg_cholesky_decomp(matrix * A) {
  size_t n = A->size1;
  // Block Dimension borrowed from Eigen.
  size_t blk_dim = (size_t) fmax(fmin( (n / 128) * 16, 8), 128);
  for (size_t i = 0; i < n; i += blk_dim) {
    size_t n11 = (size_t) fmin(blk_dim, n - i);
    matrix * l11 = matrix_submatrix(A, i, i, n11, n11);
    linalg_cholesky_decomp_noblk(l11);
    if (i + blk_dim < n) {
      matrix * l21 = matrix_submatrix(A, i + n11, i, n - i - n11, n11);
      blas_trsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, (gsl_float) 1, l11, l21);
      matrix * a22 = matrix_submatrix(A, i + blk_dim, i + blk_dim, n - i - blk_dim, n - i - blk_dim);
      blas_syrk(CblasLower, CblasNoTrans, (gsl_float) -1, l21, (gsl_float) 1, a22);
    }
  }
}

void linalg_cholesky_decomp(void * A) {
  __linalg_cholesky_decomp( (matrix *) A);
}


/* Cholesky solve */
void __linalg_cholesky_svx(const matrix * LLT, vector * x) {
  blas_trsv(CblasLower, CblasNoTrans, CblasNonUnit, LLT, x);
  blas_trsv(CblasLower, CblasTrans, CblasNonUnit, LLT, x);
}

void linalg_cholesky_svx(const void  * LLT, void * x) {
  __linalg_cholesky_svx( (const matrix *) LLT, (vector *) x);
}



#ifdef __cplusplus
}
#endif