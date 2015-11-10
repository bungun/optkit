#ifndef GSL_H_GUARD
#define GSL_H_GUARD

#include "gsl_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* VECTOR definition and methods */

typedef struct vector {
	size_t size, stride;
	gsl_float * data;
} vector;


void __vector_alloc(vector * v, size_t n);
void __vector_calloc(vector * v, size_t n);
void __vector_free(vector * v);
inline void __vector_set(vector * v, size_t i, gsl_float x);
inline gsl_float __vector_get(const vector * v, size_t i);
void __vector_set_all(vector * v, gsl_float x);
void __vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n);
void __vector_view_array(vector * v, gsl_float * base, size_t n);
void __vector_memcpy_vv(vector * v1, const vector * v2);
void __vector_memcpy_va(vector * v, const gsl_float * y);
void __vector_memcpy_av(gsl_float * x, const vector * v);
void __vector_print(const vector *v);
void __vector_scale(vector * v, gsl_float x);
void __vector_add(vector * v1, const vector * v2);
void __vector_sub(vector * v1, const vector * v2);
void __vector_mul(vector * v1, const vector * v2);
void __vector_div(vector * v1, const vector * v2);
void __vector_add_constant(vector *v, const gsl_float x);

void vector_alloc(void * vec, size_t n);
void vector_calloc(void * vec, size_t n);
void vector_free(void * vec);
inline void vector_set(void * vec, size_t i, gsl_float x);
inline gsl_float vector_get(const void * vec, size_t i);
void vector_set_all(void * vec, gsl_float x);
void vector_subvector(void * vec_out, void * vec_in, size_t offset, size_t n);
void vector_view_array(void * vec_out, gsl_float * base, size_t n);
void vector_memcpy_vv(void * vec1, const void * vec2);
void vector_memcpy_va(void * vec, const gsl_float * y);
void vector_memcpy_av(gsl_float * x, const void * vec);
void vector_print(const void *vec);
void vector_scale(void * vec, gsl_float x);
void vector_add(void * vec1, const void * vec2);
void vector_sub(void * vec1, const void * vec2);
void vector_mul(void * vec1, const void * vec2);
void vector_div(void * vec1, const void * vec2);
void vector_add_constant(void *vec, const gsl_float x);

/* MATRIX defition and methods */

typedef struct matrix {
  size_t size1, size2, tda;
  gsl_float *data;
  CBLAS_ORDER_t rowmajor;
} matrix;


matrix * __matrix_alloc(size_t m, size_t n, CBLAS_ORDER_t ord);
matrix * __matrix_calloc(size_t m, size_t n, CBLAS_ORDER_t ord);
void __matrix_free(matrix * A);
matrix * __matrix_submatrix(matrix * A, size_t i, size_t j, size_t n1, size_t n2);
vector * __matrix_row(matrix * A, size_t i);
vector * __matrix_column(matrix * A, size_t j);
vector * __matrix_diagonal(matrix * A);
matrix * __matrix_view_array(const gsl_float * base, size_t n1, size_t n2, CBLAS_ORDER_t ord);
inline gsl_float __matrix_get(const matrix * A, size_t i, size_t j);
inline void __matrix_set(matrix * A, size_t i, size_t j, gsl_float x);
void __matrix_set_all(matrix * A, gsl_float x);
void __matrix_memcpy_mm(matrix * A, const matrix *B);
void __matrix_memcpy_ma(matrix * A, const gsl_float *B);
void __matrix_memcpy_am(gsl_float * A, const matrix *B);
void __matrix_print(const matrix * A);
void __matrix_scale(matrix * A, gsl_float x);

int matrix_order_compat(const matrix * A, const matrix * B, const char * nm_A, 
                 const char * nm_B, const char * nm_routine){

  if (A->rowmajor == B->rowmajor) return 1;
  printf("GSL ERROR (%s) matrices %s and %s must have same row layout.\n", 
         nm_routine, nm_A, nm_B);
  return 0;
}

void * matrix_alloc(size_t m, size_t n, CBLAS_ORDER_t ord);
void * matrix_calloc(size_t m, size_t n, CBLAS_ORDER_t ord);
void matrix_free(void * A);
void * matrix_submatrix(void * A, size_t i, size_t j, size_t n1, size_t n2);
void * matrix_row(void * A, size_t i);
void * matrix_column(void * A, size_t j);
void * matrix_diagonal(void * A);
void * matrix_view_array(const gsl_float * base, size_t n1, size_t n2, CBLAS_ORDER_t ord);
inline gsl_float matrix_get(const void * A, size_t i, size_t j);
inline void matrix_set(void * A, size_t i, size_t j, gsl_float x);
void matrix_set_all(void * A, gsl_float x);
void matrix_memcpy_mm(void * A, const void * B);
void matrix_memcpy_ma(void * A, const gsl_float * B);
void matrix_memcpy_am(gsl_float * A, const void * B);
void matrix_print(const void * A);
void matrix_scale(void * A, gsl_float x);


/* BLAS routines */

/* BLAS LEVEL 1 */
void __blas_axpy(gsl_float alpha, const vector *x, vector *y);
gsl_float __blas_nrm2(vector *x);
void __blas_scal(const gsl_float alpha, vector *x);
gsl_float __blas_asum(const vector *x);
void __blas_dot(const vector *x, const vector *y, gsl_float *result);

void blas_axpy(gsl_float alpha, const void * x, void * y);
gsl_float blas_nrm2(void * x);
void blas_scal(const gsl_float alpha, void * x);
gsl_float blas_asum(const void * x);
void blas_dot(const void * x, const void * y, gsl_float * result);

/* BLAS LEVEL 2 */
void __blas_gemv(CBLAS_TRANSPOSE_t TransA, gsl_float alpha, const matrix * A, 
               const vector * x, gsl_float beta, vector * y);
void __blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, 
               CBLAS_DIAG_t Diag, const matrix * A, vector * x);

void blas_gemv(CBLAS_TRANSPOSE_t TransA, gsl_float alpha, const void * A, 
               const void * x, gsl_float beta, void * y);
void blas_trsv(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, 
               CBLAS_DIAG_t Diag, const void * A,  void * x);

/* BLAS LEVEL 3 */
void __blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, gsl_float alpha, 
               const matrix *A, gsl_float beta, matrix *C);

void __blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, gsl_float alpha, 
               const matrix *A, const matrix *B, gsl_float beta, matrix *C);

void __blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, gsl_float alpha, const matrix *A, matrix *B);


void blas_syrk(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, gsl_float alpha, 
               const void *A, gsl_float beta, void *C);

void blas_gemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, gsl_float alpha, 
               const void *A, const void *B, gsl_float beta, void *C);

void blas_trsm(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, gsl_float alpha, const void *A, void *B);


/* LINEAR ALGEBRA routines */
void __linalg_cholesky_decomp_noblk(matrix * A);
void __linalg_cholesky_decomp(matrix * A);
void __linalg_cholesky_svx(const matrix * LLT, vector * x);

void linalg_cholesky_decomp_noblk(void * A);
void linalg_cholesky_decomp(void * A);
void linalg_cholesky_svx(const void * LLT, void * x);



#ifdef __cplusplus
}
#endif

#endif  // GSL_H_GUARD

