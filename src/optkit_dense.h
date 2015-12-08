#ifndef OPTKIT_DENSE_H_GUARD
#define OPTKIT_DENSE_H_GUARD

#include "optkit_defs.h"

#ifdef __cplusplus
extern "C" {
#endif


/* VECTOR definition and methods */

typedef struct vector {
	size_t size, stride;
	ok_float * data;
} vector;


void __vector_alloc(vector * v, size_t n);
void __vector_calloc(vector * v, size_t n);
void __vector_free(vector * v);
inline void __vector_set(vector * v, size_t i, ok_float x);
inline ok_float __vector_get(const vector * v, size_t i);
void __vector_set_all(vector * v, ok_float x);
void __vector_subvector(vector * v_out, vector * v_in, size_t offset, size_t n);
void __vector_view_array(vector * v, ok_float * base, size_t n);
void __vector_memcpy_vv(vector * v1, const vector * v2);
void __vector_memcpy_va(vector * v, const ok_float * y);
void __vector_memcpy_av(ok_float * x, const vector * v);
void __vector_print(const vector *v);
void __vector_scale(vector * v, ok_float x);
void __vector_add(vector * v1, const vector * v2);
void __vector_sub(vector * v1, const vector * v2);
void __vector_mul(vector * v1, const vector * v2);
void __vector_div(vector * v1, const vector * v2);
void __vector_add_constant(vector *v, const ok_float x);
void __vector_pow(vector *v, const ok_float x);

/* MATRIX defition and methods */

typedef struct matrix {
  size_t size1, size2, tda;
  ok_float *data;
  CBLAS_ORDER_t rowmajor;
} matrix;


void __matrix_alloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord);
void __matrix_calloc(matrix * A, size_t m, size_t n, CBLAS_ORDER_t ord);
void __matrix_free(matrix * A);
void __matrix_submatrix(matrix * A_sub, matrix * A, size_t i, size_t j, size_t n1, size_t n2);
void __matrix_row(vector * row, matrix * A, size_t i);
void __matrix_column(vector * col, matrix * A, size_t j);
void __matrix_diagonal(vector * diag, matrix * A);
void __matrix_view_array(matrix * A, const ok_float * base, size_t n1, size_t n2, CBLAS_ORDER_t ord);
inline ok_float __matrix_get(const matrix * A, size_t i, size_t j);
inline void __matrix_set(matrix * A, size_t i, size_t j, ok_float x);
void __matrix_set_all(matrix * A, ok_float x);
void __matrix_memcpy_mm(matrix * A, const matrix *B);
void __matrix_memcpy_ma(matrix * A, const ok_float *B, const CBLAS_ORDER_t ord);
void __matrix_memcpy_am(ok_float * A, const matrix *B, const CBLAS_ORDER_t ord);
void __matrix_print(const matrix * A);
void __matrix_scale(matrix * A, ok_float x);

int matrix_order_compat(const matrix * A, const matrix * B, const char * nm_A, 
                 const char * nm_B, const char * nm_routine){

  if (A->rowmajor == B->rowmajor) return 1;
  printf("OPTKIT ERROR (%s) matrices %s and %s must have same layout.\n", 
         nm_routine, nm_A, nm_B);
  return 0;
}


/* BLAS routines */

/* BLAS context */
void __blas_make_handle(void * linalg_handle);
void __blas_destroy_handle(void * linalg_handle);


/* BLAS LEVEL 1 */
void __blas_axpy(void * linalg_handle, ok_float alpha, const vector *x, vector *y);
ok_float __blas_nrm2(void * linalg_handle, const vector *x);
void __blas_scal(void * linalg_handle, const ok_float alpha, vector *x);
ok_float __blas_asum(void * linalg_handle, const vector *x);
ok_float __blas_dot(void * linalg_handle, const vector *x, const vector *y);


/* BLAS LEVEL 2 */
void __blas_gemv(void * linalg_handle, CBLAS_TRANSPOSE_t TransA, 
                 ok_float alpha, const matrix * A, const vector * x, 
                 ok_float beta, vector * y);

void __blas_trsv(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, 
                 const matrix * A, vector * x);

/* BLAS LEVEL 3 */
void __blas_syrk(void * linalg_handle, CBLAS_UPLO_t Uplo, 
                 CBLAS_TRANSPOSE_t Trans, ok_float alpha, 
                 const matrix *A, ok_float beta, matrix *C);

void __blas_gemm(void * linalg_handle, CBLAS_TRANSPOSE_t TransA, 
                 CBLAS_TRANSPOSE_t TransB, ok_float alpha, 
                 const matrix *A, const matrix *B, 
                 ok_float beta, matrix *C);

void __blas_trsm(void * linalg_handle, CBLAS_SIDE_t Side, 
                 CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
                 CBLAS_DIAG_t Diag, ok_float alpha, 
                 const matrix *A, matrix *B);

/* LINEAR ALGEBRA routines */
void __linalg_cholesky_decomp_noblk(void * linalg_handle, matrix * A);
void __linalg_cholesky_decomp(void * linalg_handle, matrix * A);
void __linalg_cholesky_svx(void * linalg_handle, const matrix * L, 
                            vector * x);



#ifdef __cplusplus
}
#endif

#endif  // OPTKIT_DENSE_H_GUARD

