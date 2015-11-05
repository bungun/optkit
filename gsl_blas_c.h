#ifndef GSL_BLAS_H_GUARD
#define GSL_BLAS_H_GUARD

#include "cblas.h"
#include "gsl_matrix_c.h"
#include "gsl_vector_c.h"

#ifdef __cplusplus
extern "C" {
#endif



#ifndef FLOAT
    #define CUBLAS(x) cublasD ## x
    #define CUSPARSE(x) cusparseD ## x
#else
    #define CUBLAS(x) cublasS ## x
    #define CUSPARSE(x) cusparseS ## x
#endif
#else

// Gnu Scientific Library

//
// BLAS LEVEL 1
//

// Axpy.
void blas_axpy_f(float alpha, const vector_f *x, vector_f *y) {
  cblas_saxpy( (int) x->size, alpha, x->data, (int) x->stride, y->data, (int) y->stride);
}
void blas_axpy_d(double alpha, const vector_d *x, vector_d *y) {
  cblas_daxpy( (int) x->size, alpha, x->data, (int) x->stride, y->data, (int) y->stride);
}

// Nrm2.
float blas_nrm2_f(vector_f *x) {
  return cblas_snrm2((int) x->size, x->data, (int) x->stride);
}
float blas_nrm2_d(vector_d *x) {
  return cblas_dnrm2((int) x->size, x->data, (int) x->stride);
}

// Scal.
void blas_scal_f(const float alpha, vector_f *x) {
  cblas_sscal((int) x->size, alpha, x->data, (int) x->stride)
}
void blas_scal_d(const double alpha, vector_d *x) {
  cblas_dscal((int) x->size, alpha, x->data, (int) x->stride)
}

// Asum.
float blas_asum_f(const vector_f *x) {
  return cblas_sasum((int) x->size, x->data, (int) x->stride);
}

double blas_asum_d(const vector_d *x) {
  return cblas_dasum((int) x->size, x->data, (int) x->stride);
}


// Dot.
void blas_dot_f(const vector_f *x, const vector_f *y, float *result) {
  *result = cblas_sdot((int) x->size, x->data, (int) x->stride, y->date, (int) y->stride);
}

void blas_dot_f(const vector_f *x, const vector_f *y, float *result) {
  *result = cblas_ddot((int) x->size, x->data, (int) x->stride, y->date, (int) y->stride);
}

//
// BLAS LEVEL 2
//

// Gemv.
void blas_gemv_f(CBLAS_TRANSPOSE_t TransA, float alpha, matrix_f *A, const vector_f *x, float beta, vector_f *y){
  if (A->rowmajor)
    cblas_sgemv(CblasRowMajor, TransA, (int) A->size1, (int) A->size2, alpha, A->data, (int) A->tda, x->data, (int) x->stride, beta, y->data, (int) y->stride);
  else
    cblas_sgemv(CblasColMajor, TransA, (int) A->size1, (int) A->size2, alpha, A->data, (int) A->tda, x->data, (int) x->stride, beta, y->data, (int) y->stride);
}

void blas_gemv_f(CBLAS_TRANSPOSE_t TransA, double alpha, matrix_d *A, const vector_d *x, double beta, vector_d *y){
  if (A->rowmajor)
    cblas_dgemv(CblasRowMajor, TransA, (int) A->size1, (int) A->size2, alpha, A->data, (int) A->tda, x->data, (int) x->stride, beta, y->data, (int) y->stride);
  else
    cblas_dgemv(CblasColMajor, TransA, (int) A->size1, (int) A->size2, alpha, A->data, (int) A->tda, x->data, (int) x->stride, beta, y->data, (int) y->stride);
}



// Trsv.
void blas_trsv_f(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, const matrix_f *A, vector_f *x){
  if (A->rowmajor)
     cblas_strsv(CblasRowMajor, Uplo, TransA, Diag, (int) A->size1, A->data, (int) A->tda, x->data, (int) x->stride ); 
  else
     cblas_strsv(CblasColMajor, Uplo, TransA, Diag, (int) A->size1, A->data, (int) A->tda, x->data, (int) x->stride ); 
}

void blas_trsv_d(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA, CBLAS_DIAG_t Diag, const matrix_d *A, vector_d *x){
  if (A->rowmajor)
     cblas_dtrsv(CblasRowMajor, Uplo, TransA, Diag, (int) A->size1, A->data, (int) A->tda, x->data, (int) x->stride ); 
  else
     cblas_dtrsv(CblasColMajor, Uplo, TransA, Diag, (int) A->size1, A->data, (int) A->tda, x->data, (int) x->stride ); 
}

//
// BLAS LEVEL 3
//

// Syrk
void blas_syrk_f(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, float alpha, const matrix_f *A, float beta, matrix_f *C) {
  if (A->rowmajor != C->rowmajor){
    printf("\nGSL BLAS ERROR: blas_syrk_f: matrices A and C must have same layout.\n");
    return;
  }

  const int N = (int) C->size2;
  const int K = (Trans == CblasNoTrans) ? (int) A->size2 : (int) A->size1;
  if (A->rowmajor)
    cblas_ssyrk(CblasRowMajor, Uplo, Trans, N, K, alpha, A->data, (int) A->tda, beta, C->data, (int) C->tda);
  else
    cblas_ssyrk(CblasColMajor, Uplo, Trans, N, K, alpha, A->data, (int) A->tda, beta, C->data, (int) C->tda);
}

void blas_syrk_d(CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t Trans, double alpha, const matrix_d *A, double beta, matrix_d *C) {
  if (A->rowmajor != C->rowmajor){
    printf("\nGSL BLAS ERROR: blas_syrk_d: matrices A and C must have same layout.\n");
    return;
  }

  const int N = (int) C->size2;
  const int K = (Trans == CblasNoTrans) ? (int) A->size2 : (int) A->size1;
  if (A->rowmajor)
    cblas_dsyrk(CblasRowMajor, Uplo, Trans, N, K, alpha, A->data, (int) A->tda, beta, C->data, (int) C->tda);
  else
    cblas_dsyrk(CblasColMajor, Uplo, Trans, N, K, alpha, A->data, (int) A->tda, beta, C->data, (int) C->tda);
}


// Gemm
void blas_gemm_f(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, float alpha, const matrix_f *A, const matrix_f *B, float beta, matrix_f *C){
  if (A->rowmajor != B->rowmajor){
    printf("\nGSL BLAS ERROR: blas_gemm_f: matrices A and B must have same layout.\n");
    return;
  } else if (A->rowmajor != C->rowmajor){
    printf("\nGSL BLAS ERROR: blas_gemm_f: matrices A and C must have same layout.\n");
    return;
  }

  const int M = (int) C->size1;
  const int N = (int) C->size2;
  const int NA = (TransA == CblasNoTrans) ? (int) A->size2 : (int) A->size1; 
  if (A->rowmajor)
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, NA, alpha, A->data, (int) A->tda, B->data, (int) B->tda, beta, C->data, (int) C->tda);
  else
    cblas_sgemm(CblasColMajor, TransA, TransB, M, N, NA, alpha, A->data, (int) A->tda, B->data, (int) B->tda, beta, C->data, (int) C->tda);
}

void blas_gemm_d(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, double alpha, const matrix_d *A, const matrix_d *B, double beta, matrix_d *C){
  if (A->rowmajor != B->rowmajor){
    printf("\nGSL BLAS ERROR: blas_gemm_d: matrices A and B must have same layout.\n");
    return 
  } else if (A->rowmajor != C->rowmajor){
    printf("\nGSL BLAS ERROR: blas_gemm_d: matrices A and C must have same layout.\n");
    return;
  }

  const int M = (int) C->size1;
  const int N = (int) C->size2;
  const int NA = (TransA == CblasNoTrans) ? (int) A->size2 : (int) A->size1; 
  if (A->rowmajor)
    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, NA, alpha, A->data, (int) A->tda, B->data, (int) B->tda, beta, C->data, (int) C->tda);
  else
    cblas_dgemm(CblasColMajor, TransA, TransB, M, N, NA, alpha, A->data, (int) A->tda, B->data, (int) B->tda, beta, C->data, (int) C->tda);
}


// Trsm
void blas_trsm_f(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, float alpha, const matrix_f *A,
               matrix_f *B) {
  if (A->rowmajor != B->rowmajor){
    printf("\nGSL BLAS ERROR: blas_trsm_f: matrices A and B must have same layout.\n");
    return;
  }
  if (A->rowmajor)
    cblas_strsm(CblasRowMajor, Side, Uplo, TransA, Diag,(int) B->size1, (int) B->size2, alpha, A->data,(int) A->tda, B->data, (int) B->tda);
  else
    cblas_strsm(CblasColMajor, Side, Uplo, TransA, Diag,(int) B->size1, (int) B->size2, alpha, A->data,(int) A->tda, B->data, (int) B->tda);

}

void blas_trsm_d(CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, CBLAS_TRANSPOSE_t TransA,
               CBLAS_DIAG_t Diag, double alpha, const matrix_d *A, matrix_d *B) {
  if (A->rowmajor != B->rowmajor){
    printf("\nGSL BLAS ERROR: blas_trsm_d: matrices A and B must have same layout.\n");
    return;
  }
  if (A->rowmajor)
    cblas_dtrsm(CblasRowMajor, Side, Uplo, TransA, Diag,(int) B->size1, (int) B->size2, alpha, A->data,(int) A->tda, B->data, (int) B->tda);
  else
    cblas_dtrsm(CblasColMajor, Side, Uplo, TransA, Diag,(int) B->size1, (int) B->size2, alpha, A->data,(int) A->tda, B->data, (int) B->tda);
}


#ifdef __cplusplus
}
#endif

#endif  // GSL_BLAS_H_GUARD

