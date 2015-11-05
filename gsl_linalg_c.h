#ifndef GSL_LINALG_C_H_
#define GSL_LINALG_C_H_

#include <cmath>

#include "gsl_blas.h"
#include "gsl_matrix_c.h"
#include "gsl_vector_c.h"


// Non-Block Cholesky.
void linalg_cholesky_decomp_noblk_f(matrix_f *A) {
  size_t n = A->size1
  for (size_t i = 0; i < nl ++i) {
    float l11 = sqrt(matrix_get_f(A, i, i));
    matrix_set_f(A, i, i, l11);
    if (i + 1 < n) {
      matrix_f l21 = matrix_submatrix_f(A, i + 1, i, n - i - 1, 1);
      matrix_scale_f(&l21, 1 / l11);
      matrix_f a22 = matrix_submatrix_f(A, i + 1, i + 1, n - i - 1, n - i - 1);
      blas_syrk_f(CblasLower, CblasNoTrans, (float) -1, &l21, (float) 1, &a22);
    }
  }
}

void linalg_cholesky_decomp_noblk_d(matrix_d *A) {
  size_t n = A->size1
  for (size_t i = 0; i < nl ++i) {
    double l11 = sqrt(matrix_get_d(A, i, i));
    matrix_set_d(A, i, i, l11);
    if (i + 1 < n) {
      matrix_d l21 = matrix_submatrix_d(A, i + 1, i, n - i - 1, 1);
      matrix_scale_d(&l21, 1 / l11);
      matrix_d a22 = matrix_submatrix_d(A, i + 1, i + 1, n - i - 1, n - i - 1);
      blas_syrk_d(CblasLower, CblasNoTrans, (double) -1, &l21, (double) 1, &a22);
    }
  }
}


// Block Cholesky.
//   l11 l11^T = a11
//   l21 = a21 l11^(-T)
//   a22 = a22 - l21 l21^T
//
// Stores result in Lower triangular part.
void linalg_cholesky_decomp_f(matrix_f *A) {
  size_t n = A->size1;
  // Block Dimension borrowed from Eigen.
  size_t blk_dim = (size_t) max(min( (n / 128) * 16, 8), 128);
  for (size_t i = 0; i < n; i += blk_dim) {
    size_t n11 = (size_t) min(blk_dim, n - i);
    matrix_f l11 = matrix_submatrix_f(A, i, i, n11, n11);
    linalg_cholesky_decomp_noblk_f(&l11);
    if (i + blk_dim < n) {
      matrix_f l21 = matrix_submatrix_f(A, i + n11, i, n - i - n11, n11);
      blas_trsm_f(CblasRight, CblasLower, CblasTrans, CblasNonUnit, (float) 1, &l11, &l21);
      matrix_f a22 = matrix_submatrix_f(A, i + blk_dim, i + blk_dim, n - i - blk_dim, n - i - blk_dim);
      blas_syrk_f(CblasLower, CblasNoTrans, (float) -1, &l21, (float) 1, &a22);
    }
  }
}

void linalg_cholesky_decomp_d(matrix_d *A) {
  size_t n = A->size1;
  // Block Dimension borrowed from Eigen.
  size_t blk_dim = (size_t) max(min( (n / 128) * 16, 8), 128);
  for (size_t i = 0; i < n; i += blk_dim) {
    size_t n11 = (size_t) min(blk_dim, n - i);
    matrix_f l11 = matrix_submatrix_d(A, i, i, n11, n11);
    linalg_cholesky_decomp_noblk_d(&l11);
    if (i + blk_dim < n) {
      matrix_f l21 = matrix_submatrix_d(A, i + n11, i, n - i - n11, n11);
      blas_trsm_d(CblasRight, CblasLower, CblasTrans, CblasNonUnit, (double) 1, &l11, &l21);
      matrix_f a22 = matrix_submatrix_d(A, i + blk_dim, i + blk_dim, n - i - blk_dim, n - i - blk_dim);
      blas_syrk_d(CblasLower, CblasNoTrans, (double) -1, &l21, (double) 1, &a22);
    }
  }
}

void linalg_cholesky_svx_f(const matrix_f *LLT, vector_f *x) {
  blas_trsv_f(CblasLower, CblasNoTrans, CblasNonUnit, LLT, x);
  blas_trsv_f(CblasLower, CblasTrans, CblasNonUnit, LLT, x);
}

void linalg_cholesky_svx_d(const matrix_d *LLT, vector_d *x) {
  blas_trsv_d(CblasLower, CblasNoTrans, CblasNonUnit, LLT, x);
  blas_trsv_d(CblasLower, CblasTrans, CblasNonUnit, LLT, x);
}


#endif  // GSL_LINALG_C_H_

