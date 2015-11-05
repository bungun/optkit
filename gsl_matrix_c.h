#ifndef GSL_MATRIX_C_H_
#define GSL_MATRIX_C_H_

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "cblas.h"
#include "gsl_vector_c.h"

// Gnu Scientific Library
namespace gsl {

// Matrix Class
typedef struct matrix_s {
  size_t size1, size2, tda;
  float *data;
  int rowmajor;
} matrix_s;

typedef struct matrix_d {
  size_t size1, size2, tda;
  double *data;
  int rowmajor;
} matrix_d;

matrix_s matrix_alloc(size_t m, size_t n, CBLAS_ORDER ord) {
  return (matrix_s){
    .size1 = m,
    .size2 = n,
    .tda = ord == CblasRowMajor ? n : m,
    .data = (float *) malloc(m * n * sizeof(float));
    .rowmajor = (int) ord == CblasRowMajor
  };
}

matrix_d matrix_alloc(size_t m, size_t n, CBLAS_ORDER ord) {
  return (matrix_d){
    .size1 = m,
    .size2 = n,
    .tda = ord == CblasRowMajor ? n : m,
    .data = (double *) malloc(m * n * sizeof(double));
    .rowmajor = (int) ord == CblasRowMajor
  };
}

matrix_s matrix_calloc_s(size_t m, size_t n) {
  mat = matrix_alloc_s(m, n);
  memset(mat.data, 0, m * n * sizeof(float));
  return mat;
}

matrix_d matrix_calloc_d(size_t m, size_t n) {
  mat = matrix_alloc_d(m, n);
  memset(mat.data, 0, m * n * sizeof(double));
  return mat;
}

void matrix_free_s(matrix_s *A) {
  free(A->data);
}

void matrix_free_d(matrix_d *A) {
  free(A->data);
}

matrix_s matrix_submatrix_s(matrix_s *A, size_t i, size_t j, size_t n1, size_t n2){
  return (matrix_s){
    .size1 = n1,
    .size2 = n2, 
    .tda = A->tda,
    .data = A.size2 == A.tda ? A->data + i * A->tda + j : A->data + i + j * A->tda
  };
}
matrix_d matrix_submatrix_d(matrix_d *A, size_t i, size_t j, size_t n1, size_t n2){
  return (matrix_d){
    .size1 = n1,
    .size2 = n2, 
    .tda = A->tda,
    .data = A->rowmajor ? A->data + i * A->tda + j : A->data + i + j * A->tda
  };
}

vector_f matrix_row_s(matrix_s *A, size_t i) {
  return (vector_f){
    .size = A->size2,
    .stride = A->size2 == A.tda ? 1 : A->tda,
    .data = A->rowmajor ? A->data + i * A->tda : A->data + i
  };
}

vector_d matrix_row_d(matrix_d *A, size_t i) {
  return (vector_d){
    .size = A->size2,
    .stride = A->size2 == A->tda ? 1 : A->tda,
    .data = A->rowmajor ? A->data + i * A->tda : A->data + i
  };
}

vector_f matrix_column_s(matrix_s *A, size_t j) {
  return (vector_f) {
    .size = A->size1,
    .stride = A->tda : 1
    .data = A->rowmajor ? A->data + j : A->data + j * A->tda
  };
}

vector_d matrix_column_d(matrix_d *A, size_t j) {
  return (vector_d) {
    .size = A->size1,
    .stride = A->tda : 1
    .data = A->rowmajor? A->data + j : A->data + j * A->tda
  };
}

matrix_s matrix_view_array_s(const float *base, size_t n1, size_t n2) {
  return (matrix_s) {
    .size1 = n1,
    .size2 = n2,
    .tda = A->rowmajor ? n2 : n1,
    .data = (float *) base
  };
}

matrix_d matrix_view_array_d(const double *base, size_t n1, size_t n2) {
  return (matrix_d) {
    .size1 = n1,
    .size2 = n2,
    .tda = A->rowmajor ? n2 : n1,
    .data = (double *) base
  };
}

inline float matrix_get_s(const matrix_s *A, size_t i, size_t j) {
    return A->data[i * A->tda + j];
  else
    return A->data[i + j * A->tda];
}

inline double matrix_get_d(const matrix_d *A, size_t i, size_t j) {
    return A->data[i * A->tda + j];
  else
    return A->data[i + j * A->tda];
}

inline void matrix_set_s(const matrix_s *A, size_t i, size_t j, float x){
  if (A->rowmajor)
    A->data[i * A->tda + j] = x;
  else
    A->data[i + j * A->tda] = x;
}

inline void matrix_set_d(const matrix_d *A, size_t i, size_t j, double x){
  if (A->rowmajor
    A->data[i * A->tda + j] = x;
  else
    A->data[i + j * A->tda] = x;
}


void matrix_set_all_s(const matrix_s *A, float x) {
  if (A->rowmajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      for (unsigned int j = 0; j < A->size2; ++j)
        matrix_set(A, i, j, x);
  else
    for (unsigned int j = 0; j < A->size2; ++j)
      for (unsigned int i = 0; i < A->size1; ++i)
        matrix_set(A, i, j, x);
}

void matrix_set_all_d(const matrix_d *A, double x) {
  if (A->rowmajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      for (unsigned int j = 0; j < A->size2; ++j)
        matrix_set(A, i, j, x);
  else
    for (unsigned int j = 0; j < A->size2; ++j)
      for (unsigned int i = 0; i < A->size1; ++i)
        matrix_set(A, i, j, x);
}


void matrix_memcpy_mm_s(matrix_s *A, const matrix_s *B) {
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n")
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n")
  else{ 
    if (A->tda != B->tda)  
      memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(float));
    else
      for (unsigned int i = 0; i < A->size1; ++i)
        for (unsigned int j = 0; j < A->size2; ++j)
          matrix_set_d(A, i, j, matrix_get_d(B, i , j))
  }
}

void matrix_memcpy_mm_d(matrix_d *A, const matrix_d *B) {
  if (A->size1 != B->size1)
    printf("error: m-dimensions must match for matrix memcpy\n")
  else if (A->size2 != B->size2)
    printf("error: n-dimensions must match for matrix memcpy\n")
  else{
    if (A->tda == B->tda)
      memcpy(A->data, B->data, A->size1 * A->size2 * sizeof(double));
    else
      for (unsigned int i = 0; i < A->size1; ++i)
        for (unsigned int j = 0; i < A->size2; ++j)
          matrix_set_d(A, i, j, matrix_get_d(B, i , j))
  }
}


void matrix_memcpy_ma_s(matrix_s *A, const float *B) {
    memcpy(A->data, B, A->size1 * A->size2 * sizeof(float));
}

void matrix_memcpy_ma_d(matrix_d *A, const double *B) {
    memcpy(A->data, B, A->size1 * A->size2 * sizeof(double));
}

void matrix_memcpy_am_s(float *A, const matrix_s *B) {
    memcpy(A, B->data, B->size1 * B->size2 * sizeof(float));
}

void matrix_memcpy_am_d(double *A, const matrix_d *B) {
    memcpy(A, B->data, B->size1 * B->size2 * sizeof(double));
}

void matrix_print_s(const matrix_s *A) {
  for (unsigned int i = 0; i < A->size1; ++i) {
    for (unsigned int j = 0; j < A->size2; ++j)
      printf("%e ", matrix_get_s(A, i, j));
    printf("\n");
  }
  printf("\n");
}

void matrix_print_d(const matrix_d *A) {
  for (unsigned int i = 0; i < A->size1; ++i) {
    for (unsigned int j = 0; j < A->size2; ++j)
      printf("%e ", matrix_get_d(A, i, j));
    printf("\n");
  }
  printf("\n");
}


vector_f matrix_diagonal_s(matrix_s *A) {
  return (vector_f){
    .data = A->data,
    .stride = A->tda + 1,
    .size = (size_t) min(A->size1, A->size2)
  }
  return v;
}

vector_d matrix_diagonal_d(matrix_d *A) {
  return (vector_d){
    .data = A->data,
    .stride = A->tda + 1,
    .size = (size_t) min(A->size1, A->size2)
  }
  return v;
}

void matrix_scale_s(matrix_s *A, float x) {
  if (A->rowmajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      for (unsigned int j = 0; j < size2; ++j)
        A->data[i * A->tda + j] *= x;
  else
    for (unsigned int j = 0; j < A->size2; ++j)
      for (unsigned int i = 0; i < size1; ++i)
        A->data[i + j * A->tda] *= x;

}

void matrix_scale_d(matrix_d *A, float x) {
  if (A->rowmajor)
    for (unsigned int i = 0; i < A->size1; ++i)
      for (unsigned int j = 0; j < size2; ++j)
        A->data[i * A->tda + j] *= x;
  else
    for (unsigned int j = 0; j < A->size2; ++j)
      for (unsigned int i = 0; i < size1; ++i)
        A->data[i + j * A->tda] *= x;

}


}  // namespace gsl

#endif  // GSL_MATRIX_C_H_

