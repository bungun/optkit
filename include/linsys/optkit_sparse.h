#ifndef OPTKIT_LINSYS_SPARSE_H_
#define OPTKIT_LINSYS_SPARSE_H_

#include "optkit_vector.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef OK_CHECK_SPARSEMAT
#define OK_CHECK_SPARSEMAT(M) \
	do { \
		if (!M || !M->val || !M->ind || !M->ptr) \
			return OK_SCAN_ERR( OPTKIT_ERROR_UNALLOCATED ); \
	} while(0)
#endif

/* transpose data from forward->adjoint or adjoint->forward */
typedef enum SparseTransposeDirection {
	Forward2Adjoint,
	Adjoint2Forward
} SPARSE_TRANSPOSE_DIRECTION;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
template<typename T, typename I>
struct sp_matrix_ {
public:
	size_t size1, size2, nnz, ptrlen;
	T *val;
	I *ind, *ptr;
	enum CBLAS_ORDER order;
};

template<typename T, typename I>
ok_status sp_matrix_alloc_(sp_matrix_<T, I> *A, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order);
template<typename T, typename I>
ok_status sp_matrix_calloc_(sp_matrix_<T, I> *A, size_t m, size_t n,
	size_t nnz, enum CBLAS_ORDER order);
template<typename T, typename I>
ok_status sp_matrix_free_(sp_matrix_<T, I> *A);
template<typename T, typename I>
ok_status sp_matrix_memcpy_mm_(sp_matrix_<T, I> *A,
	const sp_matrix_<T, I> *B);
template<typename T, typename I>
ok_status sp_matrix_memcpy_vals_mm_(sp_matrix_<T, I> *A,
	const sp_matrix_<T, I> *B);
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
typedef sp_matrix_<ok_float, ok_int> sp_matrix;
#else
typedef struct sp_matrix {
	size_t size1, size2, nnz, ptrlen;
	ok_float *val;
	ok_int *ind, *ptr;
	enum CBLAS_ORDER order;
} sp_matrix;
#endif

/* memory management */
ok_status sp_make_handle(void **sparse_handle);
ok_status sp_destroy_handle(void *sparse_handle);

ok_status sp_matrix_alloc(sp_matrix *A, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order);
ok_status sp_matrix_calloc(sp_matrix *A, size_t m, size_t n, size_t nnz,
	enum CBLAS_ORDER order);
ok_status sp_matrix_free(sp_matrix *A);

/* copy, I/O */
ok_status sp_matrix_memcpy_mm(sp_matrix *A, const sp_matrix *B);
ok_status sp_matrix_memcpy_ma(void *sparse_handle, sp_matrix *A,
        const ok_float *val, const ok_int *ind, const ok_int *ptr);
ok_status sp_matrix_memcpy_am(ok_float *val, ok_int *ind, ok_int *ptr,
        const sp_matrix *A);
ok_status sp_matrix_memcpy_vals_mm(sp_matrix *A, const sp_matrix *B);
ok_status sp_matrix_memcpy_vals_ma(void *sparse_handle, sp_matrix *A,
        const ok_float *val);
ok_status sp_matrix_memcpy_vals_am(ok_float *val, const sp_matrix *A);

/* elementwise, row-wise & column-wise math */
ok_status sp_matrix_abs(sp_matrix *A);
ok_status sp_matrix_pow(sp_matrix *A, const ok_float x);
ok_status sp_matrix_scale(sp_matrix *A, const ok_float alpha);
ok_status sp_matrix_scale_left(void *sparse_handle, sp_matrix *A,
	const vector *v);
ok_status sp_matrix_scale_right(void *sparse_handle, sp_matrix *A,
	const vector *v);

/* print */
ok_status sp_matrix_print(const sp_matrix *A);
ok_status sp_matrix_print_transpose(const sp_matrix *A);

/* matrix-vector multiplication */
ok_status sp_blas_gemv(void *sparse_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, sp_matrix *A, vector *x, ok_float beta, vector *y);

ok_status sp_blas_gemm(void *sparse_handle, enum CBLAS_TRANSPOSE transA,
        ok_float alpha, sp_matrix *A, matrix *X, ok_float beta, matrix *Y);

#ifdef __cplusplus
}
#endif

#endif  /* OPTKIT_LINSYS_SPARSE_H_ */
