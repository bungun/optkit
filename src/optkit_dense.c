#include "optkit_dense.h"

#ifdef __cplusplus
extern "C" {
#endif

void denselib_version(int * maj, int * min, int * change, int * status)
{
	* maj = OPTKIT_VERSION_MAJOR;
	* min = OPTKIT_VERSION_MINOR;
	* change = OPTKIT_VERSION_CHANGE;
	* status = (int) OPTKIT_VERSION_STATUS;
}

/*
 * VECTOR methods
 * ==============
 */

template<typename T>
inline int __vector_exists(vector<T> * v)
{
	if (v == OK_NULL) {
		printf("Error: cannot write to uninitialized vector<T> pointer\n");
		return 0;
	} else {
		return 1;
	}
}

template<typename T>
void vector_alloc(vector<T> * v, size_t n)
{
	if (!__vector_exists(v))
		return;
	v->size = n;
	v->stride = 1;
	v->data = (ok_float *) malloc(n * sizeof(T));
}

template<typename T>
void vector_calloc(vector<T> * v, size_t n)
{
	vector_alloc(v, n);
	memset(v->data, 0, n * sizeof(T));
}

template<typename T>
void vector_free(vector<T> * v)
{
	if (v->data != OK_NULL)
		ok_free(v->data);
	v->size = (size_t) 0;
	v->stride = (size_t) 0;
}

template<typename T>
inline void __vector_set(vector<T> * v, size_t i, T x)
{
        v->data[i * v->stride] = x;
}

template<typename T>
ok_float __vector_get(const vector<T> *v, size_t i)
{
	return v->data[i * v->stride];
}

template<typename T>
void vector_set_all(vector<T> * v, ok_float x)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		__vector_set(v, i, x);
}

template<typename T>
void vector_subvector(vector<T> * v_out, vector<T> * v_in, size_t offset, size_t n)
{
	if (!__vector_exists(v_out))
		return;
	v_out->size = n;
	v_out->stride = v_in->stride;
	v_out->data = v_in->data + offset * v_in->stride;
}

template<typename T>
void vector_view_array(vector<T> * v, T * base, size_t n)
{
	if (!__vector_exists(v))
		return;
	v->size = n;
	v->stride = 1;
	v->data = base;
}

template<typename T>
void vector_memcpy_vv(vector<T> * v1, const vector<T> * v2)
{
	uint i;
	if ( v1->stride == 1 && v2->stride == 1)
		memcpy(v1->data, v2->data, v1->size * sizeof(T));
	else
		for (i = 0; i < v1->size; ++i)
			__vector_set(v1, i, __vector_get(v2,i));
}

template<typename T>
void vector_memcpy_va(vector<T> * v, const T *y, size_t stride_y)
{
	uint i;
	if (v->stride == 1 && stride_y == 1)
		memcpy(v->data, y, v->size * sizeof(T));
	else
		for (i = 0; i < v->size; ++i)
			__vector_set(v, i, y[i * stride_y]);
}

template<typename T>
void vector_memcpy_av(T * x, const vector<T> * v, size_t stride_x)
{
	uint i;
	if (v->stride == 1 && stride_x == 1)
		memcpy(x, v->data, v->size * sizeof(T));
	else
		for (i = 0; i < v->size; ++i)
			x[i * stride_x] = __vector_get(v,i);
}

template<typename T>
void vector_print(const vector<T> * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		printf("%e ", __vector_get(v, i));
	printf("\n");
}

template<typename T>
void vector_scale(vector<T> * v, T x)
{
	CBLAS(scal)( (int) v->size, x, v->data, (int) v->stride);
}

template<typename T>
void vector_add(vector<T> * v1, const vector<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] += v2->data[i * v2->stride];
}

template<typename T>
void vector_sub(vector<T> * v1, const vector<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] -= v2->data[i * v2->stride];
}

template<typename T>
void vector_mul(vector<T> * v1, const vector<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] *= v2->data[i * v2->stride];
}

template<typename T>
void vector_div(vector<T> * v1, const vector<T> * v2)
{
	uint i;
	for (i = 0; i < v1->size; ++i)
		v1->data[i * v1->stride] /= v2->data[i * v2->stride];
}

template<typename T>
void vector_add_constant(vector<T> * v, const ok_float x)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] += x;
}

template<typename T>
void vector_abs(vector<T> * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(fabs)(v->data[i * v->stride]);
}

template<typename T>
void vector_recip(vector<T> * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = kOne / v->data[i * v->stride];
}

template<typename T>
void vector_safe_recip(vector<T> * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = ((ok_float)
			(v->data[i * v->stride] == kZero)) *
			kOne / v->data[i * v->stride];
}

template<typename T>
void vector_sqrt(vector<T> * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(sqrt)(v->data[i * v->stride]);
}

template<typename T>
void vector_pow(vector<T> * v, const ok_float x)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(pow)(v->data[i * v->stride], x);
}

template<typename T>
void vector_exp(vector<T> * v)
{
	uint i;
	for (i = 0; i < v->size; ++i)
		v->data[i * v->stride] = MATH(exp)(v->data[i * v->stride], x);

}

template<typename T>
size_t vector_indmin(vector<T> * v)
{
	T minval = static_cast<T>(OK_FLOAT_MAX); /* todo: type detection here */
	size_t minind = 0;
	size_t i;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : minval, minind)
	#endif
	for(i = 0; i < v->size; ++i)
	if(v->data[i] > minval) {
		minval = v->data[i];
		minind = i;
	}

	return i;
}

template<typename T>
T vector_min(vector<T> * v)
{
	T minval = static_cast<T>(OK_FLOAT_MAX); /* todo: type detection here */
	size_t i;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : minval)
	#endif
	for(i = 0; i < v->size; ++i)
	if(v->data[i] > minval)
		minval = v->data[i];

	return minval;
}

	T minval = static_cast<T>(OK_FLOAT_MAX); /* todo: type detection here */

template<typename T>
T vector_max(vector<T> * v)
{
	T maxval = -1 * static_cast<T>(OK_FLOAT_MAX); /* todo: type detection here */
	size_t i;

	#ifdef _OPENMP
	#pragma omp parallel for reduction(min : maxval)
	#endif
	for(i = 0; i < v->size; ++i)
	if(v->data[i] > maxval)
		maxval = v->data[i];

	return maxval;
}

/*
 * MATRIX methods
 * ==============
 */

template<typename T>
inline int __matrix_exists(matrix<T> * A)
{
	if (A == OK_NULL) {
	printf("Error: cannot write to uninitialized matrix pointer\n");
		return 0;
	} else {
		return 1;
	}
}

template<typename T>
void matrix_alloc(matrix<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	A->size1 = m;
	A->size2 = n;
	A->data = (ok_float *) malloc(m * n * sizeof(T));
	A->ld = (ord == CblasRowMajor) ? n : m;
	A->order = ord;
}

template<typename T>
void matrix_calloc(matrix<T> * A, size_t m, size_t n, enum CBLAS_ORDER ord)
{
	if (!__matrix_exists(A)) return;
	matrix_alloc(A, m, n, ord);
	memset(A->data, 0, m * n * sizeof(T));
}

template<typename T>
void matrix_free(matrix<T> * A)
{
	if (!__matrix_exists(A))
		return;
	if (A->data != OK_NULL)
		ok_free(A->data);
	A->size1 = (size_t) 0;
	A->size2 = (size_t) 0;
	A->ld = (size_t) 0;
}

template<typename T>
void matrix_submatrix(matrix<T> * A_sub, matrix<T> * A,
    size_t i, size_t j, size_t n1, size_t n2)
{
	if (!__matrix_exists(A_sub))
		return;
	if (!__matrix_exists(A))
		return;
	A_sub->size1 = n1;
	A_sub->size2 = n2;
	A_sub->ld = A->ld;
	A_sub->data = (A->order == CblasRowMajor) ?
		      A->data + (i * A->ld) + j :
		      A->data + i + (j * A->ld);
	A_sub->order = A->order;
}

template<typename T>
void matrix_row(vector<T> * row, matrix<T> * A, size_t i)
{
	if (!__vector_exists(row))
		return;
	if (!__matrix_exists(A))
		return;
	row->size = A->size2;
	row->stride = (A->order == CblasRowMajor) ? 1 : A->ld;
	row->data = (A->order == CblasRowMajor) ?
		    A->data + (i * A->ld) :
		    A->data + i;
}

template<typename T>
void matrix_column(vector<T> * col, matrix<T> *A, size_t j)
{
	if (!__vector_exists(col))
		return;
	if (!__matrix_exists(A))
		return;
	col->size = A->size1;
	col->stride = (A->order == CblasRowMajor) ? A->ld : 1;
	col->data = (A->order == CblasRowMajor) ?
		    A->data + j :
		    A->data + (j * A->ld);
	}

template<typename T>
void matrix_diagonal(vector<T> * diag, matrix<T> * A)
{
	if (!__vector_exists(diag))
		return;
	if (!__matrix_exists(A))
		return;
	diag->data = A->data;
	diag->stride = A->ld + 1;
	diag->size = (size_t) (A->size1 <= A->size2) ? A->size1 : A->size2;
}

template<typename T>
void matrix_cast_vector(vector<T> * v, matrix<T> * A)
{
	if (!__vector_exists(v))
		return;
	if (!__matrix_exists(A))
		return;
	v->size = A->size1 * A->size2;
	v->stride = 1;
	v->data = A->data;
}

template<typename T>
void matrix_view_array(matrix<T> * A, const T *base, size_t n1, size_t n2,
        enum CBLAS_ORDER ord)
{
	if (!__matrix_exists(A))
		return;
	A->size1 = n1;
	A->size2 = n2;
	A->data = (T *) base;
	A->ld = (ord == CblasRowMajor) ? n2 : n1;
	A->order = ord;
}

template<typename T>
T __matrix_get_colmajor(const matrix<T> * A, size_t i, size_t j)
{
	return A->data[i + j * A->ld];
}

template<typename T>
T __matrix_get_rowmajor(const matrix<T> * A, size_t i, size_t j)
{
	return A->data[i * A->ld + j];
}

template<typename T>
void __matrix_set_rowmajor(matrix<T> * A, size_t i, size_t j, T x)
{
	A->data[i * A->ld + j] = x;
}

template<typename T>
void __matrix_set_colmajor(matrix<T> * A, size_t i, size_t j, T x)
{
	A->data[i + j * A->ld] = x;
}

template<typename T>
void matrix_set_all(matrix<T> * A, T x)
{
	size_t i, j;
	if (!__matrix_exists(A))
		return;

	if (A->order == CblasRowMajor)
		for (i = 0; i < A->size1; ++i)
			for (j = 0; j < A->size2; ++j)
				__matrix_set_rowmajor(A, i, j, x);
	else
		for (j = 0; j < A->size2; ++j)
			for (i = 0; i < A->size1; ++i)
				__matrix_set_colmajor(A, i, j, x);
}

template<typename T>
void matrix_memcpy_mm(matrix<T> * A, const matrix<T> * B)
{
	uint i, j;
	if (A->size1 != B->size1) {
		printf("error: m-dimensions must match for matrix memcpy\n");
		return;
	} else if (A->size2 != B->size2) {
		printf("error: n-dimensions must match for matrix memcpy\n");
		return;
	}

	void (* mset)(matrix<T> * M, size_t i, size_t j, ok_float x) =
		(A->order == CblasRowMajor) ?
		__matrix_set_rowmajor :
		__matrix_set_colmajor;

	T (* mget)(const matrix<T> * M, size_t i, size_t j) =
		(B->order == CblasRowMajor) ?
		__matrix_get_rowmajor :
		__matrix_get_colmajor;

	for (i = 0; i < A->size1; ++i)
		for (j = 0; j < A->size2; ++j)
			  mset(A, i, j, mget(B, i , j));
}

template<typename T>
void matrix_memcpy_ma(matrix<T> * A, const T * B, const enum CBLAS_ORDER ord)
{
	uint i, j;
	void (* mset)(matrix<T> * M, size_t i, size_t j, T x) =
		(A->order == CblasRowMajor) ?
		__matrix_set_rowmajor :
		__matrix_set_colmajor;

	if (ord == CblasRowMajor)
		for (i = 0; i < A->size1; ++i)
			for (j = 0; j < A->size2; ++j)
				mset(A, i, j, B[i * A->size2 + j]);
	else
		for (i = 0; i < A->size1; ++i)
			for (j = 0; j < A->size2; ++j)
				mset(A, i, j, B[i + j * A->size1]);
}

template<typename T>
void matrix_memcpy_am(T * A, const matrix<T> * B, const enum CBLAS_ORDER ord)
{
	uint i, j;
	T (* mget)(const matrix<T> * M, size_t i, size_t j) =
		(B->order == CblasRowMajor) ?
		__matrix_get_rowmajor :
		__matrix_get_colmajor;

	if (ord == CblasRowMajor)
		for (i = 0; i < B->size1; ++i)
			for (j = 0; j < B->size2; ++j)
				A[i * B->size2 + j] = mget(B, i, j);
	else
		for (j = 0; j < B->size2; ++j)
			for (i = 0; i < B->size1; ++i)
				A[i + B->size1 * j] = mget(B, i, j);
}

template<typename T>
void matrix_print(matrix<T> * A)
{
	uint i, j;
	T (* mget)(const matrix<T> * M, size_t i, size_t j) =
		(A->order == CblasRowMajor) ?
		__matrix_get_rowmajor : __matrix_get_colmajor;

	for (i = 0; i < A->size1; ++i) {
		for (j = 0; j < A->size2; ++j)
			printf("%e ", mget(A, i, j));
		printf("\n");
	}
	printf("\n");
}

template<typename T>
void matrix_scale(matrix<T> *A, T x)
{
	size_t i;
	vector<T> row_col = (vector){0, 0, OK_NULL};
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			vector_scale(&row_col, x);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			vector_scale(&row_col, x);
		}
}

template<typename T>
void matrix_scale_left(matrix<T> * A, const vector<T> * v)
{
	size_t i;
	vector<T> row = (vector){0, 0, OK_NULL};
	for(i = 0; i < A->size1; ++i) {
		matrix_row(&row, A, i);
		vector_scale(&row, v->data[i]);
	}
}

template<typename T>
void matrix_scale_right(matrix<T> * A, const vector<T> * v)
{
	size_t i;
	vector<T> col = (vector){0, 0, OK_NULL};
	for(i = 0; i < A->size2; ++i) {
		matrix_column(&col, A, i);
		vector_scale(&col, v->data[i]);
	}
}

template<typename T>
void matrix_abs(matrix<T> * A)
{
	size_t i;
	vector<T> row_col = (vector){0, 0, OK_NULL};
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			vector_abs(&row_col);
		}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			vector_abs(&row_col);
		}
}

template<typename T>
void matrix_pow(matrix<T> * A, const ok_float x)
{
	size_t i;
	vector<T> row_col = (vector){0, 0, OK_NULL};
	if (A->order == CblasRowMajor)
		for(i = 0; i < A->size1; ++i) {
			matrix_row(&row_col, A, i);
			vector_pow(&row_col, x);
	}
	else
		for(i = 0; i < A->size2; ++i) {
			matrix_column(&row_col, A, i);
			vector_pow(&row_col, x);
	}
}

template<typename T>
int __matrix_order_compat(const matrix<T> * A, const matrix<T> * B,
	const char * nm_A, const char * nm_B, const char * nm_routine)
{
	if (A->order == B->order)
		return 1;

	printf("OPTKIT ERROR (%s) matrices %s and %s must have same layout.\n",
		nm_routine, nm_A, nm_B);
	return 0;
}

/*
 * BLAS routines
 * =============
 */
ok_status blas_make_handle(void ** linalg_handle)
{
	*linalg_handle = OK_NULL;
	return OPTKIT_SUCCESS;
}

ok_status blas_destroy_handle(void * linalg_handle)
{
	linalg_handle = OK_NULL;
	return OPTKIT_SUCCESS;
}

/* BLAS LEVEL 1 */

template<typename T>
void blas_axpy(void * linalg_handle, T alpha, const vector<T> *x, vector<T> *y)
{
	CBLAS(axpy)((int) x->size, alpha, x->data, (int) x->stride, y->data,
		(int) y->stride);
}

template<typename T>
T blas_nrm2(void * linalg_handle, const vector<T> *x)
{
	return CBLAS(nrm2)((int) x->size, x->data, (int) x->stride);
}

template<typename T>
void blas_scal(void * linalg_handle, const ok_float alpha, vector<T> *x)
{
	CBLAS(scal)((int) x->size, alpha, x->data, (int) x->stride);
}

template<typename T>
ok_float blas_asum(void * linalg_handle, const vector<T> * x)
{
	return CBLAS(asum)((int) x->size, x->data, (int) x->stride);
}

template<typename T>
blas_dot(void * linalg_handle, const vector<T> * x, const vector<T> * y)
{
	return CBLAS(dot)((int) x->size, x->data, (int) x->stride, y->data,
		(int) y->stride);
}

template<typename T>
void blas_dot_inplace(void * linalg_handle, const vector<T> * x, const vector<T> * y,
	ok_float * deviceptr_result)
{
	*deviceptr_result = CBLAS(dot)((int) x->size, x->data, (int) x->stride,
		y->data, (int) y->stride);
}

/* BLAS LEVEL 2 */

template<typename T>
void blas_gemv(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	ok_float alpha, const matrix<T> *A, const vector<T> *x, ok_float beta,
	vector<T> *y)
{
	CBLAS(gemv)(A->order, transA, (int) A->size1, (int) A->size2, alpha,
		A->data, (int) A->ld, x->data, (int) x->stride, beta,
		y->data, (int) y->stride);
}

template<typename T>
void blas_trsv(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag, const matrix<T> *A,
	vector<T> *x)
{
	CBLAS(trsv)(A->order, uplo, transA, Diag, (int) A->size1, A->data,
		(int) A->ld, x->data, (int) x->stride);
}

template<typename T>
void blas_sbmv(void * linalg_handle, enum CBLAS_ORDER order,
	enum CBLAS_UPLO uplo, const size_t num_superdiag, const T alpha,
	const vector<T> * vecA, const vector<T> * x, const T beta,
	vector<T> * y)
{
	CBLAS(sbmv)(order, uplo, (int) y->size, (int) num_superdiag, alpha,
		vecA->data, (int) num_superdiag + 1, x->data, (int) x->stride,
		beta, y->data, (int) y->stride);
}

template<typename T>
void blas_diagmv(void * linalg_handle, const ok_float alpha,
	const vector<T> * vecA, const vector<T> * x, const ok_float beta,
	vector<T> * y)
{
	blas_sbmv(linalg_handle, CblasColMajor, CblasLower, 0, alpha, vecA, x,
		beta, y);
}

/* BLAS LEVEL 3 */

template<typename T>
void blas_syrk(void * linalg_handle, enum CBLAS_UPLO uplo,
	enum CBLAS_TRANSPOSE transA, T alpha, const matrix<T> * A, T beta,
	matrix<T> * C)
{
	const int k = (transA == CblasNoTrans) ? (int) A->size2 :
						 (int) A->size1;
	void (* syrk)(enum CBLAS_ORDER o, enum CBLAS_UPLO u,
		enum CBLAS_TRANSPOSE tA, T alpha, T* A, T beta, T* B, T* C);

	if (!( __matrix_order_compat(A, C, "A", "C", "blas_syrk") ))
		return;

	syrk = (T == float)

	CBLAS(syrk)(A->order, uplo, transA, (int) C->size2, k, alpha, A->data,
		(int) A->ld, beta, C->data, (int) C->ld);
}

template<typename T>
void blas_gemm(void * linalg_handle, enum CBLAS_TRANSPOSE transA,
	enum CBLAS_TRANSPOSE transB, T alpha, const matrix<T> * A,
	const matrix<T> * B, T beta, matrix<T> * C)
{
	const int NA = (transA == CblasNoTrans) ? (int) A->size2 :
						  (int) A->size1;

	if (!( __matrix_order_compat(A, B, "A", "B", "gemm") &&
		__matrix_order_compat(A, C, "A", "C", "blas_gemm") ))
		return;
	CBLAS(gemm)(A->order, transA, transB, (int) C->size1, (int) C->size2,
		NA, alpha, A->data, (int) A->ld, B->data, (int) B->ld,
		beta, C->data, (int) C->ld);
}

template<typename T>
void blas_trsm(void * linalg_handle, enum CBLAS_SIDE Side,
	enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE transA, enum CBLAS_DIAG Diag,
	T alpha, const matrix<T> *A, matrix<T> *B)
{
	if (!( __matrix_order_compat(A, B, "A", "B", "blas_trsm") ))
		return;
	CBLAS(trsm)(A->order, Side, uplo, transA, Diag, (int) B->size1,
		(int) B->size2, alpha, A->data,(int) A->ld, B->data,
		(int) B->ld);
}

/*
 * LINEAR ALGEBRA routines
 * =======================
 */

/* Non-Block Cholesky. */
template<typename T>
void __linalg_cholesky_decomp_noblk(void * linalg_handle, matrix<T> *A) {
	T l11;
	matrix l21, a22;
	size_t n = A->size1, i;

	/* get order-specific matrix getter/setter */
	void (* mset)(matrix<T> * M, size_t i, size_t j, ok_float x) =
		(A->order == CblasRowMajor) ?
		__matrix_set_rowmajor : __matrix_set_colmajor;
	T (* mget)(const matrix<T> * M, size_t i, size_t j) =
		(A->order == CblasRowMajor) ?
		__matrix_get_rowmajor : __matrix_get_colmajor;

	l21= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};
	a22= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};

	for (i = 0; i < n; ++i) {
		/* L11 = sqrt(A11) */
		l11 = static_cast<T>(MATH(sqrt)(mget(A, i, i)));
		mset(A, i, i, l11);

		if (i + 1 == n)
			break;

		/* L21 = A21 / L11 */
		matrix_submatrix(&l21, A, i + 1, i, n - i - 1, 1);
		matrix_scale(&l21, kOne / l11);

		/* A22 -= L12*L12'*/
		matrix_submatrix(&a22, A, i + 1, i + 1, n - i - 1, n - i - 1);
		blas_syrk(linalg_handle, CblasLower, CblasNoTrans, -kOne, &l21,
			kOne, &a22);
	}
}

/*
 * Block Cholesky.
 *   l11 l11^T = a11
 *   l21 = a21 l11^(-T)
 *   a22 = a22 - l21 l21^T
 *
 * Stores result in Lower triangular part.
 */
template<typename T>
void linalg_cholesky_decomp(void * linalg_handle, matrix<T> * A)
{
	matrix L11, L21, A22;
	size_t n = A->size1, blk_dim, i, n11;

	L11= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};
	L21= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};
	A22= (matrix){0, 0, 0, OK_NULL, CblasRowMajor};

	/* block dimension borrowed from Eigen. */
	blk_dim = ((n / 128) * 16) < 8 ? (n / 128) * 16 : 8;
	blk_dim = blk_dim > 128 ? blk_dim : 128;

	for (i = 0; i < n; i += blk_dim) {
		n11 = blk_dim < n - i ? blk_dim : n - i;

		/* L11 = chol(A11) */
		matrix_submatrix(&L11, A, i, i, n11, n11);
		__linalg_cholesky_decomp_noblk(linalg_handle, &L11);

		if (i + blk_dim >= n)
			break;

                /* L21 = A21 L21^-T */
		matrix_submatrix(&L21, A, i + n11, i, n - i - n11, n11);
		blas_trsm(linalg_handle, CblasRight, CblasLower, CblasTrans,
			CblasNonUnit, kOne, &L11, &L21);

		/* A22 -= L21*L21^T */
		matrix_submatrix(&A22, A, i + blk_dim, i + blk_dim,
			n - i - blk_dim, n - i - blk_dim);

		blas_syrk(linalg_handle, CblasLower, CblasNoTrans, -kOne, &L21,
			kOne, &A22);
	}
}

/* Cholesky solve */
template<typename T>
void linalg_cholesky_svx(void * linalg_handle, const matrix<T> * L, vector<T> * x)
{
	blas_trsv(linalg_handle, CblasLower, CblasNoTrans, CblasNonUnit, L, x);
	blas_trsv(linalg_handle, CblasLower, CblasTrans, CblasNonUnit, L, x);
}

/* device reset */
ok_status ok_device_reset()
{
	return OPTKIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
